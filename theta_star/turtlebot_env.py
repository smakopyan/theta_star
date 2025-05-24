import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData    
from sensor_msgs.msg import LaserScan, Image
import math
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import cv2
from theta_star import ThetaStar
from theta_star_old import ThetaStar as Old
import matplotlib.pyplot as plt
from collections import deque
import logging
import os
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import time
from scipy.ndimage import distance_transform_edt

logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def slam_to_grid_map(slam_map, threshold=200, expansion_size = 5, resolution = 0.05):
    grid_map = np.where(slam_map < threshold, 1, 0)  
    wall = np.where(grid_map == 1)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,grid_map.shape[0]-1)
            y = np.clip(y,0,grid_map.shape[1]-1)
            grid_map[x,y] = 1

    return grid_map
    
def grid_to_world(x_grid, y_grid, map_shape, map_resolution = 0.05, map_origin = (-7.76,-7.15), viz = False):

    y_map = map_shape[0] - y_grid - 1
    x_map = x_grid
    x_w = map_origin[0] + x_map * map_resolution
    if viz:
        y_w = map_origin[1] + y_grid * map_resolution
    else:
        y_w = map_origin[1] + y_map * map_resolution
    return x_w, y_w

def world_to_map(world_coords, resolution, origin, map_offset, map_shape):
    """
    Преобразует мировые координаты в пиксельные, учитывая смещение
    и возможный переворот карты SLAM.

    Параметры:
      world_coords: (x_world, y_world) или массив (две компоненты)
      resolution: масштаб (размер одного пикселя в мировых единицах)
      origin: мировые координаты начала карты (нижний левый угол SLAM)
      map_offset: (offset_x, offset_y) – сдвиг, чтобы центрировать карту
      map_shape: (map_height, map_width) – размеры карты в пикселях
    
    Возвращает:
      (x_map, y_map): координаты в пиксельной системе
    """
    x_world, y_world = world_coords

    # Обработка массивов и одиночных значений
    if isinstance(x_world, np.ndarray):
        x_map = ((x_world - origin[0]) / resolution).astype(int) 
        y_map = ((y_world - origin[1]) / resolution).astype(int) 
    else:
        x_map = int((x_world - origin[0]) / resolution) 
        y_map = int((y_world - origin[1]) / resolution) 

    # Переворачиваем Y, если SLAM-карта инвертирована
    if isinstance(y_map, np.ndarray):
        y_map = map_shape[0] - y_map - 1
        x_map = np.clip(x_map, 0, map_shape[1] - 1)
        y_map = np.clip(y_map, 0, map_shape[0] - 1)
    else:
        y_map = map_shape[0] - y_map - 1
        x_map = max(0, min(x_map, map_shape[1] - 1))
        y_map = max(0, min(y_map, map_shape[0] - 1))

    return x_map, y_map



def euler_from_quaternion(x, y , z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw
# -------------------------------
# Функция для расчёта отклонения от оптимального пути
# -------------------------------
def compute_deviation_from_path(current_pos, optimal_path):
    if optimal_path:
        path_points = np.array(optimal_path)
        distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
        min_distance = np.min(distances)
        return min_distance
    return np.inf

def generate_potential_field(grid_map, goal, path_points, resolution = 0.05, 
                             k_att=100.0, k_rep=300.0, d0=80.0, scale = 0.07, normalize = False):
    """
    Генерация потенциального поля:
      - Квадратичное притяжение к цели и оптимальному пути.
      - Квадратичное отталкивание от препятствий в радиусе d0.
    """
    h, w = grid_map.shape
    # подготовим маски
    # 1) препятствия
    obstacle_mask = (grid_map == 1)
    # 2) оптимальный путь: построим булевую карту
    path_mask = np.zeros_like(grid_map, dtype=bool)
    for pt in path_points:
        x_pix, y_pix = pt
        path_mask[y_pix, x_pix] = True
    goal_mask = np.zeros_like(grid_map, dtype=bool) 
    gx, gy = goal
    goal_mask[gy, gx] = True

    # 1) расстояние до препятствий
    dist_obs = distance_transform_edt(~obstacle_mask) 
    # 2) расстояние до пути
    dist_path = distance_transform_edt(~path_mask) 
    # 3) расстояние до цели: сделаем одну точку
    dist_goal = distance_transform_edt(~goal_mask) 

    # ---------- поля ----------
    # квадратичное притяжение к цели (отрицательное)
    att_goal = - 0.5 * k_att * np.exp(-scale * dist_goal) 
    
    # квадратичное притяжение к пути
    att_path = - 0.5 * k_att * np.exp(-scale * dist_path) 

    # отталкивающее поле (квадратичное), только внутри d0
    rep_field = np.zeros_like(grid_map, dtype=np.float32) 
    # для каждой клетки, где dist_obs <= d0: 
    mask = (dist_obs <= d0) & (dist_obs > 0)
    inv = 1.0 / (dist_obs[mask] + 1e-5)
    rep_field[mask] = 0.5 * k_rep * (inv - 1.0/d0)**2

    # объединяем
    field = rep_field + att_goal + att_path

    # по желанию нормализуем на [-1,1] или 0..1
    if normalize:
        mn, mx = field.min(), field.max()
        field = 2*(field - mn)/(mx - mn) - 1  # теперь в [-1..1]

    return field
class Robot():
    def __init__(self, namespace, state_pos, goal, grid_map):
        self.namespace = namespace
        self.cmd_vel_pub = None
        self.occupation_pub = None
        self.obstacle_detected = False
        self.bridge = CvBridge()
        self.camera_obstacle_detected = False
        self.lidar_obstacle_detected = False
        self.path_marker_pub = None
        self.wp_marker_pub = None

        
        self.target_x = goal[0]
        self.target_y = goal[1]
        self.goal = goal
        
        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.state_pose = state_pos

        self.grid_map = None
        self.camera_history = deque(maxlen=20)  

        self.optimal_path = None
        self.potential_field = None
        self.prev_potential = 0 
        self.prev_x = None  # Предыдущая координата X
        self.prev_y = None  # Предыдущая координата Y
        self.obstacle_count = 0  
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacles = []
        self.prev_distance = None
        self.past_distance = 0
        # self.max_steps = 250
        self.steps = 0 
        self.recent_obstacles = []
        self.state = np.array([])
        self.reward = 0.0
        self.done = False
        self.min_obstacle_dist = 0
        self.penalty = 0.0
        self.last_waypoint_idx = 5
        self.beta_spin = 0.1
        self.linear_velocity = 0.
        self.angular_velocity = 0.
        self.velocity_vector = np.array([])
        self.occupation_map = np.zeros_like(grid_map, dtype=np.float32)
        self.nearest_wp = None
        self.world_path = None
        self.done = False

class RunningStat(object):
    def __init__(self,shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x-oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

        
class Zfilter:
    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter
    def __call__(self, x, **kwargs):
        self.prev_filter = x
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean 
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        if self.prev_filter:
            self.prev_filter.reset()




class TurtleBotEnv(Node, gym.Env):
    def __init__(self):
        super().__init__('turtlebot_env')
        self.num_robots = 2
        spawn_points = [[2.0, -5.0], [2.0, -2.0]]
        # goals = [[0.483287, -2.73528],[0.583933, -3.66662] ]
        goals = [[3.0, -2.0], [3.0, -5.0]]


        slam_map = cv2.imread(os.path.join(get_package_share_directory('theta_star'),
                                           'maps','map3.pgm'), cv2.IMREAD_GRAYSCALE)
        
  
        self.grid_map = slam_to_grid_map(slam_map)
        self.robots = [Robot(f"tb{i}", spawn_points[i], goals[i], self.grid_map) for i in range(self.num_robots)]

        self.map_initialized = False
        self.penalty_map = np.ones_like(self.grid_map, dtype=np.float32)
        self.max_steps = 250
        self.reset_world = self.create_client(Empty, '/reset_world')
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.step_ = 0
        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.gridmap_pub = self.create_publisher(OccupancyGrid, '/grid_costmap', qos_profile)
        for robot in self.robots:
            robot.cmd_vel_pub = self.create_publisher(Twist, f'/{robot.namespace}/cmd_vel', 10)
            robot.occupation_pub = self.create_publisher(OccupancyGrid, f'/{robot.namespace}/occupation_costmap', qos_profile)
            
            robot.path_marker_pub = self.create_publisher(
                Marker, 
                f'/{robot.namespace}/path_marker', 
                10
            )
            robot.wp_marker_pub = self.create_publisher(
                Marker, 
                f'/{robot.namespace}/wp_marker', 
                10
            )
            self.create_subscription(
                Odometry, 
                f'/{robot.namespace}/odom', 
                self.create_odom_callback(robot),

                10
            )
            self.create_subscription(
                LaserScan,
                f'/{robot.namespace}/scan',
                self.create_scan_callback(robot),
                10
            )

            self.create_subscription(
                Image,
                f'/{robot.namespace}/camera/image_raw',
                self.create_camera_callback(robot),
                10
            )
            other = self.robots[1] if robot.namespace == 'tb0' else self.robots[0] 
            optimal_path_ = self.path(robot, self.grid_map, other.occupation_map, self.penalty_map)
            
            # robot.optimal_path = optimal_path_
            robot.optimal_path = self.interpolate_path(optimal_path_)
            # print(robot.optimal_path)
            robot.potential_field = generate_potential_field(self.grid_map, world_to_map(robot.goal, 0.05, (-7.76,-7.15), (0, 0), self.grid_map.shape), robot.optimal_path)
            # self.show_potential_field(robot) 
            self.goal_reach_bonus = 15.0  
            self.inactivity_penalty = -0.5 
            self.min_goal_dist = 0.3  
                    
        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.action_space = spaces.Box(low=np.array([0.05, -0.82]), high=np.array([0.26, 0.82]), dtype=np.float32)
        self.wp_step = 5
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(9,),  
            dtype=np.float32
        )                       

        self.timer = self.create_timer(0.05, self._timer_callback)
        self.costmap_timer = self.create_timer(1.0, self.publish_costmaps)

    
    def path(self, robot, grid_map, occupation_map, penalty_map,  map_resolution = 0.05, map_origin = (-7.76,-7.15)):
        # spawn_points = [[2.0, -5.0], [2.0, -2.0]]
        state_pixel = world_to_map([robot.current_x, robot.current_y + 0.02], 0.05, (-7.76,-7.15), (0,0), grid_map.shape)
        goal_world_x, goal_world_y  = robot.goal
        goal_pixel = world_to_map([goal_world_x, goal_world_y], 0.05, (-7.76,-7.15), (0,0), grid_map.shape)

        map_offset = (0, 0)  # Смещение координат

        map_shape = grid_map.shape  # (высота, ширина) карты

        theta_star = ThetaStar()
        # start = time.time()
        optimal_path = theta_star.plan((state_pixel[1], state_pixel[0]), 
                                (goal_pixel[1], goal_pixel[0]),
                                grid_map, occupation_map, penalty_map)
        # end = time.time() - start
        # print("new: ", end)

        if optimal_path is None:
            # print("Путь не найден")
            return []
        else:
            # print("Найденный путь:")
            # print(optimal_path)
            return optimal_path


    def _timer_callback(self):
        # pass
        self.publish_grid_map()
        for robot in self.robots:
            path_= self.get_path(robot.optimal_path)
            self.visualize_path(robot, path_)
            self.publish_occupation_map(robot)

    def get_path(self, opt_path):
        path = [grid_to_world(i[0], i[1], map_shape = self.grid_map.shape, viz = True) for i in opt_path]
        return path

    def get_observations(self, dones):
        observations = []
        for robot in self.robots:
            x, y = world_to_map(
                (robot.current_x, robot.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )
            other = self.robots[1] if robot.namespace == 'tb0' else self.robots[0]
            robots_dist = math.sqrt((other.current_x - robot.current_x)**2 + (other.current_y - robot.current_y)**2)

            dynamic_cost = other.occupation_map[y][x]
            penalty = self.penalty_map[y][x]
            distance = math.sqrt((robot.target_x - robot.current_x) ** 2 + (robot.target_y - robot.current_y) ** 2)
            angle_to_goal = math.atan2(robot.target_y - robot.current_y, robot.target_x - robot.current_x)
            angle_diff = (angle_to_goal - robot.current_yaw + np.pi) % (2 * np.pi) - np.pi
            min_obstacle_dist = 3.5 if not robot.obstacles else min(robot.obstacles)
            wp_distance = math.sqrt((robot.nearest_wp[0] - robot.current_x) ** 2 + (robot.nearest_wp[1] - robot.current_y) ** 2)
            dx_goal = (robot.target_x - robot.current_x)
            dy_goal = (robot.target_y - robot.current_y)
            
            if robot.nearest_wp:
                dx_wp = (robot.nearest_wp[0] - robot.current_x)
                dy_wp = (robot.nearest_wp[1] - robot.current_y)
            else:
                dx_wp, dy_wp = 0.0, 0.0
            obs = np.array([robot.current_x, robot.current_y, angle_diff, min_obstacle_dist, 2*(dynamic_cost + penalty), dx_goal, dy_goal, dx_wp, dy_wp])
            
            observations.append(obs)
        return np.array(observations)
    
    def interpolate_path(self, raw_path, step=2):
        if not raw_path or len(raw_path) < 2:
            return []
        interpolated = []
        for i in range(len(raw_path)-1):
            start = np.array(raw_path[i])
            end = np.array(raw_path[i+1])
            distance = np.linalg.norm(end - start)
            num_points = int(distance / 1)
            
            for t in np.linspace(0, 1, num_points):
                point = start * (1 - t) + end * t
                interpolated.append([point[1], point[0]])
        interpolated = [[int(round(x)), int(round(y))] for (x, y) in interpolated]
        new_path = [[p[1], p[0]] for p in raw_path]
        # print(interpolated)
        return interpolated

    def create_odom_callback(self, robot):
        def callback(msg):
            robot.current_x = msg.pose.pose.position.x
            robot.current_y = msg.pose.pose.position.y
            orientation_q = msg.pose.pose.orientation
            x, y = world_to_map(
                (robot.current_x, robot.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )
            robot.linear_velocity = msg.twist.twist.linear.x
            robot.angular_velocity = msg.twist.twist.angular.z
            robot.velocity_vector = np.array([
                robot.linear_velocity * np.cos(robot.current_yaw),
                robot.linear_velocity * np.sin(robot.current_yaw)
            ])
            height, width = self.grid_map.shape

            robot.occupation_map[y][x] += 0.3
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        robot.occupation_map[ny][nx] += 0.3 
            robot.current_yaw = euler_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)

            robot.current_yaw = euler_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        return callback

    def create_scan_callback(self, robot):
        def callback(msg):
            raw_obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max 
                            else msg.range_max for r in msg.ranges]

            robot.obstacles = raw_obstacles
            min_obstacle_dist = min(raw_obstacles) if raw_obstacles else msg.range_max
            logger.info(f'Min obstacle dist: {min_obstacle_dist}') 

            # Конвертация координат
            current_x, current_y = world_to_map(
                (robot.current_x, robot.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )

            logger.info(f'Current_x, current_y: {current_x, current_y}') 
            # Получаем значение поля
            if 0 <= current_x < self.grid_map.shape[1] and 0 <= current_y < self.grid_map.shape[0]:
                potential_value = robot.potential_field[current_y, current_x]
                # print(f'Potential value: {potential_value}')
                logger.info(f'Potential value: {potential_value}') 
            else:
                potential_value = 1

            # Проверяем, было ли препятствие на большинстве последних кадров
            camera_obstacle_count = sum(robot.camera_history)  # Считаем количество True
            camera_obstacle_threshold = 12  

            # Если LiDAR видит препятствие вблизи
            if min_obstacle_dist < 0.2:
                # Принудительно увеличиваем потенциал в случае обнаружения препятствия
                potential_value = max(potential_value, np.percentile(robot.potential_field, 90))
        
            # Логика определения препятствий
            robot.lidar_obstacle_detected = (
                (min_obstacle_dist < 0.2) and (  # Лидар обнаружил близкое препятствие И
                    (potential_value > 1) or  # Более чувствительный порог
                    # (potential_value > np.percentile(self.potential_field, 90)) or  # Высокий потенциал
                    (camera_obstacle_count >= camera_obstacle_threshold)  # Камера часто видела препятствие
                )
            )
            robot.recent_obstacles.append(robot.lidar_obstacle_detected)
            if len(robot.recent_obstacles) > 5:  # Храним только 5 последних значений
                robot.recent_obstacles.pop(0)

            # Если хотя бы 3 из 5 последних измерений показали препятствие — считаем его подтверждённым
            if sum(robot.recent_obstacles) >= 3:
                robot.lidar_obstacle_detected = True

            logger.info(f'Detect of obstacle: {robot.lidar_obstacle_detected}') 
        return callback
    
    def create_camera_callback(self, robot):
        def callback(msg):
            try:
                cv_image = robot.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                if cv_image is not None:
                    detected = self.process_camera_image(cv_image)
                    robot.camera_obstacle_detected = detected
                    robot.camera_history.append(detected)  # Добавляем в историю
                else:
                    robot.camera_obstacle_detected = False
                    robot.camera_history.append(False)
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")
                robot.camera_obstacle_detected = False
                robot.camera_history.append(False)
        return callback
        
    def show_potential_field(self, robot):

        goal_pixel = world_to_map(robot.goal, resolution=0.05, origin=(-7.76, -7.15), map_offset=(0, 0),map_shape=self.grid_map.shape)
        # plt.figure(figsize=(10, 8))
        # plt.imshow(robot.potential_field, cmap='jet')
        # plt.colorbar(label='Potential')
        # plt.scatter(goal_pixel[0], goal_pixel[1], c='green', s=200, marker='*', label='Goal')
        # plt.title("Potential Field Visualization")
        # plt.legend()
        # plt.show()

    def process_camera_image(self, cv_image):
   
        # Преобразование изображения в формат для обработки
        pixel_values = cv_image.reshape((-1, 3))  # Преобразование в список пикселей
        pixel_values = np.float32(pixel_values)

        # Применение K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 2  # Количество кластеров (фон и препятствие)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Преобразование обратно в изображение
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(cv_image.shape)

        # Анализ кластеров
        obstacle_detected = np.count_nonzero(labels == 1) > 210000 # Если пикселей кластеров > чего-то, то препятствие обнаружено
        # print(obstacle_detected)
        return obstacle_detected
    
    def compute_potential_reward(self, intermediate_points, obstacle_detected, robot, k_att=10.0, k_rep=30.0, d0=5.0, lam=0.5):
        current_x = robot.state[0]
        current_y = robot.state[1]
        angle_diff = robot.state[2]
        min_obstacle_dist = robot.state[3]
        # Преобразуем координаты в пиксельные
        current_x, current_y = world_to_map(
            (robot.current_x, robot.current_y),
            resolution=0.05,
            origin=(-7.76, -7.15),
            map_offset=(0, 0),
            map_shape=self.grid_map.shape
        )

        goal_x, goal_y = world_to_map(robot.goal,
            resolution=0.05,
            origin=(-7.76, -7.15),
            map_offset=(0, 0),
            map_shape=self.grid_map.shape
        )
        potential_value = robot.potential_field[current_y, current_x] 
        delta_potential = robot.prev_potential - potential_value
        R_potential = np.clip(-delta_potential, -1.0, 1.0)
        robot.prev_potential = potential_value

        # === Притяжение к промежуточной точке ===
        R_intermediate = 0.0
        if robot.prev_x is None or robot.prev_y is None:
            robot.prev_x, robot.prev_y = current_x, current_y

        if intermediate_points:
            nearest_idx = np.argmin([np.linalg.norm([current_x-p[0], current_y-p[1]]) 
                            for p in intermediate_points])
            nearest_intermediate = min(intermediate_points, key=lambda p: np.linalg.norm([current_x - p[0], current_y - p[1]]))
            prev_dist = np.linalg.norm([robot.prev_x - nearest_intermediate[0], robot.prev_y - nearest_intermediate[1]])
            curr_dist = np.linalg.norm([current_x - nearest_intermediate[0], current_y - nearest_intermediate[1]])
            R_intermediate = np.clip(k_att * (prev_dist - curr_dist), -1.0, 1.0)

        R_intermediate = np.clip(k_att * (prev_dist - curr_dist), -1.0, 1.0)

        # === Градиент к цели ===
        motion_direction = np.array([np.cos(robot.current_yaw), np.sin(robot.current_yaw)])
        direction_to_goal = np.array([goal_x - current_x, goal_y - current_y])
        norm = np.linalg.norm(direction_to_goal)
        direction_to_goal = direction_to_goal / norm if norm > 0 else np.array([1.0, 0.0])
        projection = np.dot(motion_direction, direction_to_goal)
        grad_reward = np.clip(lam * projection, -1.0, 1.0)

        if robot.lidar_obstacle_detected and projection < 0:
            grad_reward -= 2.0

        # === Отталкивающее поле ===
        R_repulsive = 0.0
        if min_obstacle_dist < d0 and obstacle_detected:
            R_repulsive = -k_rep * (1 / min_obstacle_dist - 1 / d0) ** 2 if min_obstacle_dist!=0 else 0
            R_repulsive = np.clip(R_repulsive, -10.0, 0.0)

        # === Штраф за ложный путь ===
        R_fake_path = 0.0
        if robot.lidar_obstacle_detected and (potential_value - robot.prev_potential > 0.2 or potential_value == robot.prev_potential):
            R_fake_path = -5.0

        # === Суммарная награда ===
        total_reward = (
            2.0 * R_potential +
            1.0 * R_intermediate +
            # 1.0 * grad_reward +
            1.0 * R_repulsive +
            0.5 * R_fake_path
        )
        # if nearest_idx > robot.last_waypoint_idx:
        #     passed_points = nearest_idx - robot.last_waypoint_idx
        #     total_reward += 30 * passed_points
        #     robot.last_waypoint_idx = nearest_idx

        # total_reward = np.clip(total_reward, -50.0, 100.0)

        # === Логгирование ===
        logger.info(f"R_potential: {R_potential:.2f}, R_intermediate: {R_intermediate:.2f}, grad: {grad_reward:.2f}, rep: {R_repulsive:.2f}, fake: {R_fake_path:.2f}, total: {total_reward:.2f}")

        robot.prev_x, robot.prev_y = current_x, current_y

        return total_reward

    
    def compute_deviation_from_path(self, robot, current_pos):
        """
        Вычисляет минимальное расстояние от текущей позиции агента до оптимального пути.
        :param current_pos: (x, y) текущая позиция агента.
        :param optimal_path: список точек пути [(x1, y1), (x2, y2), ...]
        :return: минимальное расстояние до пути (скаляр).
        """
        if robot.optimal_path:
            path_points = np.array(robot.optimal_path)
            distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
            min_distance = float(distances.min())
            return min_distance
        return np.inf



    def get_deviation_penalty(self, current_pos, robot, max_penalty=10):
        """
        Рассчитывает штраф за отклонение от пути.
        :param current_pos: (x, y) текущая позиция агента.
        :param optimal_path: список точек пути [(x1, y1), (x2, y2), ...].
        :param max_penalty: максимальный штраф за сильное отклонение.
        :return: штраф (скаляр, отрицательный).
        """
        if current_pos.ndim == 2 and current_pos.shape[0] == 1:
            current_pos = current_pos[0]

        state = world_to_map(current_pos, resolution = 0.05, origin = (-7.76, -7.15),  map_offset = (0, 0), map_shape = self.grid_map.shape)
        
        deviation = self.compute_deviation_from_path(robot, state)
        
        # Можно сделать штраф линейным или экспоненциальным в зависимости от задачи
        penalty = -min(max_penalty, deviation)  # Чем дальше от пути, тем больше штраф
        return penalty
    
    def compute_collision_penalty(self, robot1, robot2):
        distance = np.sqrt((robot1.current_x - robot2.current_x)**2 + 
                        (robot1.current_y - robot2.current_y)**2)
        if distance < 0.5:  # Безопасная дистанция
            return -50 * (0.5 - distance)
        return 0
    
    def step(self, actions):
        EPSILON = 1e-5
        rewards = []
        dones = []

        robots_dist = math.sqrt((self.robots[1].current_x - self.robots[0].current_x)**2 + (self.robots[1].current_y - self.robots[0].current_y)**2)
        if (robots_dist <= 0.7 or self.step_ % 50 == 0) and self.step_ % 10 == 0:

            for robot in self.robots:
                # print('Replaning paths for agents')
                if np.linalg.norm(robot.velocity_vector) < EPSILON:
                    robot.velocity_vector = np.array([EPSILON, 0.0])
                other = self.robots[1] if robot.namespace == 'tb0' else  self.robots[0]
                robot.optimal_path = self.interpolate_path(self.path(robot, self.grid_map, other.occupation_map, self.penalty_map))
                robot.potential_field = generate_potential_field(self.grid_map, world_to_map(robot.goal, 0.05, (-7.76,-7.15), (0, 0), self.grid_map.shape), robot.optimal_path)

        self.step_ += 1

        for robot, action in zip(self.robots, actions):

            if robot.optimal_path != []:
                robot.world_path = [
                    grid_to_world(p[0], p[1], self.grid_map.shape)
                    for p in robot.optimal_path
                ]

            cmd_msg = Twist()
            linear = float(np.clip(action[0], 0.05, 0.26))
            angular = float(np.clip(action[1], -0.82, 0.82))


            cmd_msg.linear.x  = linear
            cmd_msg.angular.z = angular
            robot.cmd_vel_pub.publish(cmd_msg)
            rclpy.spin_once(self, timeout_sec=0.1)

            other = self.robots[1] if robot.namespace == 'tb0' else self.robots[0]

            prev_xw, prev_yw = robot.current_x, robot.current_y

            robot.steps += 1

            distance = math.sqrt((robot.target_x - robot.current_x) ** 2 + (robot.target_y - robot.current_y) ** 2)
            angle_to_wp = math.atan2(robot.nearest_wp[1] - robot.current_y, robot.nearest_wp[0] - robot.current_x)
            angle_to_goal = math.atan2(robot.target_y - robot.current_y, robot.target_x - robot.current_x)
            angle_diff = (angle_to_goal - robot.current_yaw + np.pi) % (2 * np.pi) - np.pi
            angle_to_wp_diff = (angle_to_wp - robot.current_yaw + np.pi) % (2 * np.pi) - np.pi

            min_obstacle_dist = min(robot.obstacles) if robot.obstacles else 3.5

            obstacle_detected = robot.lidar_obstacle_detected or robot.camera_obstacle_detected

            if robot.optimal_path != []:
                reward_potent_val = self.compute_potential_reward(robot.optimal_path, obstacle_detected, robot)
                reward_optimal_path = self.get_deviation_penalty(robot.state[:2], robot)
                
            dx = robot.current_x - prev_xw
            dy = robot.current_y - prev_yw
            disp = np.array([dx, dy])
            to_goal = np.array([robot.target_x - prev_xw, robot.target_y - prev_yw])
            norm = np.linalg.norm(to_goal)
            to_goal /= norm if norm>0 else 1.0
            dist_forward = np.dot(disp, to_goal)            
            forward_reward = 100.0 * dist_forward
            spin_penalty = - robot.beta_spin * abs(angular)
            # print(f"{robot.namespace} {robot.reward} потенциальная награда")

            robot.done = False

            if obstacle_detected:
                robot.obstacle_count += 1
                # print(f"{robot.namespace} reward до {robot.reward} обнаружения препятствия")
                robot.reward -= 10  
                # print(f"{robot.namespace} reward после {robot.reward} обнаружения препятствия")
                if robot.obstacle_count >= 300:
                    robot.reward -= 50
                    for robot in self.robots:
                        robot.done = True
                        robot.obstacle_count = 0

                    print("Episode terminated due to repeated obstacle detection")

            if min_obstacle_dist < 0.3:
                # print(f"{robot.namespace} reward до {robot.reward} обнаружения препятствия ОЧЕНЬ БЛИЗКО")

                robot.reward -= 3.0 * (0.3 - min_obstacle_dist)
                # print(f"{robot.namespace} reward после {robot.reward} обнаружения препятствия ОЧЕНЬ БЛИЗКО")

            if abs(robot.current_x - robot.prev_x) < 0.01 and distance > 0.35:
                # print(f"{robot.namespace} reward до {robot.reward} (остался в той же точке что и был)")
                robot.reward -= 20
                # print(f"{robot.namespace} reward после {robot.reward} (остался в той же точке что и был)")

            if robot.optimal_path == []:
                robot.done = True
                robot.reward -= 100 
            if distance < 0.3:
                print('GOAL REACHED!!!!')
                robot.reward += 500
                robot.done = True
            if distance > 10:
                robot.reward -= 100
                robot.done = True 
            if self.step_ >= self.max_steps:
            # if robot.steps >= 5:
                robot.reward -= 20
                for robot in self.robots:
                    robot.done = True
                    robot.steps = 0

            if robot.optimal_path != [] and robot.last_waypoint_idx < len(robot.optimal_path):
                # Берем точку с шагом WAYPOINT_STEP
                target_idx = min(robot.last_waypoint_idx + self.wp_step, 
                            len(robot.optimal_path)-1)
                wp = robot.optimal_path[target_idx]
                robot.nearest_wp = grid_to_world(wp[0],wp[1],self.grid_map.shape)
                self.visualize_nearest_wp(robot, grid_to_world(wp[0],wp[1],self.grid_map.shape, viz = True))
                # Проверяем достижение любой точки после текущего индекса
                nearest_reached_idx = np.argmin([
                    np.linalg.norm([
                        robot.current_x - wp[0], 
                        robot.current_y - wp[1]
                    ]) 
                    for wp in robot.world_path[robot.last_waypoint_idx:]
                ])
                if nearest_reached_idx >= robot.last_waypoint_idx:
                    print('Int point reached by', robot.namespace)
                    robot.last_waypoint_idx += nearest_reached_idx + self.wp_step
                    robot.reward += 30.0 * (nearest_reached_idx + 1)

            # print('before hausdorff_dist reward', robot.namespace, robot.reward)
            # robot.reward += -0.5 * self.hausdorff_dist(robot)
            # print('after hausdorff_dist reward', robot.namespace, robot.reward)

            if math.sqrt((other.current_x - robot.current_x)**2 + (other.current_y - robot.current_y)**2) < 0.4:
                # print(f"{robot.namespace} reward {robot.reward} before collision with other")
                robot.reward -= 20
                # print(f"{robot.namespace} reward {robot.reward} after collision with other")
            # print(f"{robot.namespace} reward {robot.reward} до награды за близость к цели")
            goal_vector = np.array([robot.target_x - robot.current_x, 
                                robot.target_y - robot.current_y])
            goal_dist = np.linalg.norm(goal_vector)

            # if goal_dist > 0.3:  # epsilon = 0.3
            #     if np.linalg.norm(robot.velocity_vector) > 0.01:
            #         direction_to_goal = goal_vector / goal_dist
            #         velocity_dir = robot.velocity_vector / np.linalg.norm(robot.velocity_vector)
            #         alignment = np.dot(velocity_dir, direction_to_goal)
            #         R_goal = alignment * np.linalg.norm(robot.velocity_vector)
            #     else:
            #         R_goal = -0.5  
            # else:
            #     R_goal = 15.0  
            #     robot.done = True
            
            R_goal = ( 1 / (distance + 1e-5) ) * 20 if distance <= 2. else -(1 / (distance + 1e-5) ) * 10
            
            # print(f"{robot.namespace} reward {robot.reward} после награды за близость к цели")
            robot.occupation_map = np.maximum(robot.occupation_map - 0.0015, 0)
            angle_bonus = 1 * (1 - abs(angle_diff)/math.pi) + 1 * (1 - abs(angle_to_wp_diff)/math.pi)
            robot.reward += angle_bonus
            # print('before potential reward', robot.namespace, robot.reward)

            # robot.reward = self.reward_filter(robot.reward)
            if robot.optimal_path != []:
                robot.reward += reward_potent_val + reward_optimal_path + forward_reward + spin_penalty - 0.01  + R_goal
            else:
                robot.reward += forward_reward + spin_penalty - 0.01  + R_goal
            # print(robot.namespace, robot.reward)
            robot.reward *= 0.001

            rewards.append(robot.reward)
            dones.append(robot.done)
        return self.get_observations(dones), rewards, dones, {}
    
    def hausdorff_dist(self, robot):
        if not robot.optimal_path or not robot.world_path:
            return 0.0
        
        # Расстояния от точек пути до робота
        dist_path_to_robot = [
            np.linalg.norm([wp[0] - robot.current_x, wp[1] - robot.current_y])
            for wp in robot.world_path
        ]
        
        # Расстояния от робота до точек пути
        dist_robot_to_path = [
            np.linalg.norm([robot.current_x - wp[0], robot.current_y - wp[1]])
            for wp in robot.world_path
        ]
        
        d1 = np.max(np.min(dist_path_to_robot))
        d2 = np.max(np.min(dist_robot_to_path))
        return 0.5*(d1 + d2)
    
    def reset(self):
        states = []
        spawn_points = [[2.0, -5.0], [2.0, -2.0]]
        self.step_ = 0

        for i, robot in enumerate(self.robots):
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0  
            cmd_msg.angular.z = 0.0
            robot.cmd_vel_pub.publish(cmd_msg)  
            rclpy.spin_once(self, timeout_sec=0.1) 
        client = self.create_client(Empty, '/reset_world')

        if client.wait_for_service(timeout_sec=1.0):
            try:
                req = self.reset_world.call_async(Empty.Request())
                while not req.done():
                    rclpy.spin_once(self, timeout_sec=0.1)
            except:
                import traceback
                traceback.print_exc()
        else:
            self.get_logger().info('reset : service not available, waiting again...')

        print('env reseted')

        for i, robot in enumerate(self.robots):
            set_robot_state = EntityState()
            set_robot_state.name = f"tb{i}"
            set_robot_state.pose.position.x = spawn_points[i][0]
            set_robot_state.pose.position.y = spawn_points[i][1]
            set_robot_state.pose.position.z = 0.1

            robot_state = SetEntityState.Request()
            robot_state._state = set_robot_state
            set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
            if set_state.wait_for_service(timeout_sec=1.0):
                try:
                    req = set_state.call_async(robot_state)
                    while not req.done():
                        rclpy.spin_once(self, timeout_sec=0.1)
                except rclpy.ServiceException as e:
                    print("/gazebo/reset_simulation service call failed")
            else:
                self.get_logger().info('reset : service not available, waiting again...')
            timeout_counter = 0
            while (abs(robot.current_x) < 0.01 and abs(robot.current_y) < 0.01 and 
                abs(robot.current_yaw) < 1e-3 and timeout_counter < 50):
                rclpy.spin_once(self, timeout_sec=0.1)
                timeout_counter += 1

        for i, robot in enumerate(self.robots):
            robot.steps = 0
            robot.obstacle_count = 0
            robot.current_x = spawn_points[i][0]
            robot.current_y = spawn_points[i][1]
            robot.prev_x = None
            robot.prev_y = None
            robot.current_yaw = 0.0
            robot.done = False
            robot.obstacles = []
            robot.camera_obstacle_detected = False
            robot.lidar_obstacle_detected = False
            robot.reward = 0
            robot.last_waypoint_idx = 5
            robot.nearest_wp = None
            robot.occupation_map = np.zeros_like(self.grid_map, dtype=np.float32)
            other = self.robots[1] if robot.namespace == 'tb0' else self.robots[0]
            robot.optimal_path = self.interpolate_path(
                self.path(robot, self.grid_map, other.occupation_map, self.penalty_map)
            )
            # robot.potential_field = generate_potential_field(self.grid_map, world_to_map(robot.goal, 0.05, (-7.76,-7.15), (0, 0), self.grid_map.shape), robot.optimal_path)

            state = self._get_initial_state(robot)
            states.append(state)
        self.step_ = 0

        return np.array(states)

    def _get_initial_state(self, robot):
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        x, y = world_to_map(
            (robot.current_x, robot.current_y),
            resolution=0.05,
            origin=(-7.76, -7.15),
            map_offset=(0, 0),
            map_shape=self.grid_map.shape
        )
        other = self.robots[1] if robot.namespace == 'tb0' else self.robots[0]
        dynamic_cost = other.occupation_map[y][x]
        penalty = self.penalty_map[y][x]
        min_obstacle_dist = 3.5 if not robot.obstacles else min(robot.obstacles)
        angle_to_goal = math.atan2(robot.target_y - robot.current_y, robot.target_x - robot.current_x)
        angle_diff = (angle_to_goal - robot.current_yaw + np.pi) % (2 * np.pi) - np.pi
        distance = math.sqrt((robot.target_x - robot.current_x) ** 2 + (robot.target_y - robot.current_y) ** 2)
        robots_dist = math.sqrt((other.current_x - robot.current_x)**2 + (other.current_y - robot.current_y)**2)
        if robot.optimal_path:
            target_idx = min(robot.last_waypoint_idx + self.wp_step, len(robot.optimal_path)-1)
            wp = robot.optimal_path[target_idx]
            robot.nearest_wp = grid_to_world(wp[0], wp[1], self.grid_map.shape, viz=True)
        else:
            robot.nearest_wp = (robot.target_x, robot.target_y)
        
        angle_to_wp_diff = math.atan2(robot.nearest_wp[1] - robot.current_y, robot.nearest_wp[1] - robot.current_x)
        wp_distance = math.sqrt((robot.nearest_wp[0] - robot.current_x) ** 2 + (robot.nearest_wp[1] - robot.current_y) ** 2)
        dx_goal = (robot.target_x - robot.current_x)
        dy_goal = (robot.target_y - robot.current_y)
        
        if robot.nearest_wp:
            dx_wp = (robot.nearest_wp[0] - robot.current_x)
            dy_wp = (robot.nearest_wp[1] - robot.current_y)
        else:
            dx_wp, dy_wp = 0.0, 0.0
        robot.state  = np.array([robot.current_x, robot.current_y, angle_diff, min_obstacle_dist, 2*(dynamic_cost + penalty), dx_goal, dy_goal, dx_wp, dy_wp])

        return robot.state
    
    def visualize_nearest_wp(self, robot, wp):
        wp_marker = Marker()
        wp_marker.header.frame_id = "map"
        wp_marker.header.stamp = self.get_clock().now().to_msg()
        wp_marker.ns = f"{robot.namespace}_wp"
        wp_marker.id = 0
        wp_marker.type = Marker.SPHERE
        wp_marker.action = Marker.ADD
        wp_marker.pose.position.x = wp[0]
        wp_marker.pose.position.y = wp[1]
        wp_marker.pose.position.z = 0.0
        wp_marker.scale.x = 0.1
        wp_marker.scale.y = 0.1
        wp_marker.scale.z = 0.1
        wp_marker.color.a = 1.0
        wp_marker.color.r = 1.0  
        wp_marker.color.g = 0.0
        wp_marker.color.b = 0.0
        wp_marker.color.a = 1.0
        if robot.namespace == 'tb1':
            wp_marker.color.g = 1.0
        else:
            wp_marker.color.g = 0.0
        if robot.namespace == 'tb1':
            wp_marker.color.r = 0.0
        else:
            wp_marker.color.r = 1.0
        wp_marker.color.b = 0.0
        robot.wp_marker_pub.publish(wp_marker)

    def visualize_path(self, robot, path):
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = f"{robot.namespace}_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05 
        path_marker.color.a = 1.0
        if robot.namespace == 'tb0':
            path_marker.color.g = 1.0
        else:
            path_marker.color.g = 0.0
        if robot.namespace == 'tb0':
            path_marker.color.r = 0.0
        else:
            path_marker.color.r = 1.0
        path_marker.color.b = 0.0
        for (x, y) in path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            path_marker.points.append(p)
        robot.path_marker_pub.publish(path_marker)

    def publish_costmaps(self):
        pass


    def publish_grid_map(self):
        if self.grid_map is None:
            return
        
        map_a_msg_pose = Pose()
        map_a_msg_pose.position.x = -7.76
        map_a_msg_pose.position.y = -7.15
        map_a_msg_pose.position.z = 0.0

        msg = OccupancyGrid(
            header = Header(stamp = self.get_clock().now().to_msg(), frame_id="map"), 
            info = MapMetaData(width=self.grid_map.shape[1], height=self.grid_map.shape[0], resolution=0.05, map_load_time= self.get_clock().now().to_msg(), origin=map_a_msg_pose)
        )        
        for i in range(0,self.grid_map.shape[0]):
            for j in range(0,self.grid_map.shape[1]):
                msg.data.append(int(self.grid_map[i][j] * 100))

        self.gridmap_pub.publish(msg)

    def publish_occupation_map(self, robot):
        map_a_msg_pose = Pose()
        map_a_msg_pose.position.x = -7.76
        map_a_msg_pose.position.y = -7.15
        map_a_msg_pose.position.z = 0.0
        msg = OccupancyGrid()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        msg.info = MapMetaData(width=self.grid_map.shape[1], height=self.grid_map.shape[0], resolution=0.05, map_load_time= self.get_clock().now().to_msg(), origin=map_a_msg_pose)

        
        occupation_data = np.clip(robot.occupation_map * 10, 0, 100).astype(np.int8)
        for i in range(0,self.grid_map.shape[0]):
            for j in range(0,self.grid_map.shape[1]):
                msg.data.append(occupation_data[i][j])
        robot.occupation_pub.publish(msg)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)