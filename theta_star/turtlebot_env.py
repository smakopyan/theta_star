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
import collections
from theta_star import ThetaStar
import matplotlib.pyplot as plt
from collections import deque
import logging
from multi_robot_navigator_example import euler_from_quaternion
import os
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def slam_to_grid_map(slam_map, threshold=200):
    
    grid_map = np.where(slam_map < threshold, 1, 0)  
    num_obstacles = np.count_nonzero(grid_map == 1)
    return grid_map
    
def grid_to_world(x_grid, y_grid, map_shape, map_resolution = 0.05, map_origin = (-7.76,-7.15)):
    
    x_world = x_grid * map_resolution + map_origin[0]
    y_world = y_grid * map_resolution + map_origin[1]
    return (x_world, y_world)

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

def generate_potential_field(grid_map, goal, path_points, k_att=1.0, k_rep=20.0, d0=3.0, scale=0.05, k_str=8.0, p_s=8.0):
    """
    Улучшенная генерация потенциального поля:
    - Притягивающий потенциал (quadratic)
    - Отталкивающий потенциал (logarithmic attenuation)
    - Промежуточные точки маршрута усиливают притягивающий потенциал
    """
    height, width = grid_map.shape
    y_coords, x_coords = np.indices(grid_map.shape)
    obstacles = np.argwhere(grid_map == 1)

    dx = x_coords - goal[0]
    dy = y_coords - goal[1]
    dist_to_goal = np.sqrt(dx**2 + dy**2)  
    
    visibility_mask = np.ones_like(grid_map, dtype=np.float32)
    for (y, x) in obstacles:
        visibility_mask[y, x] = 0  
        
    att_field = -0.5 * k_att * np.exp(-scale * dist_to_goal) * (0.5 + 0.5 * visibility_mask)

    mask_blackhole = (dist_to_goal <= p_s) 
    str_field = np.zeros_like(grid_map, dtype=np.float32)
    str_field[mask_blackhole] = -0.5 * k_str * (p_s - dist_to_goal[mask_blackhole])**2

    str_points = np.zeros_like(grid_map, dtype=np.float32)
    for pt in path_points:
        dx_pt = x_coords - pt[0]
        dy_pt = y_coords - pt[1]
        dist_to_pt = np.sqrt(dx_pt**2 + dy_pt**2)
        mask = (dist_to_pt <= p_s)
        str_points[mask] += -0.5 * k_str * (p_s - dist_to_pt[mask])**2

    rep_field = np.zeros_like(grid_map, dtype=np.float64)
    for (y, x) in obstacles:
        dist_map = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        mask = (dist_map < d0) & (dist_map > 0)
        rep_field[mask] += 0.5 * k_rep / (dist_map[mask]**2 - 1/d0)**2

    field = att_field + str_field + str_points + rep_field
    return field

class Robot():
    def __init__(self, namespace, state_pos, goal):
        self.namespace = namespace
        self.occupations = None
        self.penalties = None
        self.cmd_vel_pub = None
        self.obstacle_detected = False
        self.bridge = CvBridge()
        self.camera_obstacle_detected = False
        self.lidar_obstacle_detected = False
        self.path_marker_pub = None
        
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
        self.max_steps = 150
        self.steps = 0 
        self.recent_obstacles = []
        self.state = np.array([])
        self.reward = 0.0
        self.done = False
        self.min_obstacle_dist = 0
        self.penalty = 0.0
        self.last_waypoint_idx = 0

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
        spawn_points = [[-0.7, 0.05], [-2.5, 0.05]]
        # goals = [[0.483287, -2.73528],[0.583933, -3.66662] ]
        goals = [[-4.0, 0.05], [1.0, 0.05]]

        self.robots = [Robot(f"tb{i}", spawn_points[i], goals[i]) for i in range(self.num_robots)]

        slam_map = cv2.imread(os.path.join(get_package_share_directory('theta_star'),
                                           'maps','map3.pgm'), cv2.IMREAD_GRAYSCALE)
        
  
        self.grid_map = slam_to_grid_map(slam_map)

        self.map_initialized = False
        self.occupation_map = np.zeros_like(self.grid_map, dtype=np.float32)
        self.penalty_map = np.ones_like(self.grid_map, dtype=np.float32)
        self.max_steps = 150

        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.gridmap_pub = self.create_publisher(OccupancyGrid, '/grid_costmap', qos_profile)
        self.occupation_pub = self.create_publisher(OccupancyGrid, '/occupation_costmap', 10)
        self.penalty_pub = self.create_publisher(OccupancyGrid, '/penalty_costmap', 10)
        
        for robot in self.robots:
            robot.cmd_vel_pub = self.create_publisher(Twist, f'/{robot.namespace}/cmd_vel', 10)
            
            robot.path_marker_pub = self.create_publisher(
                Marker, 
                f'/{robot.namespace}/path_marker', 
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
            robot.optimal_path_ = self.path(robot, self.grid_map, self.occupation_map, self.penalty_map)
            
            robot.optimal_path = self.interpolate_path(robot.optimal_path_)
            # print(robot.optimal_path)
            robot.potential_field = generate_potential_field(self.grid_map, world_to_map(robot.goal, 0.05, (-7.76,-7.15), (45, 15), self.grid_map.shape), robot.optimal_path)
            self.show_potential_field(robot) 
                    
        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.action_space = spaces.Discrete(3)  

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]),  
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(6,),  
            dtype=np.float32
        )                       

        self.timer = self.create_timer(0.05, self._timer_callback)
        self.costmap_timer = self.create_timer(1.0, self.publish_costmaps)

        self.state_filter = Zfilter(prev_filter=None, shape=self.observation_space.shape[0], clip=1.0)
        self.reward_filter = Zfilter(prev_filter=None, shape=(), center = False, \
                                            clip=5.0)
    
    def path(self, robot, grid_map, occupation_map, penalty_map,  map_resolution = 0.05, map_origin = (-7.76,-7.15)):
        spawn_points = [[-0.7, 0.05], [-2.5, 0.05]]
        if robot.namespace == 'tb0':
            state_world = world_to_map(spawn_points[0], 0.05, (-7.76,-7.15), (0,0), grid_map.shape)
        else:
            state_world = world_to_map(spawn_points[1], 0.05, (-7.76,-7.15), (0,0), grid_map.shape)
        goal_world = robot.goal  # Цель в мировых координатах
        map_offset = (0, 0)  # Смещение координат

        map_shape = grid_map.shape  # (высота, ширина) карты

        goal_pixel = world_to_map(goal_world, map_resolution, map_origin, map_offset, map_shape)
        theta_star = ThetaStar()
        optimal_path = theta_star.plan((state_world[1], state_world[0]), 
                                (goal_pixel[1], goal_pixel[0]),
                                grid_map, occupation_map, penalty_map)
        if optimal_path is None:
            print("Путь не найден")
            return []
        else:
            print("Найденный путь:")
            print(optimal_path)
        return optimal_path


    def _timer_callback(self):
        pass
        # for robot in self.robots:
        #     robot.optimal_path = self.path(robot, self.grid_map, self.occupation_map, self.penalty_map)
        #     path_= self.get_path(robot.optimal_path)
        #     self.visualize_path(robot, path_)

    def get_path(self, opt_path):
        path = [grid_to_world(i[0], i[1], map_shape = self.grid_map.shape) for i in opt_path]
        return path

    def update_dynamic_maps(self):
        for robot in self.robots:
            x, y = world_to_map(
                (robot.current_x, robot.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )
            self.occupation_map[y][x] += 0.3
            self.penalty_map[y][x] = self.calculate_dynamic_penalty(x, y)

    def get_observations(self):
        observations = []
        for robot in self.robots:
            x, y = world_to_map(
                (robot.current_x, robot.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )
            
            dynamic_cost = self.occupation_map[y][x]
            penalty = self.penalty_map[y][x]
            distance = math.sqrt((robot.target_x - robot.current_x) ** 2 + (robot.target_y - robot.current_y) ** 2)
            
            obs = np.concatenate([
                robot.state,
                [2*(dynamic_cost + penalty), distance],
            ])
            # print("obs before:", obs)
            obs = self.state_filter(obs)
            # print("obs after: ", obs)
            observations.append(obs)
            # observations.append(self.normalize_state(obs))
        return observations
    
    def interpolate_path(self, raw_path, step=1):
        if not raw_path or len(raw_path) < 2:
            return []
            
        interpolated = []
        for i in range(len(raw_path)-1):
            start = np.array(raw_path[i])
            end = np.array(raw_path[i+1])
            distance = np.linalg.norm(end - start)
            num_points = int(distance / step)
            
            for t in np.linspace(0, 1, num_points):
                point = start * (1 - t) + end * t
                interpolated.append([point[1], point[0]])
        interpolated = [(int(round(x)), int(round(y))) for (x, y) in interpolated]
        return interpolated

    def calculate_dynamic_penalty(self, x, y):
        penalty = 0.0
        for other in self.robots:
            ox, oy = world_to_map(
                (other.current_x, other.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )
            distance = np.hypot(x - ox, y - oy)
            penalty += 1.0 / (distance + 1e-5)
        return min(penalty, 5.0)
    
    def create_odom_callback(self, robot):
        def callback(msg):
            robot.current_x = msg.pose.pose.position.x
            robot.current_y = msg.pose.pose.position.y
            orientation_q = msg.pose.pose.orientation
            robot.current_yaw = euler_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        return callback

    # def scan_callback(self, msg):
        # self.obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max else msg.range_max for r in msg.ranges]
    
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
    
    # def get_next_state(self, state, action, angle):
    #     """
    #     Предсказывает следующее состояние на основе текущего состояния, действия и переданного угла.

    #     :param state: текущее состояние [x, y, min_obstacle_dist] (без угла)
    #     :param action: выбранное действие (0 - поворот вправо, 1 - движение вперёд, 2 - поворот влево)
    #     :param angle: текущий угол робота (передаётся отдельно)
    #     :return: следующее состояние [next_x, next_y, next_min_obstacle_dist], next_angle
    #     """
    #     current_x, current_y, _, min_obstacle_dist = state.squeeze()  # Распаковываем текущее состояние
    #     next_x, next_y = current_x, current_y  # По умолчанию остаются неизменными
    #     next_angle = angle # Начинаем с текущего угла
    #     logger.info(f'Next_angel in get_next_state (current_yaw): {next_angle}') 
    #     # print(next_angle)
    #     # Определяем действие
    #     if action == 0:  # Поворот вправо
    #         next_angle = angle - 0.5  
    #     elif action == 1:  # Движение вперёд
    #         if not (self.x_range[0] <= next_x <= self.x_range[1] and self.y_range[0] <= next_y <= self.y_range[1]):
    #             next_x, next_y = current_x, current_y  # Оставляем на местея
    #         else:
    #             next_x = current_x + np.cos(angle) * 0.2  # Двигаемся в направлении угла
    #             next_y = current_y + np.sin(angle) * 0.2
    #     elif action == 2:  
    #         next_angle = angle + 0.5  # Увеличиваем угол

    #     # Ограничиваем угол в диапазоне [-π, π]
    #     next_angle = (next_angle + np.pi) % (2 * np.pi) - np.pi

    #     # min_obstacle_dist остаётся прежним (или можно пересчитывать)
    #     return np.array([next_x, next_y, next_angle, min_obstacle_dist], dtype=np.float32)
    def get_next_state(self, state, action, angle, robot):
        """
        Предсказывает следующее состояние на основе текущего состояния, действия и угла.
        :param state: текущее состояние [x, y, angle_diff, min_obstacle_dist, dynamic_cost, penalty]
        :param action: выбранное действие (0-2)
        :param angle: текущий угол робота (yaw)
        :return: следующее состояние и угол
        """
        # Распаковываем только необходимые компоненты состояния
        current_x = state[0]
        current_y = state[1]
        angle_diff = state[2]
        min_obstacle_dist = state[3]
        
        next_x, next_y = current_x, current_y
        next_angle = angle
        
        # Остальная логика обработки действия...
        if action == 0:  # Поворот вправо
            next_angle = angle - 0.5
        elif action == 1:  # Вперёд
            next_x = current_x + np.cos(angle) * 0.2
            next_y = current_y + np.sin(angle) * 0.2
        elif action == 2:  # Поворот влево
            next_angle = angle + 0.5

        next_angle = (next_angle + np.pi) % (2 * np.pi) - np.pi
        
        next_x_map, next_y_map = world_to_map(
            (current_x, current_y),
            resolution=0.05,
            origin=(-7.76, -7.15),
            map_offset=(0, 0),
            map_shape=self.grid_map.shape
        )
        dynamic_cost = self.occupation_map[next_y_map][next_x_map] 
        penalty = self.penalty_map[next_y_map][next_x_map]

        # print(2*(dynamic_cost + penalty))
        distance = math.sqrt((robot.target_x - next_x)**2 + (robot.target_y - next_y)**2)
        # Сохраняем остальные компоненты состояния без изменений
        return np.array([
            next_x,
            next_y,
            angle_diff,  # Обновить при необходимости
            min_obstacle_dist,
            2*(dynamic_cost + penalty) if (dynamic_cost and penalty) else 1.0, 
            distance,
        ], dtype=np.float32)
    
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
            R_repulsive = -k_rep * (1 / min_obstacle_dist - 1 / d0) ** 2
            R_repulsive = np.clip(R_repulsive, -10.0, 0.0)

        # === Штраф за ложный путь ===
        R_fake_path = 0.0
        if robot.lidar_obstacle_detected and (potential_value - robot.prev_potential > 0.2 or potential_value == robot.prev_potential):
            R_fake_path = -5.0

        # === Суммарная награда ===
        total_reward = (
            1.0 * R_potential +
            1.0 * R_intermediate +
            1.0 * grad_reward +
            1.0 * R_repulsive +
            1.0 * R_fake_path
        )
        if nearest_idx > robot.last_waypoint_idx:
            passed_points = nearest_idx - robot.last_waypoint_idx
            total_reward += 10 * passed_points
            robot.last_waypoint_idx = nearest_idx

        total_reward = np.clip(total_reward, -50.0, 100.0)

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
            min_distance = np.min(distances)
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
        penalty = -min(max_penalty, deviation*1.5)  # Чем дальше от пути, тем больше штраф
        return penalty
    
    def compute_collision_penalty(self, robot1, robot2):
        distance = np.sqrt((robot1.current_x - robot2.current_x)**2 + 
                        (robot1.current_y - robot2.current_y)**2)
        if distance < 0.5:  # Безопасная дистанция
            return -50 * (0.5 - distance)
        return 0
    
    # def normalize_state(self, state):
    #     return np.array([
    #         state[0]/10.0,           
    #         state[1]/10.0,           
    #         state[2]/np.pi,          
    #         state[3]/12.0,           
    #         state[4]/100.0, 
    #         state[5]/12,          
    #     ], dtype=np.float32)
    

    
    def step(self, actions):
        states, rewards, dones = [], [], []
        # self.occupation_map = np.zeros_like(self.grid_map, dtype=np.float32)
        # self.penalty_map = np.zeros_like(self.grid_map, dtype=np.float32)
        self.update_dynamic_maps()

        for robot, action in zip(self.robots, actions):
            cmd_msg = Twist()
            if action == 0:
                cmd_msg.angular.z = 0.5
            elif action == 1:
                cmd_msg.linear.x = 0.2
            elif action == 2:
                cmd_msg.angular.z = -0.5

            robot.cmd_vel_pub.publish(cmd_msg)
        
            rclpy.spin_once(self, timeout_sec=0.1) 
        
            robot.steps += 1
            
            # print(self.obstacles)

            distance = math.sqrt((robot.target_x - robot.current_x) ** 2 + (robot.target_y - robot.current_y) ** 2)
            angle_to_goal = math.atan2(robot.target_y - robot.current_y, robot.target_x - robot.current_x)
            angle_diff = (angle_to_goal - robot.current_yaw + np.pi) % (2 * np.pi) - np.pi

            min_obstacle_dist = min(robot.obstacles) if robot.obstacles else 3.5

            obstacle_detected = robot.lidar_obstacle_detected or robot.camera_obstacle_detected
            robot.state = np.array([
                robot.current_x,
                robot.current_y,
                angle_diff,
                min_obstacle_dist])


            states.append(robot.state)

            # distance_rate = (self.past_distance - distance)
            # print(min_obstacle_dist)

            reward_potent_val = self.compute_potential_reward(robot.optimal_path, obstacle_detected, robot)
            reward_optimal_path = self.get_deviation_penalty(robot.state[:2], robot)

            robot.reward = reward_potent_val + reward_optimal_path
            # reward += 50.0 * distance_rate
            # self.past_distance = distance
            # print(obstacle_detected)
            robot.done = False

            # Обнаружено препятствие
            if obstacle_detected:
                robot.obstacle_count += 1
                robot.reward -= 50  # менее агрессивно
                if robot.obstacle_count >= 100:
                    robot.done = True
                    for robot in self.robots:
                        robot.obstacle_count = 0

                    print("Episode terminated due to repeated obstacle detection")

            # Очень близко к препятствию
            if min_obstacle_dist < 0.4:
                robot.reward -= 20 * (0.5 - min_obstacle_dist)

            if abs(robot.current_x - robot.prev_x) < 0.05 and distance > 0.35:
                robot.reward -= 700
            # Достигли цели
            other = self.robots[1] if robot.namespace == 'tb0' else self.robots[0]
            if distance < 0.3:
                print('GOAL REACHED!!!!')
                robot.reward += 1000
                other.reward += 700
                robot.done = True

            prev_distance = math.sqrt((robot.target_x - robot.prev_x)**2 + (robot.target_y - robot.prev_y)**2)
            current_distance = math.sqrt((robot.target_x - robot.current_x)**2 + (robot.target_y - robot.current_y)**2)
            distance_reward = (prev_distance - current_distance) * 0.3  # Масштабирующий коэффициент
            goal_distance_reward = (distance - current_distance) * 0.5
            
            robot.reward += (distance_reward + goal_distance_reward)

            if distance > 10:
                robot.reward -= 200
                robot.done = True 
            # Превышен лимит шагов
            if robot.steps >= robot.max_steps:
                robot.reward -= 100
                robot.done = True

            collision_penalty = self.compute_collision_penalty(self.robots[0], self.robots[1])
            rewards = [r + collision_penalty for r in rewards]
            
            # Условие завершения при столкновении
            if collision_penalty < -10:
                robot.reward -= 100
                other.reward -= 100
                for robot in self.robots:
                    robot.done==True 
            
            # print(robot.reward)
            robot.reward = self.reward_filter(robot.reward)
            # print(robot.reward)

            
            rewards.append(robot.reward)


            dones = [robot.done for robot in self.robots]

        return self.get_observations(), rewards, any(dones), {}

    def reset_state(self, robot, cur_pos):
        cur_x, cur_y = cur_pos
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0  
        cmd_msg.angular.z = 0.0
        robot.cmd_vel_pub.publish(cmd_msg)  
        rclpy.spin_once(self, timeout_sec=0.1) 
    
        client = self.create_client(Empty, '/reset_simulation')
        request = Empty.Request()
        if client.wait_for_service(timeout_sec=1.0):
            client.call_async(request)
        else:
            self.get_logger().warn('Gazebo reset service not available!')

        robot.current_x = cur_x
        robot.current_y = cur_y
        robot.current_yaw = 0.0
        robot.steps = 0
        robot.prev_distance = None
        robot.obstacles = []
        robot.camera_obstacle_detected = False
        robot.done = False
        
        return np.array([robot.current_x, robot.current_y, 0.0, 0.0])
    
    def reset(self):
        states = []
        spawn_points = [[-0.7, 0.05], [-2.5, 0.05]]

        for i, robot in enumerate(self.robots):
            robot.state = self.reset_state(robot, spawn_points[i])
        for robot in self.robots:
            x, y = world_to_map(
                (robot.current_x, robot.current_y),
                resolution=0.05,
                origin=(-7.76, -7.15),
                map_offset=(0, 0),
                map_shape=self.grid_map.shape
            )
                
            dynamic_cost = self.occupation_map[y][x]
            penalty = self.penalty_map[y][x]
            distance = math.sqrt((robot.target_x - robot.current_x) ** 2 + (robot.target_y - robot.current_y) ** 2)
                
            robot.state = np.concatenate([
                robot.state,
                [2*(dynamic_cost + penalty), distance]  
                ])
            # print("state before:",robot.state)

            robot.state = self.state_filter(robot.state)
            # print("state after:",robot.state)
            
            states.append(robot.state)
            # states = [self.normalize_state(state) for state in states]
        return states
    
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
        # self.publish_grid_map()
        # self.publish_occupation_map()
        # self.publish_penalty_map()

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
        # msg.data = [int(val) for val in self.grid_map.flatten()]

        self.gridmap_pub.publish(msg)

    def publish_occupation_map(self):
        msg = OccupancyGrid()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        
        occupation_data = np.clip(self.occupation_map * 10, 0, 100).astype(np.int8)
        
        msg.info = self.gridmap_pub.info
        msg.data = occupation_data.flatten().tolist()
        self.occupation_pub.publish(msg)

    def publish_penalty_map(self):
        msg = OccupancyGrid()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        
        penalty_data = np.clip(self.penalty_map * 20, 0, 100).astype(np.int8)
        
        msg.info = self.gridmap_pub.info
        msg.data = penalty_data.flatten().tolist()
        self.penalty_pub.publish(msg)


    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)