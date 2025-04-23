import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
import math
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import cv2
import collections
from rrt_star import RRTStar
from theta_star import ThetaStar
import matplotlib.pyplot as plt
from collections import deque
import logging

logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def slam_to_grid_map(slam_map, threshold=128):

    grid_map = np.where(slam_map < threshold, 1, 0)  
    num_obstacles = np.count_nonzero(grid_map == 1)

    # print(num_obstacles)
    
    # Визуализация grid_map
    # plt.figure(figsize=(8, 8))
    # plt.imshow(grid_map, cmap='gray')
    # plt.title(f'Grid Map с порогом {threshold}')
    # plt.axis('off')
    # plt.show()
    
    return grid_map


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
        x_map = ((x_world - origin[0]) / resolution).astype(int) + map_offset[0]
        y_map = ((y_world - origin[1]) / resolution).astype(int) + map_offset[1]
    else:
        x_map = int((x_world - origin[0]) / resolution) + map_offset[0]
        y_map = int((y_world - origin[1]) / resolution) + map_offset[1]

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


def path(state, goal, grid_map, map_resolution = 0.05, map_origin = (-4.86, -7.36)):
    
    state_world = state # Текущая позиция в мировых координатах
    goal_world = goal  # Цель в мировых координатах

    map_offset = (45, 15)  # Смещение координат
    map_shape = grid_map.shape  # (высота, ширина) карты

    state_pixel = world_to_map(state_world, map_resolution, map_origin, map_offset, map_shape)
    goal_pixel = world_to_map(goal_world, map_resolution, map_origin, map_offset, map_shape)
    # print(goal_pixel)

    # print(state_pixel)
    # print(goal_pixel)
    occupations = np.zeros(grid_map.shape)
    penalties = np.zeros(grid_map.shape)


    rrt_star = RRTStar(state_pixel, goal_pixel, grid_map)
    optimal_path = rrt_star.plan()
    
    # print(optimal_path)

    # optimal_path = [map_to_world(p, map_resolution, map_origin) for p in optimal_path]

    if optimal_path is None:
        print("Путь не найден")
        return []
    else:
        print("Найденный путь:")
        print(optimal_path)
        
        # Визуализация результатов:
        # plt.figure(figsize=(8, 8))
        # plt.imshow(grid_map, cmap='gray')
        
        # # Отрисовываем все узлы дерева
        # for node in rrt_star.node_list:
        #     if node.parent is not None:
        #         p1 = node.point
        #         p2 = node.parent.point
        #         # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-g")
                
        # # Отрисовываем найденный путь
        # path_x = [p[0] for p in optimal_path]
        # path_y = [p[1] for p in optimal_path]
        # # plt.plot(path_x, path_y, "-r", linewidth=2)
        
        # plt.scatter(state_pixel[0], state_pixel[1], color="blue", s=100, label="Старт")
        # plt.scatter(goal_pixel[0], goal_pixel[1], color="magenta", s=100, label="Цель")
        # plt.legend()
        # plt.title("RRT*")
        # plt.show()
    return optimal_path

# -------------------------------
# Функция для расчёта отклонения от оптимального пути
# -------------------------------
def compute_deviation_from_path(current_pos, optimal_path):
    path_points = np.array(optimal_path)
    distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
    min_distance = np.min(distances)
    return min_distance

def generate_potential_field(grid_map, goal, path_points, k_att=5.0, k_rep=50.0, d0=5.0, scale = 0.07):
    """
    Генерация потенциального поля:
    - Квадратичное притяжение к цели и промежуточным точкам.
    - Квадратичное отталкивание от препятствий в радиусе d0.
    """
    height, width = grid_map.shape
    y_coords, x_coords = np.indices(grid_map.shape)
    obstacles = np.argwhere(grid_map == 1)

    # Притягивающее поле к цели
    dx_goal = x_coords - goal[0]
    dy_goal = y_coords - goal[1]
    distance_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)
    att_field = -0.5 * k_att * np.exp(-scale * distance_to_goal)

    # Притяжение к промежуточным точкам пути
    att_points = np.zeros_like(grid_map, dtype=np.float32)
    for pt in path_points:
        dx_pt = x_coords - pt[0]
        dy_pt = y_coords - pt[1]
        distance_to_pt = np.sqrt(dx_pt**2 + dy_pt**2)
        att_points += -0.5 * k_att * np.exp(-scale * distance_to_pt)

    # Отталкивающее поле от препятствий
    rep_field = np.zeros_like(grid_map, dtype=np.float32)
    for (y, x) in obstacles:
        dist_map = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        mask = (dist_map <= d0) & (dist_map > 0)
        inv_dist = 1 / (dist_map[mask] + 1e-5)
        rep_field[mask] += 0.5 * k_rep * (inv_dist - 1/d0)**2

    # Итоговое поле (притяжение - отталкивание)
    field = att_field + att_points + rep_field
    return field

class TurtleBotEnv(Node, gym.Env):
    def __init__(self):
        super().__init__('turtlebot_env')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_laser = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.subscription_camera = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        
        self.bridge = CvBridge()
        self.camera_obstacle_detected = False
        self.lidar_obstacle_detected = False
        
        self.target_x = -2.0
        self.target_y = -6.0
        self.goal = [self.target_x, self.target_y]
        
        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.state_pose = [-2.0, -0.5]

        slam_map = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
        self.grid_map = slam_to_grid_map(slam_map)
        self.camera_history = deque(maxlen=20)  

        self.optimal_path = path(self.state_pose, self.goal, self.grid_map)
        self.potential_field = generate_potential_field(self.grid_map, world_to_map(self.goal, 0.05, (-4.86, -7.36), (45, 15), self.grid_map.shape), self.optimal_path)
        self.show_potential_field() 
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
        self.max_steps = 5000
        self.steps = 0 
        self.recent_obstacles = []
        
        self.action_space = spaces.Box(low=np.array([0.0, -2.84]), high=np.array([0.22, 2.84]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0, -np.pi, 0.0]), 
                                            high=np.array([10.0, 10.0, np.pi, 12.0]), 
                                            shape=(4,), dtype=np.float32)
        
        self.timer = self.create_timer(0.05, self._timer_callback)


    def _timer_callback(self):
        pass 

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # def scan_callback(self, msg):
        # self.obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max else msg.range_max for r in msg.ranges]
    
    def scan_callback(self, msg):
        raw_obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max 
                        else msg.range_max for r in msg.ranges]

        self.obstacles = raw_obstacles
        min_obstacle_dist = min(raw_obstacles) if raw_obstacles else msg.range_max
        logger.info(f'Min obstacle dist: {min_obstacle_dist}') 

        # Конвертация координат
        current_x, current_y = world_to_map(
            (self.current_x, self.current_y),
            resolution=0.05,
            origin=(-4.86, -7.36),
            map_offset=(45, 15),
            map_shape=self.grid_map.shape
        )

        logger.info(f'Current_x, current_y: {current_x, current_y}') 
        # Получаем значение поля
        if 0 <= current_x < self.grid_map.shape[1] and 0 <= current_y < self.grid_map.shape[0]:
            potential_value = self.potential_field[current_y, current_x]
            # print(f'Potential value: {potential_value}')
            logger.info(f'Potential value: {potential_value}') 
        else:
            potential_value = 1

        # Проверяем, было ли препятствие на большинстве последних кадров
        camera_obstacle_count = sum(self.camera_history)  # Считаем количество True
        camera_obstacle_threshold = 12  

        # Если LiDAR видит препятствие вблизи
        if min_obstacle_dist < 0.2:
            # Принудительно увеличиваем потенциал в случае обнаружения препятствия
            potential_value = max(potential_value, np.percentile(self.potential_field, 90))
    
        # Логика определения препятствий
        self.lidar_obstacle_detected = (
            (min_obstacle_dist < 0.2) and (  # Лидар обнаружил близкое препятствие И
                (potential_value > 3) or  # Более чувствительный порог
                # (potential_value > np.percentile(self.potential_field, 90)) or  # Высокий потенциал
                (camera_obstacle_count >= camera_obstacle_threshold)  # Камера часто видела препятствие
            )
        )
        self.recent_obstacles.append(self.lidar_obstacle_detected)
        if len(self.recent_obstacles) > 5:  # Храним только 5 последних значений
            self.recent_obstacles.pop(0)

        # Если хотя бы 3 из 5 последних измерений показали препятствие — считаем его подтверждённым
        if sum(self.recent_obstacles) >= 3:
            self.lidar_obstacle_detected = True

        logger.info(f'Detect of obstacle: {self.lidar_obstacle_detected}') 

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if cv_image is not None:
                detected = self.process_camera_image(cv_image)
                self.camera_obstacle_detected = detected
                self.camera_history.append(detected)  # Добавляем в историю
            else:
                self.camera_obstacle_detected = False
                self.camera_history.append(False)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            self.camera_obstacle_detected = False
            self.camera_history.append(False)
    
    def show_potential_field(self):

        goal_pixel = world_to_map(self.goal, resolution=0.05, origin=(-4.86, -7.36), map_offset=(45, 15),map_shape=self.grid_map.shape)
        plt.figure(figsize=(10, 8))
        plt.imshow(self.potential_field, cmap='jet')
        plt.colorbar(label='Potential')
        plt.scatter(goal_pixel[0], goal_pixel[1], c='green', s=200, marker='*', label='Goal')
        plt.title("Potential Field Visualization")
        plt.legend()
        plt.show()

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
    
   
    def compute_potential_reward(self, state, goal, intermediate_points, obstacle_detected, k_att=0.3, k_rep=2.0, d0=5.0, lam=0.5):
        current_x, current_y, _, min_obstacle_dist = state

        # Преобразуем координаты в пиксельные
        current_x, current_y = world_to_map(
            (current_x, current_y),
            resolution=0.05,
            origin=(-4.86, -7.36),
            map_offset=(45, 15),
            map_shape=self.grid_map.shape
        )

        goal_x, goal_y = world_to_map(goal, 0.05, (-4.86, -7.36), (45, 15), self.grid_map.shape)

        potential_value = self.potential_field[current_y, current_x] 
        delta_potential = self.prev_potential - potential_value
        R_potential = np.clip(-delta_potential, -1.0, 1.0)
        self.prev_potential = potential_value

        # === Притяжение к промежуточной точке ===
        R_intermediate = 0.0
        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = current_x, current_y

        if intermediate_points:
            nearest_intermediate = min(intermediate_points, key=lambda p: np.linalg.norm([current_x - p[0], current_y - p[1]]))
            prev_dist = np.linalg.norm([self.prev_x - nearest_intermediate[0], self.prev_y - nearest_intermediate[1]])
            curr_dist = np.linalg.norm([current_x - nearest_intermediate[0], current_y - nearest_intermediate[1]])
            R_intermediate = np.clip(k_att * (prev_dist - curr_dist), -1.0, 1.0)

        # === Градиент к цели ===
        motion_direction = np.array([np.cos(self.current_yaw), np.sin(self.current_yaw)])
        direction_to_goal = np.array([goal_x - current_x, goal_y - current_y])
        norm = np.linalg.norm(direction_to_goal)
        direction_to_goal = direction_to_goal / norm if norm > 0 else np.array([1.0, 0.0])
        projection = np.dot(motion_direction, direction_to_goal)
        grad_reward = np.clip(lam * projection, -1.0, 1.0)

        if self.lidar_obstacle_detected and projection < 0:
            grad_reward -= 2.0

        # === Отталкивающее поле ===
        R_repulsive = 0.0
        if min_obstacle_dist < d0 and obstacle_detected:
            R_repulsive = -k_rep * (1 / min_obstacle_dist - 1 / d0) ** 2
            R_repulsive = np.clip(R_repulsive, -10.0, 0.0)

        # === Штраф за ложный путь ===
        R_fake_path = 0.0
        if self.lidar_obstacle_detected and (potential_value - self.prev_potential > 0.2 or potential_value == self.prev_potential):
            R_fake_path = -5.0

        if self.steps % 5 != 0:
            R_repulsive = 0
            R_fake_path = 0
        # === Суммарная награда ===
        total_reward = (
            1.0 * R_potential +
            1.0 * R_intermediate +
            1.0 * grad_reward +
            1.0 * R_repulsive +
            1.0 * R_fake_path
        )
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # === Логгирование ===
        logger.info(f"R_potential: {R_potential:.2f}, R_intermediate: {R_intermediate:.2f}, grad: {grad_reward:.2f}, rep: {R_repulsive:.2f}, fake: {R_fake_path:.2f}, total: {total_reward:.2f}")

        self.prev_x, self.prev_y = current_x, current_y
        return total_reward

    
    def compute_deviation_from_path(self, current_pos):
        """
        Вычисляет минимальное расстояние от текущей позиции агента до оптимального пути.
        :param current_pos: (x, y) текущая позиция агента.
        :param optimal_path: список точек пути [(x1, y1), (x2, y2), ...]
        :return: минимальное расстояние до пути (скаляр).
        """
        path_points = np.array(self.optimal_path)
        distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
        min_distance = np.min(distances)
        return min_distance

    def get_deviation_penalty(self, current_pos, max_penalty=1):
        """
        Рассчитывает штраф за отклонение от пути.
        :param current_pos: (x, y) текущая позиция агента.
        :param optimal_path: список точек пути [(x1, y1), (x2, y2), ...].
        :param max_penalty: максимальный штраф за сильное отклонение.
        :return: штраф (скаляр, отрицательный).
        """
        if current_pos.ndim == 2 and current_pos.shape[0] == 1:
            current_pos = current_pos[0]

        state = world_to_map(current_pos, resolution = 0.05, origin = (-4.86, -7.36),  map_offset = (45, 15), map_shape = self.grid_map.shape)
        
        deviation = self.compute_deviation_from_path(state)
        
        # Можно сделать штраф линейным или экспоненциальным в зависимости от задачи
        penalty = -min(max_penalty, deviation)  # Чем дальше от пути, тем больше штраф
        return penalty

    def step(self, action):
        cmd_msg = Twist()
        linear = float(np.clip(action[0], 0.0, 0.22))
        angular = float(np.clip(action[1], -2.84, 2.84))

        cmd_msg.linear.x = linear
        cmd_msg.angular.z = angular
        rclpy.spin_once(self, timeout_sec=0.1) 
        self.publisher_.publish(cmd_msg)
    
        self.steps += 1
        
        # print(self.obstacles)
        distance = math.sqrt((self.target_x - self.current_x) ** 2 + (self.target_y - self.current_y) ** 2)
        angle_to_goal = math.atan2(self.target_y - self.current_y, self.target_x - self.current_x)
        angle_diff = (angle_to_goal - self.current_yaw + np.pi) % (2 * np.pi) - np.pi

        min_obstacle_dist = min(self.obstacles) if self.obstacles else 3.5

        obstacle_detected = self.lidar_obstacle_detected or self.camera_obstacle_detected
        state = np.array([self.current_x, self.current_y, angle_diff, min_obstacle_dist])

        # distance_rate = (self.past_distance - distance)
        # print(min_obstacle_dist)
        reward_potent_val = self.compute_potential_reward(state, self.goal, self.optimal_path, obstacle_detected)
        reward_optimal_path = self.get_deviation_penalty(state[:2])

        # Награда за приближение к цели
        reward_goal_progress = 0.0
        if self.prev_distance is not None:
            delta = self.prev_distance - distance
            reward_goal_progress = np.clip(delta * 100.0, -10.0, 10.0)  # усиленный сигнал
        self.prev_distance = distance

        # Общая награда
        reward = reward_potent_val + reward_goal_progress + reward_optimal_path

        # ====== Терминальные случаи ======
        done = False

        # 1. Обнаружено препятствие
        if obstacle_detected:
            self.obstacle_count += 1
            reward -= 5  # менее жёстко
            if self.obstacle_count >= 100:
                done = True
                self.obstacle_count = 0
                print("Episode terminated due to repeated obstacle detection")

        # 2. Очень близко к препятствию
        if min_obstacle_dist < 0.5:
            reward -= 10 * (0.5 - min_obstacle_dist)

        # 3. Достигли цели
        if distance < 0.3:
            reward += 300  # усиленное вознаграждение
            done = True
            print("Goal reached!")

        # 4. Превышен лимит шагов
        if self.steps >= self.max_steps:
            reward -= 50  # чуть мягче
            done = True
            print("Episode terminated due to step limit")

        # 5. Безопасное ограничение награды
        reward = np.clip(reward, -20.0, 30.0)

        return state, reward, done, {}


    def reset(self):
        # === 1. Остановить движение ===
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0  
        cmd_msg.angular.z = 0.0
        self.publisher_.publish(cmd_msg)  
        rclpy.spin_once(self, timeout_sec=0.1)

        # === 2. Сброс симуляции ===
        client = self.create_client(Empty, '/reset_simulation')
        request = Empty.Request()
        if client.wait_for_service(timeout_sec=1.0):
            future = client.call_async(request)
            while not future.done():
                rclpy.spin_once(self, timeout_sec=0.1)
        else:
            self.get_logger().warn('Gazebo reset service not available!')

        # === 3. Ждём, пока обновится одометрия после сброса ===
        timeout_counter = 0
        while (abs(self.current_x) < 0.01 and abs(self.current_y) < 0.01 and 
            abs(self.current_yaw) < 1e-3 and timeout_counter < 50):
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_counter += 1

        # === 4. Обновляем переменные среды ===
        self.current_x = -2.0
        self.current_y = -0.5
        self.current_yaw = 0.0
        self.steps = 0
        self.prev_distance = None
        self.obstacles = []
        self.camera_obstacle_detected = False
        self.prev_potential = 0
        self.prev_x = None
        self.prev_y = None
        self.obstacle_count = 0

        # === 5. Возвращаем корректное состояние ===
        min_obstacle_dist = 3.5 if not self.obstacles else min(self.obstacles)
        return np.array([self.current_x, self.current_y, self.current_yaw, min_obstacle_dist])

 
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)