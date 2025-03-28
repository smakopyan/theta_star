import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue
from rclpy.qos import QoSProfile
import math
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
import time
import cv2

lookahead_distance = 0.3
speed = 0.2
expansion_size = 4

class node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  
        self.h = 0  
        self.f = 0  
        
    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f 

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def theta_star(start, end, grid, occupations, penalties):
    start_node = node(None, start)
    end_node = node(None, end)

    open_list = PriorityQueue()
    closed_list = []

    open_list.put((start_node.f, start_node))
    
    while not open_list.empty():
        current_node = open_list.get()[1]
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1] 

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for new_position in neighbors:
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])

            if (0 <= node_position[0] < grid.shape[0]) and (0 <= node_position[1] < grid.shape[1]):
                if grid[node_position[0]][node_position[1]] != 1:  
                    neighbor_node = node(current_node, node_position)

                    if neighbor_node in closed_list:
                        continue

                    base_cost = 1.0
                    dynamic_cost = occupations[node_position[0]][node_position[1]]
                    penalty = penalties[node_position[0]][node_position[1]]
                    total_cost = base_cost + dynamic_cost * penalty

                    neighbor_node.g = current_node.g + total_cost
                    neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    if add_to_open(open_list, neighbor_node):
                        open_list.put((neighbor_node.f, neighbor_node))

    return None

def add_to_open(open_list, neighbor):
    for item in open_list.queue:
        if neighbor == item[1] and neighbor.g >= item[1].g:
            return False
    return True

def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z
# def costmap(data, width, height, resolution):
#     data = np.array(data).reshape(height, width)
#     wall = np.where(data == 100)
#     for i in range(-expansion_size, expansion_size + 1):
#         for j in range(-expansion_size, expansion_size + 1):
#             if i == 0 and j == 0:
#                 continue
#             x = wall[0] + i
#             y = wall[1] + j
#             x = np.clip(x, 0, height - 1)
#             y = np.clip(y, 0, width - 1)
#             data[x, y] = 100
#     data = data.astype(float) * resolution
#     return data

def costmap(data, width, height, resolution):
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    
    kernel_size = 2 * expansion_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1
    cv2.imwrite('debug_map.png', (result * 2.55).astype(np.uint8)) 
    np.savetxt('debug.txt',result)
    # cv2.imwrite('debug_map.png', result)
    return result.flatten().tolist()


class Navigation(Node):
    def __init__(self, namespace0, namespace1):
        super().__init__('Navigation')
        self.map_initialized = False
        self.robots = {
            namespace0: {
                'x': 0.0,
                'y': 0.0,
                'yaw': 0.0,
                'goal': None,
                'path': [],
                'index': 0,
                'odom_init': False,
                'occupations': None,
                'penalties': None,       
                'last_time':  self.get_clock().now(),
                'previous_angle_error': 0.0,
                'i_angle_error':0.0,
            },
            namespace1: {
                'x': 0.0,
                'y': 0.0,
                'yaw': 0.0,
                'goal': None,
                'path': [],
                'index': 0,
                'odom_init': False,
                'occupations': None,
                'penalties': None,   
                'last_time':  self.get_clock().now(),
                'previous_angle_error': 0.0,
                'i_angle_error':0.0,  
            }
        }
        
        self.last_time = self.get_clock().now()
        self.previous_angle_error = 0.0
        self.i_angle_error = 0.0
        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        for namespace in [namespace0, namespace1]:
            self.create_subscription(
                Odometry, 
                f'/{namespace}/odom',
                self.create_odom_callback(namespace),
                10
            )
            self.robots[namespace]['cmd_vel_pub'] = self.create_publisher(
                Twist, 
                f'/{namespace}/cmd_vel', 
                10
            )
        #test1.sdf
        self.goals = [(3.99715, -1.6586), (3.50709, 1.44957), (1.25942, 1.25394), (-0.689823, 2.26387),
                      (-2.50234, 2.11622), (-1.7666, 0.285539), (-4.07267, 2.43495)]        
        
        self.map_init_time = 0.0
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("Wait for targets")
    
    def create_odom_callback(self, namespace):
        def callback(msg):
            if not self.map_initialized:
                return
            robot = self.robots[namespace]
            robot['x'] = msg.pose.pose.position.x
            robot['y'] = msg.pose.pose.position.y
            robot['yaw'] = euler_from_quaternion(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )
            x_grid, y_grid = self.world_to_grid(msg.pose.pose.position.x, 
                                                msg.pose.pose.position.y)
            expansion_size = 6
            if 0 <= x_grid < self.height and 0 <= y_grid < self.width:
                other = self.robots['tb0'] if namespace == 'tb1' else self.robots['tb1'] 
                other['occupations'][y_grid][x_grid] += 1
                for i in range(-expansion_size, expansion_size + 1):
                    for j in range(-expansion_size, expansion_size + 1):
                        if i == 0 and j == 0:
                            continue
                        x = x_grid + i
                        y = y_grid + j
                        x = np.clip(x, 0, self.height - 1)
                        y = np.clip(y, 0, self.width - 1)
                        if 0 <= x < self.height and 0 <= y < self.width:
                            other['occupations'][y][x] += 0.6

            robot['odom_init'] = True
        return callback
    def map_callback(self, msg):
        if not self.map_initialized:
            self.map_resolution = msg.info.resolution
            self.map_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ]
            self.width = msg.info.width
            self.height = msg.info.height
            self.grid = costmap(msg.data, self.width, self.height, self.map_resolution)
            self.grid = np.array(self.grid).reshape(self.height, self.width)
            self.grid = np.where((self.grid == 100) | (self.grid == -1), 1, 0)
            
            self.map_initialized = True
            self.map_init_time = time.time()
            
            for robot_name in self.robots:
                self.robots[robot_name]['occupations'] = np.zeros((self.height, self.width), dtype=float)
                self.robots[robot_name]['penalties'] = np.ones((self.height, self.width), dtype=float) 
    
    def timer_callback(self):

        if not self.map_initialized or time.time() - self.map_init_time < 90:
            print(time.time() - self.map_init_time)
            return
        for robot in self.robots.values():
            robot['occupations'] = np.maximum(robot['occupations'] - 0.1, 0)

        for robot_name, robot_data in self.robots.items():
            # if not robot_data['odom_init']:
            #     continue
            if robot_data['goal'] is None:
                self.generate_new_goal(robot_name)
                continue
            
            self.navigate(robot_name)
            
    def generate_new_goal(self, robot_name):
        if not self.map_initialized:
            return
        if self.robots[robot_name]['goal'] == None:
            robot = self.robots[robot_name]
            # robot['occupations'] = np.zeros((self.height, self.width))

            self.get_logger().info("Generating new goal..........")
            ind = np.random.randint(0, len(self.goals))
            goal_world = self.goals[ind]

            goal = self.world_to_grid(goal_world[0], goal_world[1])
            other_name = 'tb0' if robot_name == 'tb1' else 'tb1'
            self.robots[robot_name]['goal'] = goal  
            self.update_occupations(robot_name)

            self.get_logger().info(f"New goal for {robot_name}: {goal}")
            
            if self.robots[robot_name]['goal'] == self.robots[other_name]['goal']:
                self.robots[robot_name]['goal'] = None
                self.generate_new_goal(robot_name)
        
    def update_occupations(self, robot_name):
        other_name = 'tb1' if robot_name == 'tb0' else 'tb0'
        other_pos = self.world_to_grid(
            self.robots[other_name]['x'],
            self.robots[other_name]['y']
        )
        
        self.robots[robot_name]['occupations'] = np.zeros((self.height, self.width), dtype=float)
            
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                x = other_pos[0] + dx
                y = other_pos[1] + dy
                if 0 <= x < self.height and 0 <= y < self.width:
                    self.robots[robot_name]['occupations'][y][x] += 1.0
    
    def navigate(self, robot_name):
        if not self.map_initialized:
            return
        robot = self.robots[robot_name]
        start = self.world_to_grid(robot['x'], robot['y']) 
        goal = robot['goal']
        
        self.update_occupations(robot_name)
        
        grid = self.grid
        grid[start[1]][start[0]] = 0

        path = theta_star(
            (start[1], start[0]), (goal[1], goal[0]), grid,
            robot['occupations'],
            robot['penalties']
        )
        
        if path is None:
            self.get_logger().warn(f"No path found for {robot_name}")
            robot['goal'] = None
            return
            
        robot['path'] = [self.grid_to_world(i[1], i[0]) for i in path]
        
        robot['current_path_index']=0
        self.follow_path(robot_name)
        if self.distance_to_goal(robot) < 0.1:
            robot['goal'] = None
     
    def follow_path(self, robot_name):
        robot = self.robots[robot_name]
        if robot['path'] and robot['current_path_index'] < len(robot['path']):
            current_goal = robot['path'][robot['current_path_index']]
            error_x = current_goal[0] - robot['x']
            error_y = current_goal[1] - robot['y']

            distance = math.sqrt(pow(error_x, 2) + pow(error_y, 2))
            if distance < 0.2: 
                robot['current_path_index'] += 1
                if robot['current_path_index'] >= len(robot['path']):
                    self.get_logger().info("Reached the end of the path.")
                    vel_msg = Twist()
                    robot['cmd_vel_pub'].publish(vel_msg)  
                    return

            current_time = self.get_clock().now()
            dt = (current_time - robot['last_time']).nanoseconds / 1e9
            robot['last_time'] = current_time  # Обновляем время для конкретного робота

            angle_to_goal = math.atan2(error_y, error_x)
            angle_error = self.normalize_angle(angle_to_goal - robot['yaw'])
            d_angle_error = (angle_error - robot['previous_angle_error']) / dt if dt > 0 else 0
            robot['previous_angle_error'] = angle_error
            robot['i_angle_error'] += angle_error

            kp = 5.0
            kd = 0.5
            ki = 0.000008

            angular_velocity = kp * angle_error + ki*robot['i_angle_error'] + kd * d_angle_error
            angular_velocity = max(-1.0, min(1.0, angular_velocity))  

            linear_velocity = 0.3 if abs(angle_error) < 0.05 else 0.05

            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_velocity
            robot['cmd_vel_pub'].publish(twist)
            
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def distance_to_goal(self, robot):
        if not robot['path']:
            return float('inf')
        last_point = robot['path'][-1]
        return math.hypot(robot['x'] - last_point[0], robot['y'] - last_point[1])
            
    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_grid = int((y_world - self.map_origin[1])/ self.map_resolution)
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid):
        x_world = x_grid * self.map_resolution + self.map_origin[0]
        y_world = y_grid * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(namespace0='tb0', namespace1='tb1')
    rclpy.spin(nav)
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()