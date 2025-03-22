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

lookahead_distance = 0.15
speed = 0.2
expansion_size = 5

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
    closed_list = dict()  

    open_list.put((start_node.f, start_node))
    closed_list[start_node.position] = start_node

    while not open_list.empty():
        current_node = open_list.get()[1]

        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        neighbors = [(0,1), (1,0), (0,-1), (-1,0),
                    (1,1), (1,-1), (-1,1), (-1,-1)]

        for new_position in neighbors:
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1]
            )

            if node_position[0] < 0 or node_position[0] >= grid.shape[0]:
                continue
            if node_position[1] < 0 or node_position[1] >= grid.shape[1]:
                continue
            if grid[node_position[0]][node_position[1]] == 1:
                continue

            if current_node.parent and line_of_sight(current_node.parent.position, node_position, grid):
                new_g = current_node.parent.g + heuristic(current_node.parent.position, node_position)
                tentative_node = node(current_node.parent, node_position)
            else:
                new_g = current_node.g + heuristic(current_node.position, node_position)
                tentative_node = node(current_node, node_position)

            dynamic_cost = occupations[node_position[1]][node_position[0]]
            penalty = penalties[node_position[1]][node_position[0]]
            new_g += dynamic_cost * penalty

            if node_position in closed_list:
                existing_node = closed_list[node_position]
                if new_g >= existing_node.g:
                    continue
                closed_list.pop(node_position)

            tentative_node.g = new_g
            tentative_node.h = heuristic(tentative_node.position, end_node.position)
            tentative_node.f = tentative_node.g + tentative_node.h

            open_list.put((tentative_node.f, tentative_node))
            closed_list[tentative_node.position] = tentative_node

    return None

def line_of_sight(start, end, grid):
    x0, y0 = start
    x1, y1 = end
    
    if x0 < 0 or x0 >= grid.shape[0] or y0 < 0 or y0 >= grid.shape[1]:
        return False
    if x1 < 0 or x1 >= grid.shape[0] or y1 < 0 or y1 >= grid.shape[1]:
        return False
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if grid[x0][y0] == 1:
            return False
            
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True

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

def pure_pursuit(current_x, current_y, current_heading, path, index):
    global lookahead_distance
    closest_point = None
    v = speed
    
    for i in range(index, len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path) - 1
    
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    
    if desired_steering_angle > math.pi / 6 or desired_steering_angle < -math.pi / 6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi / 4
        v = 0.0
    return v, desired_steering_angle, index

def costmap(data, width, height, resolution):
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    
    kernel_size = 2 * expansion_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1
    cv2.imwrite('/home/sa/turtlebot3_ws/src/theta_star/debug/debug_map.png', (result * 2.55).astype(np.uint8)) 
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
            }
        }
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
        # # test1.sdf
        self.goals = [(3.99715, -1.6586), (3.50709, 1.44957), (1.25942, 1.25394), (-0.689823, 2.26387), (-2.50234, 2.11622), (-1.7666, 0.285539), (-4.07267, 2.43495)]     
        
        # world3.sdf
        # self.goals = [(-1.48882, 0.30411), (4.31674, 1.16783), (2.74416, -3.29382), (-1.68252, -2.83125), (-5.24987, 1.06107), (-1.27769, 3.59045)]   
        
        self.map_init_time = 0.0
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("Wait for targets")
        self.saved = False
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
                            other['occupations'][y][x] += 1

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
            self.grid = np.where((self.grid == 100) | (self.grid == -1), 1, 0).astype(np.int8)
            np.savetxt('/home/sa/turtlebot3_ws/src/theta_star/debug/debug.txt', self.grid, fmt='%d')
            self.map_initialized = True
            self.map_init_time = time.time()
            
            for robot_name in self.robots:
                self.robots[robot_name]['occupations'] = np.zeros((self.height, self.width), dtype=float)
                self.robots[robot_name]['penalties'] = np.ones((self.height, self.width), dtype=float) 
    def timer_callback(self):
        # test1.sdf
        if not self.map_initialized or time.time() - self.map_init_time < 90:
        
        # world3.sdf
        # if not self.map_initialized or time.time() - self.map_init_time < 20:
            print(time.time() - self.map_init_time)
            return
        for robot in self.robots.values():
            robot['occupations'] = np.maximum(robot['occupations'] - 0.07, 0)

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
            self.get_logger().info("Generating new goal..........")
            ind = np.random.randint(0, len(self.goals))
            goal_world = self.goals[ind]

            goal = self.world_to_grid(goal_world[0], goal_world[1])
            other_name = 'tb0' if robot_name == 'tb1' else 'tb1'
            self.update_occupations(robot_name)

            self.robots[robot_name]['goal'] = goal  

            self.get_logger().info(f"New goal for {robot_name}: {goal}")
            robot = self.robots[robot_name]
                        
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

        if not self.saved:
            debug_grid = self.grid.copy()
            debug_grid[start[1]][start[0]] = 2
            
            for goal in self.goals:
                goal = self.world_to_grid(goal[0],goal[1]) 

                debug_grid[goal[1]][goal[0]] = 3
            
            np.savetxt('/home/sa/turtlebot3_ws/src/theta_star/debug/debug.txt', debug_grid, fmt='%d')
            print('grid saved')
            self.saved = True
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

        v, angle, index = pure_pursuit(
            robot['x'], robot['y'], robot['yaw'],
            robot['path'], robot['index']
        )
        
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = angle
        robot['cmd_vel_pub'].publish(twist)
        
        if self.distance_to_goal(robot) < 0.1:
            self.get_logger().info("The goal reached!!!!!!!!!!!!!!!!!!!!")
            robot['goal'] = None
            
    def distance_to_goal(self, robot):
        if not robot['path']:
            return float('inf')
        init_point = self.world_to_grid(robot['x'], robot['y'])
        last_point = robot['goal']
        return math.hypot(init_point[0] - last_point[0], init_point[1] - last_point[1])
            
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