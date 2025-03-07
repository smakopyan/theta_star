import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue
from rclpy.qos import QoSProfile
import math
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist


lookahead_distance = 0.25
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
    data = np.array(data).reshape(height, width)
    wall = np.where(data == 100)
    for i in range(-expansion_size, expansion_size + 1):
        for j in range(-expansion_size, expansion_size + 1):
            if i == 0 and j == 0:
                continue
            x = wall[0] + i
            y = wall[1] + j
            x = np.clip(x, 0, height - 1)
            y = np.clip(y, 0, width - 1)
            data[x, y] = 100
    data = data.astype(float) * resolution
    return data

class Navigation(Node):
    def __init__(self, namespace0, namespace1):
        super().__init__('Navigation')
        self.map_init = False
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
        self.flag = 0
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("Wait for targets")
    def create_odom_callback(self, namespace):
        def callback(msg):
            if not self.map_init:
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
                other = self.robots['tb0'] if namespace == 'tb1' else self.robots['tb0'] 
                other['occupations'][x_grid][y_grid] += 1
                for i in range(-expansion_size, expansion_size + 1):
                    for j in range(-expansion_size, expansion_size + 1):
                        if i == 0 and j == 0:
                            continue
                        x = x_grid + i
                        y = y_grid + j
                        x = np.clip(x, 0, self.height - 1)
                        y = np.clip(y, 0, self.width - 1)
                        if 0 <= x < self.height and 0 <= y < self.width:
                            other['occupations'][x][y] += 0.6

            robot['odom_init'] = True
        return callback
    def map_callback(self, msg):
        if not self.map_init:
            self.map_resolution = msg.info.resolution
            self.map_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ]
            self.get_logger().info(f'Map resolution: {self.map_resolution}, Origin: {self.map_origin}')
            self.grid = costmap(msg.data, msg.info.width, msg.info.height, self.map_resolution)
            self.grid[self.grid == -1] = 1
            self.grid[self.grid >= self.map_resolution * 100] = 1
            self.width = msg.info.width
            self.height = msg.info.height
            self.map_init = True

            for robot_name in self.robots:
                self.robots[robot_name]['occupations'] = np.zeros((self.height, self.width), dtype=float)
                self.robots[robot_name]['penalties'] = np.ones((self.height, self.width), dtype=float) 
    def timer_callback(self):
        if not self.map_init:
            return
        for robot in self.robots.values():
            robot['occupations'] = np.maximum(robot['occupations'] - 0.1, 0)

        for robot_name, robot_data in self.robots.items():
            if not robot_data['odom_init']:
                continue
            if robot_data['goal'] is None:
                self.generate_new_goal(robot_name)
                continue
            
            self.navigate(robot_name)
    def generate_new_goal(self, robot_name):
        if not self.map_init:
            return
        while True:
            goal_x = np.random.randint(0, self.height-1)
            goal_y = np.random.randint(0, self.width-1)
            if self.grid[goal_x][goal_y] == 0:
                self.robots[robot_name]['goal'] = (goal_x, goal_y)
                self.get_logger().info(f"New goal for {robot_name}: {goal_x}, {goal_y}")
                robot = self.robots[robot_name]
                robot['occupations'] = np.zeros((self.height, self.width))
                break
    
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
                    self.robots[robot_name]['occupations'][x][y] += 1.0
    
    def navigate(self, robot_name):
        if not self.map_init:
            return
        robot = self.robots[robot_name]
        start = self.world_to_grid(robot['x'], robot['y'])
        goal = robot['goal']
        
        self.update_occupations(robot_name)
        
        if start != goal:
            grid = self.grid
            grid[start[0]][start[1]] = 0
        else:
            robot['goal'] = None
        
        path = theta_star(
            start, goal, grid,
            robot['occupations'],
            robot['penalties']
        )
        
        if path is None:
            self.get_logger().warn(f"No path found for {robot_name}")
            robot['goal'] = None
            return
            
        robot['path'] = [self.grid_to_world(*p) for p in path]
        
        v, angle, index = pure_pursuit(
            robot['x'], robot['y'], robot['yaw'],
            robot['path'], robot['index']
        )
        
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = angle
        robot['cmd_vel_pub'].publish(twist)
        
        if self.distance_to_goal(robot) < 0.1:
            robot['goal'] = None
    def distance_to_goal(self, robot):
        if not robot['path']:
            return float('inf')
        last_point = robot['path'][-1]
        return math.hypot(robot['x'] - last_point[0], robot['y'] - last_point[1])
            
    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution)
        x_grid = np.clip(x_grid, 0, self.height - 1)
        y_grid = np.clip(y_grid, 0, self.width - 1)
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