import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue
from rclpy.qos import QoSProfile
import math
from nav_msgs.msg import OccupancyGrid, Odometry 
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
import time
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# lookahead_distance = 0.15
# speed = 0.1

lookahead_distance = 0.2
speed = 0.1
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

            base_cost = 1.0
            dynamic_cost = occupations[node_position[0]][node_position[1]]
            penalty = penalties[node_position[0]][node_position[1]]
            new_g += base_cost + 2* dynamic_cost * penalty

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


def costmap(data, width, height, resolution):
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    
    kernel_size = 2 * expansion_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1
    return result.flatten().tolist()


class Robot():
    def __init__(self, namespace):
        self.namespace = namespace
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.goal = None
        self.path = []
        self.index = 0
        self.occupations = None
        self.penalties = None
        self.cmd_vel_pub = None
        self.occupancy_marker_pub = None
        self.laser_data = None  
        self.obstacle_detected = False
        self.emergency_stop = False 
        self.path_marker_pub = None 
        self.lookahead_marker_pub = None

class Navigation(Node):
    def __init__(self, namespace0, namespace1):
        super().__init__('Navigation')
        self.map_initialized = False
        self.tb0 = Robot(namespace0)
        self.tb1 = Robot(namespace1)
        self.robots = [self.tb0, self.tb1]

        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        for robot in self.robots:
            self.create_subscription(
                Odometry, 
                f'/{robot.namespace}/odom',
                self.create_odom_callback(robot),
                10
            )
            robot.cmd_vel_pub = self.create_publisher(
                Twist, 
                f'/{robot.namespace}/cmd_vel', 
                10
            )
            robot.occupancy_marker_pub = self.create_publisher(
                Marker, 
                f'/{robot.namespace}/occupations_marker', 
                10
            )

            self.create_subscription(
                LaserScan,
                f'/{robot.namespace}/scan',
                self.create_laser_callback(robot),
                10
            )

            robot.path_marker_pub = self.create_publisher(
                Marker, 
                f'/{robot.namespace}/path_marker', 
                10
            )
            robot.lookahead_marker_pub = self.create_publisher(
                Marker, 
                f'/{robot.namespace}/lookahead_marker', 
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
    
    def create_laser_callback(self, robot):
        def callback(msg):
            if not self.map_initialized:
                return
            robot.laser_data = msg.ranges
            robot.obstacle_detected = any(
                distance < 0.3 for distance in msg.ranges if distance > 0.
            )
        return callback

    def create_odom_callback(self, robot):
        def callback(msg):
            if not self.map_initialized:
                return
            robot.x = msg.pose.pose.position.x
            robot.y = msg.pose.pose.position.y
            robot.yaw = euler_from_quaternion(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )
            x_grid, y_grid = self.world_to_grid(msg.pose.pose.position.x, 
                                                msg.pose.pose.position.y)
            expansion_size = 5
            if 0 <= x_grid < self.height and 0 <= y_grid < self.width:
                other = self.tb0 if robot == self.tb1 else self.tb1
                other.occupations[y_grid][x_grid] += 1
                for i in range(-expansion_size, expansion_size + 1):
                    for j in range(-expansion_size, expansion_size + 1):
                        if i == 0 and j == 0:
                            continue
                        x = x_grid + i
                        y = y_grid + j
                        x = np.clip(x, 0, self.height - 1)
                        y = np.clip(y, 0, self.width - 1)
                        if 0 <= x < self.height and 0 <= y < self.width:
                            other.occupations[y][x] += 1
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
            self.map_initialized = True
            self.map_init_time = time.time()
            
            for robot in self.robots:
                robot.occupations = np.zeros((self.height, self.width), dtype=float)
                robot.penalties = np.ones((self.height, self.width), dtype=float) 
    def timer_callback(self):
        # test1.sdf
        if not self.map_initialized or time.time() - self.map_init_time < 90:
        
        # world3.sdf
        # if not self.map_initialized or time.time() - self.map_init_time < 20:
            print(time.time() - self.map_init_time)
            return
        # for robot in self.robots:
            # robot.occupations = np.maximum(robot.occupations - 0.07, 0)

        for robot in self.robots:
            if robot.goal is None:
                self.generate_new_goal(robot)
                continue
            self.visualize_occupations(robot)
            self.navigate(robot)
            
    def generate_new_goal(self, robot):
        if not self.map_initialized:
            return
        if robot.goal == None:
            self.get_logger().info("Generating new goal..........")
            ind = np.random.randint(0, len(self.goals))
            goal_world = self.goals[ind]

            goal = self.world_to_grid(goal_world[0], goal_world[1])
            other = self.tb0 if robot == self.tb1 else self.tb1
            self.update_occupations(robot)

            robot.goal = goal  

            self.get_logger().info(f"New goal for {robot.namespace}: {goal}")
                        
            if np.array_equal(robot.goal, other.goal):
                robot.goal = None
                self.generate_new_goal(robot)
        
    def update_occupations(self, robot):
        other = self.tb0 if robot == self.tb1 else self.tb1
        other_pos = self.world_to_grid(
            other.x,
            other.y
        )
        robot.occupations = np.zeros((self.height, self.width), dtype=float)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                x = other_pos[0] + dx
                y = other_pos[1] + dy
                if 0 <= x < self.height and 0 <= y < self.width:
                    robot.occupations[y][x] += 1.0
    
    def navigate(self, robot):
        if not self.map_initialized:
            return
        start = self.world_to_grid(robot.x, robot.y) 
        goal = robot.goal
        self.update_occupations(robot)
        
        other = self.tb0 if robot == self.tb1 else self.tb1

        grid = self.grid
        grid[start[1]][start[0]] = 0
        path = theta_star(
            (start[1], start[0]), (goal[1], goal[0]), grid,
            robot.occupations,
            robot.penalties
        )
                
        if path is None:
            self.get_logger().warn(f"No path found for {robot.namespace}")
            # robot['goal'] = None
            return
            
        robot.path = [self.grid_to_world(i[1], i[0]) for i in path]
        
        if path is not None:
            self.visualize_path(robot)
        else:
            pass
        

        v, angle, closest_point = self.pure_pursuit(
            robot.x, robot.y, robot.yaw,
            robot.path, robot.index, robot
        )

        if closest_point is not None:
            self.visualize_lookahead(robot, closest_point)
        
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = angle
        robot.cmd_vel_pub.publish(twist)
        
        if robot.laser_data:
            v = angle = None
            for i in range(60):
                if robot.laser_data[i] < 0.3:
                        v = 0.08
                        angle = -math.pi/4 
                        break
                if v == None:
                    for i in range(300,360):
                        if robot.laser_data[i] < 0.3:
                            v = 0.08
                            angle = math.pi/4
                            break
                if v and angle:
                    twist.linear.x = v
                    twist.angular.z = angle
                    robot.cmd_vel_pub.publish(twist)

        if math.hypot(robot.x - other.x, robot.y - other.y) < 0.6:
            self.navigate(robot)

        if self.distance_to_goal(robot) < 0.15:
            self.get_logger().info("The goal reached!!!!!!!!!!!!!!!!!!!!")
            robot.goal = None
            robot.path = []
            robot.index = 0
            
    def pure_pursuit(self, current_x, current_y, current_heading, path, index, robot):
        global lookahead_distance 
        closest_point = None
        v = speed
        lookahead = lookahead_distance

        if robot.obstacle_detected:
            v *= 0.5
            lookahead *= 0.5
        for i in range(index, len(path)):
            x = path[i][0]
            y = path[i][1]
            distance = math.hypot(current_x - x, current_y - y)
            if lookahead <= distance:
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
        return v, desired_steering_angle, closest_point

        
    def distance_to_goal(self, robot):
        if not robot.path:
            return float('inf')
        goal_world = self.grid_to_world(robot.goal[0], robot.goal[1])
        distance = math.hypot(robot.x - goal_world[0], robot.y - goal_world[1])
        return distance
    
    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_grid = int((y_world - self.map_origin[1])/ self.map_resolution)
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid):
        x_world = x_grid * self.map_resolution + self.map_origin[0]
        y_world = y_grid * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)

    def visualize_path(self, robot):
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = f"{robot.namespace}_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05 
        path_marker.color.a = 1.0
        path_marker.color.g = 1.0  
        path_marker.color.r = 0.0
        path_marker.color.b = 0.0
        for (x, y) in robot.path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            path_marker.points.append(p)
        robot.path_marker_pub.publish(path_marker)

    def visualize_lookahead(self, robot, closest_point):
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.ns = f"{robot.namespace}_lookahead"
        lookahead_marker.id = 0
        lookahead_marker.type = Marker.SPHERE
        lookahead_marker.action = Marker.ADD
        lookahead_marker.pose.position.x = closest_point[0]
        lookahead_marker.pose.position.y = closest_point[1]
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.scale.x = 0.1
        lookahead_marker.scale.y = 0.1
        lookahead_marker.scale.z = 0.1
        lookahead_marker.color.a = 1.0
        lookahead_marker.color.r = 1.0  
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        robot.lookahead_marker_pub.publish(lookahead_marker)
    
    def visualize_occupations(self, robot):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot.namespace}_occupations"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = self.map_resolution * 0.8  
        marker.scale.y = self.map_resolution * 0.8
        marker.color.a = 0.8 
        max_occ = np.max(robot.occupations) if np.max(robot.occupations) > 0 else 1
        for y in range(self.height):
            for x in range(self.width):
                value = robot.occupations[y][x]
                if value > 0:
                    wx, wy = self.grid_to_world(x, y)
                    p = Point()
                    p.x = wx
                    p.y = wy
                    p.z = 0.0
                    marker.points.append(p)
                    color = ColorRGBA()
                    intensity = value / max_occ
                    color.r = float(intensity)
                    color.g = 0.0
                    color.b = 0.0
                    color.a = 0.8 * intensity
                    marker.colors.append(color)
        robot.occupancy_marker_pub.publish(marker)
        
def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(namespace0='tb0', namespace1='tb1')
    rclpy.spin(nav)
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()