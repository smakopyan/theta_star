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

def theta_star(start, end, grid):
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

class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')
        self.map_initialized = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.goal = None
        self.path = []
        self.laser_data = None
        self.obstacle_detected = False

        # Subscribers
        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.subscription_odom = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.subscription_goal = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, QoSProfile(depth=10)
        )
        self.subscription_laser = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )


        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_marker_pub = self.create_publisher(Marker, '/path_marker', 10)
        self.lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.navigation_loop)

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
            self.get_logger().info("Map initialized")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"New goal received: {self.goal}")

    def laser_callback(self, msg):
        self.laser_data = msg.ranges
        self.obstacle_detected = any(d < 0.3 for d in msg.ranges if d > 0.0)

    def navigation_loop(self):
        if not self.map_initialized or self.goal is None:
            return

        start = self.world_to_grid(self.x, self.y)
        goal = self.world_to_grid(self.goal[0], self.goal[1])
        
        grid = self.grid.copy()
        grid[start[1]][start[0]] = 0
        
        path = theta_star((start[1], start[0]), (goal[1], goal[0]), grid)
        
        if path is None:
            self.get_logger().warn("Path not found!")
            return
            
        self.path = [self.grid_to_world(i[1], i[0]) for i in path]
        self.visualize_path()

        v, angle, closest_point = self.pure_pursuit()
        
        if closest_point is not None:
            self.visualize_lookahead(closest_point)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = angle
        self.cmd_vel_pub.publish(twist)

        if self.laser_data:
            v = angle = None
            for i in range(30, 60):
                if self.laser_data[i] < 0.3:
                        v = 0.08
                        angle = -math.pi/4 
                        break
                if v == None:
                    for i in range(300,330):
                        if self.laser_data[i] < 0.3:
                            v = 0.08
                            angle = math.pi/4
                            break
                if v and angle:
                    twist.linear.x = v
                    twist.angular.z = angle
                    self.cmd_vel_pub.publish(twist)

        if(abs(self.x - self.path[-1][0]) < 0.15 and abs(self.y - self.path[-1][1])< 0.15):
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

            self.get_logger().info("The goal have been reached!")
            self.goal = None
            self.path = []
            return

    def pure_pursuit(self):
        closest_point = None
        v = speed
        lookahead = lookahead_distance

        if self.obstacle_detected:
            v *= 0.5
            lookahead *= 0.5

        for i in range(len(self.path)):
            x, y = self.path[i]
            distance = math.hypot(self.x - x, self.y - y)
            if distance > lookahead:
                closest_point = (x, y)
                break

        if closest_point is None:
            closest_point = self.path[-1]

        target_heading = math.atan2(closest_point[1] - self.y, closest_point[0] - self.x)
        desired_steering_angle = target_heading - self.yaw

        if desired_steering_angle > math.pi:
            desired_steering_angle -= 2 * math.pi
        elif desired_steering_angle < -math.pi:
            desired_steering_angle += 2 * math.pi

        if abs(desired_steering_angle) > math.pi/4:
            v = 0.0

        return v, desired_steering_angle, closest_point

    def distance_to_goal(self):
        if not self.path:
            return float('inf')
        return math.hypot(self.x - self.goal[0], self.y - self.goal[1])

    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution)
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid):
        x_world = x_grid * self.map_resolution + self.map_origin[0]
        y_world = y_grid * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)

    def visualize_path(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.color.r = 0.0
        marker.color.b = 0.0

        for (x, y) in self.path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)

    def visualize_lookahead(self, closest_point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = closest_point[0]
        marker.pose.position.y = closest_point[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.lookahead_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    navigation = Navigation()
    rclpy.spin(navigation)
    navigation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()