import rclpy
from rclpy.node import Node
import numpy as np
import heapq
from nav_msgs.msg import OccupancyGrid , Odometry , Path
from geometry_msgs.msg import PoseStamped , Twist
import math
from rclpy.qos import QoSProfile
import threading ,time
from queue import PriorityQueue
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import LaserScan
import cv2


lookahead_distance = 0.15
speed = 0.1
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

def euler_from_quaternion(x,y,z,w):
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

def pure_pursuit(current_x, current_y, current_heading, path,index):
    global lookahead_distance
    closest_point = None
    v = speed
    for i in range(index,len(path)):
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
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v,desired_steering_angle,index, closest_point

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width)
    wall = np.where(data == 100)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data



class Navigation(Node):
    def __init__(self):
        super().__init__('Navigation')
        self.subscription = self.create_subscription(OccupancyGrid,'map',self.map_callback,10)
        self.subscription = self.create_subscription(Odometry,'odom',self.odom_callback,10)
        self.subscription = self.create_subscription(PoseStamped,'goal_pose',self.goal_pose_callback,QoSProfile(depth=10))
        self.subscription = self.create_subscription(LaserScan,'scan',self.laser_callback, QoSProfile(depth=10))

        self.path_marker_pub = self.create_publisher(Marker, '/path_marker', 10)
        self.lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("Wait for targets")

    def goal_pose_callback(self,msg):
        self.goal = (msg.pose.position.x,msg.pose.position.y)
        self.get_logger().info(f"New goal: {self.goal[0]}, {self.goal[1]}")
        threading.Thread(target=self.follow_path).start()

    def follow_path(self):
        start = self.world_to_grid(self.x, self.y) 
        goal = self.world_to_grid(self.goal[0], self.goal[1]) 
        
        data = costmap(self.map_data,self.width,self.height,self.map_resolution) 
        data[start[1]][start[0]] = 0
        data[data < 0] = 1 
        data[data > 5] = 1 
        path = theta_star((start[1],start[0]),(goal[1], goal[0]), data) 
        
        if path is None:
            self.get_logger().warn("Path not found!")
            return
        
        self.path = [self.grid_to_world(i[1], i[0]) for i in path] 
        self.visualize_path(self.path)
        self.get_logger().info(f"Robot coords: {self.x},{self.y}" )
        self.get_logger().info("Moving towards goal...")
        twist = Twist()
        v = 0.0
        w = 0.0
        i = 0
        while True:
            if not hasattr(self, 'x'):
                continue
            v , w ,i, cp = pure_pursuit(self.x,self.y,self.yaw,self.path,i)
            if cp:
                self.visualize_lookahead(cp)
            
            if self.check_obstacles():
                twist = Twist()  
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.publisher.publish(twist)
                self.get_logger().warn("EMERGENCY STOP!")
                time.sleep(1)  
                continue  
            if(abs(self.x - self.path[-1][0]) < 0.15 and abs(self.y - self.path[-1][1])< 0.15):
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info("The goal have been reached!")
                self.publisher.publish(twist)
                break
            twist.linear.x = v
            twist.angular.z = w
            self.publisher.publish(twist)
            time.sleep(0.1)
    def check_obstacles(self):
        for i in range(90, 270):
            if self.laser_data[i] < 0.2:  
                self.get_logger().warn(f"Obstacle detected at {self.laser_data[i]}m!")
                return True
        return False

    def visualize_path(self, path):
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = f"_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05 
        path_marker.color.a = 1.0
        path_marker.color.g = 1.0  
        path_marker.color.r = 0.0
        path_marker.color.b = 0.0
        for (x, y) in self.path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            path_marker.points.append(p)
        self.path_marker_pub.publish(path_marker)

    def visualize_lookahead(self, closest_point):
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.ns = "lookahead"
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
        self.lookahead_marker_pub.publish(lookahead_marker)
    
    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_grid = int((y_world - self.map_origin[1])/ self.map_resolution)
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid):
        x_world = x_grid * self.map_resolution + self.map_origin[0]
        y_world = y_grid * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)

    def map_callback(self,msg):
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, 
                           msg.info.origin.position.y]
        self.width = msg.info.width
        self.height = msg.info.height
        self.map_data = msg.data

    def laser_callback(self, msg):
        self.laser_data = msg.ranges
    
    def odom_callback(self,msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)


def main(args=None):
    rclpy.init(args=args)
    navigation_control = Navigation()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()