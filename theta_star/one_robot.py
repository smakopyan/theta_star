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
expansion_size = 6

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

                    neighbor_node.g = current_node.g + heuristic(current_node.position, neighbor_node.position)
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

def dynamic_lookahead(current_speed, min_dist=0.3, max_dist=0.8):
    return min(max(current_speed * 2.0, min_dist), max_dist)


def pure_pursuit(current_x, current_y, current_heading, path, index):
    closest_point = None
    v = speed
    global lookahead_distance
    # lookahead_distance = dynamic_lookahead(v)

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
    distance_to_goal = math.hypot(current_x - path[-1][0], current_y - path[-1][1])
    
    if distance_to_goal < 0.5:
        max_steering = math.radians(30)  
        desired_steering_angle = np.clip(desired_steering_angle, -max_steering, max_steering)
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
    def __init__(self):
        super().__init__('Navigation')
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_initialized = False
        self.map_initialized = False
        self.saved = False

        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.subscription_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.publisher_cmd_vel = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        timer_period = 0.01
        self.map_init_time = 0.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        print("Wait for target for ...")

        #test1.sdf
        self.goals = [(3.99715, -1.6586), (3.50709, 1.44957), (1.25942, 1.25394), (-0.689823, 2.26387),
                      (-2.50234, 2.11622), (-1.7666, 0.285539), (-4.07267, 2.43495)]        
        
        
        self.goal = None
        self.ind = 0
        self.count = 0

        
    def map_callback(self, msg):
        if not self.map_initialized:
            self.map_resolution = msg.info.resolution
            self.map_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ]
            self.get_logger().info(f'Map resolution: {self.map_resolution}, Origin: {self.map_origin}')
            self.width = msg.info.width
            self.height = msg.info.height
            self.grid = costmap(msg.data, self.width, self.height, self.map_resolution)
            self.grid[self.grid == -1] = 1
            self.grid[self.grid >= self.map_resolution * 100] = 1
            self.map_initialized = True
            self.map_init_time = time.time()
    
    def timer_callback(self):
        if not self.map_initialized or time.time() - self.map_init_time < 90:
            return
        self.navigate()
    
    def generate_goal(self):
        if self.goal == None:
            ind = np.random.randint(0, len(self.goals[0]))
            goal = self.world_to_grid(self.goals[ind][0], self.goals[ind][1])
        return goal
    
    def generate_new_goal(self):
        while self.ind < len(self.goals):
            goal_world = self.goals[self.ind]
            goal_grid = self.world_to_grid(goal_world[0], goal_world[1])
            
            if self.grid[goal_grid[1]][goal_grid[0]] != 1:
                print('goal found')
                return goal_grid
                
            print(f"Skipping invalid goal {self.ind}")
            self.ind += 1
        
        print("All goals completed")
        return None
        
    def navigate(self):
        if not self.map_initialized:
            return
        start = self.world_to_grid(self.x, self.y)
        grid = self.grid.copy()
        debug_grid = self.grid.copy()

        grid[start[1]][start[0]] = 0
        
        if not self.saved:
            debug_grid[start[1]][start[0]] = 2
            
            for i in range(len(self.goals)):
                goal = self.world_to_grid(self.goals[i][0], self.goals[i][1])
                debug_grid[goal[1]][goal[0]] = 3 
            
            np.savetxt('/home/sa/turtlebot3_ws/src/theta_star/debug/grid.txt', debug_grid.astype(int), fmt='%d')
            print('grid saved')
            self.saved = True
            
        if self.goal==None:
            print('generating new goal.....')
            self.goal = self.generate_new_goal()
        
        if self.ind >= len(self.goals):
            twist = Twist()
            self.publisher_cmd_vel.publish(twist)
            return
    
        
        path = theta_star((start[1], start[0]), (self.goal[1], self.goal[0]), grid)

        if path is None:
            print(f"No path found for.")
            # self.goal=None
            return

        self.path = [self.grid_to_world(i[1], i[0]) for i in path]
        print(f'Position for x: {self.x}, y: {self.y}')
        self.i = 0

        twist = Twist()
        twist.linear.x, twist.angular.z, self.i = pure_pursuit(self.x, self.y, self.yaw, self.path, self.i)

        target_angle = math.atan2(self.path[-1][1] - self.y, self.path[-1][0] - self.x)
        angle_error = abs(self.yaw - target_angle)
        
        if (math.hypot(self.x - self.path[-1][0], self.y - self.path[-1][1]) < 0.1 and angle_error < 0.5):
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            print(f"Goal reached.\n")
            self.ind += 1
            
            self.goal = None
        self.publisher_cmd_vel.publish(twist)
            

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self.odom_initialized_time = time.time()
        self.odom_initialized = True

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

    navigation_tb2 = Navigation()
    rclpy.spin(navigation_tb2)
    navigation_tb2.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()