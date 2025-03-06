import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue
from rclpy.qos import QoSProfile
# import scipy.interpolate as si
import math
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist


lookahead_distance = 0.2  # Дистанция для поиска следующей точки
speed = 0.2  # Скорость робота
expansion_size = 5  # Размер расширения стен на карте

class node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost from this node to end
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f  # Compare based on the total cost f

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
            return path[::-1]  # Return reversed path

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for new_position in neighbors:
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])

            if (0 <= node_position[0] < grid.shape[0]) and (0 <= node_position[1] < grid.shape[1]):
                if grid[node_position[0]][node_position[1]] != 1:  # Check if it's not an obstacle
                    neighbor_node = node(current_node, node_position)

                    if neighbor_node in closed_list:
                        continue

                    # Calculate costs
                    neighbor_node.g = current_node.g + heuristic(current_node.position, neighbor_node.position)
                    neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    if add_to_open(open_list, neighbor_node):
                        open_list.put((neighbor_node.f, neighbor_node))

    return None  # Return None if no path is found

def add_to_open(open_list, neighbor):
    for item in open_list.queue:
        if neighbor == item[1] and neighbor.g >= item[1].g:
            return False
    return True


# def bspline_planning(array, sn):
#     try:
#         array = np.array(array)
#         x = array[:, 0]
#         y = array[:, 1]
#         N = 2
#         t = range(len(x))
#         x_tup = si.splrep(t, x, k=N)
#         y_tup = si.splrep(t, y, k=N)

#         x_list = list(x_tup)
#         xl = x.tolist()
#         x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

#         y_list = list(y_tup)
#         yl = y.tolist()
#         y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

#         ipl_t = np.linspace(0.0, len(x) - 1, sn)
#         rx = si.splev(ipl_t, x_list)
#         ry = si.splev(ipl_t, y_list)
#         path = [(rx[i], ry[i]) for i in range(len(rx))]
#     except:
#         path = array
#     return path

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
    def __init__(self):
        super().__init__('Navigation')
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_initialized = False
        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.subscription_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        # self.subscription_goal = self.create_subscription(
        #     PoseStamped, '/goal_pose', self.goal_callback, QoSProfile(depth=10)
        # )

        self.publisher_cmd_vel = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        self.flag = 0
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        print("Wait for target for ...")

    # def goal_callback(self, msg):
    #     self.goal = (msg.pose.position.x, msg.pose.position.y)
    #     print(f'Target: x:{msg.pose.position.x}, y: {msg.pose.position.y}')
    #     self.flag += 1

    def map_callback(self, msg):
        if self.flag == 0:
            self.map_resolution = msg.info.resolution
            self.map_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ]
            print(f'Map resolution: {self.map_resolution}, Origin: {self.map_origin}')
            self.grid = costmap(msg.data, msg.info.width, msg.info.height, self.map_resolution)
            self.grid[self.grid == -1] = 1
            self.grid[self.grid >= self.map_resolution * 100] = 1
            self.width = msg.info.width
            self.height = msg.info.height
            self.flag += 1
    
    def timer_callback(self):
        if not self.odom_initialized:
            return 
        if self.flag == 1:
            while True:
                goal_x = np.random.randint(0, self.width-1)
                goal_y = np.random.randint(0, self.height-1)
                if  self.grid[goal_x][goal_y] == 0:
                    self.goal_x = goal_x
                    self.goal_y = goal_y
                    self.flag += 1
                    print(self.goal_x, self.goal_y)
                    break
        if self.flag == 2:
            start = self.world_to_grid(self.x, self.y)
            # goal = self.world_to_grid(self.goal[0], self.goal[1])

            grid = self.grid
            grid[start[0]][start[1]] = 0

            path = theta_star((start[0], start[1]), (self.goal_x, self.goal_y), grid)
            if path is None:
                print("No path found.")
                return
            
            self.path = [self.grid_to_world(i[0], i[1]) for i in path]
            pos = self.world_to_grid(self.x, self.y)
            print(f'Position: x: {pos[0]}, y: {pos[1]}')
            self.i = 0

            twist = Twist()
            twist.linear.x, twist.angular.z, self.i = pure_pursuit(self.x, self.y, self.yaw, self.path, self.i)

            if abs(self.x - self.path[-1][0]) < 0.05 and abs(self.y - self.path[-1][1]) < 0.05:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                print("Goal reached.\n")
                self.flag -= 1
                print("Waiting for a new goal...")

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
        self.odom_initialized = True

    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution)
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution)
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
