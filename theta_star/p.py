import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
import math
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import cv2
import collections
from theta_star import ThetaStar
from turtlebot_env import Robot, slam_to_grid_map, path, grid_to_world
import matplotlib.pyplot as plt
from collections import deque
import logging
from multi_robot_navigator_example import euler_from_quaternion
import os
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class TurtleBotEnv(Node):
    def __init__(self):
        super().__init__('turtlebot_env')
        self.num_robots = 2
        spawn_points = [[-0.7, 0.05], [-2.5, 0.05]]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        goals = [[-2.5, 0.05],[-0.7, 0.05] ]
        self.robots = [Robot(f"tb{i}", spawn_points[i], goals[i]) for i in range(self.num_robots)]
        print(self.robots)
        slam_map = cv2.imread(os.path.join(get_package_share_directory('theta_star'),
                                           'maps','map3.pgm'), cv2.IMREAD_GRAYSCALE)
    
        self.grid_map = slam_to_grid_map(slam_map)

        self.map_initialized = False
        self.occupation_map = np.zeros_like(self.grid_map, dtype=np.float32)
        self.penalty_map = np.zeros_like(self.grid_map, dtype=np.float32)
        self.max_steps = 5000
        
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
            robot.optimal_path = path(robot, self.grid_map, self.occupation_map, self.penalty_map)
            self.timer = self.create_timer(0.01, self._timer_callback)

    def _timer_callback(self):
        for robot in self.robots:
            path_= self.get_path(robot.optimal_path)
            self.visualize_path(robot, path_)

    def get_path(self, opt_path):
        path = [grid_to_world(i[1], i[0]) for i in opt_path]
        return path
    
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
        path_marker.color.g = 1.0  
        path_marker.color.r = 0.0
        path_marker.color.b = 0.0
        for (x, y) in path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            path_marker.points.append(p)
        robot.path_marker_pub.publish(path_marker)
