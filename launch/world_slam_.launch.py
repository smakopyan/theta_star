import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    ld = LaunchDescription()

    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    TURTLEBOT3_MODEL = 'burger'
    urdf_file_name = 'turtlebot3_' + TURTLEBOT3_MODEL + '.urdf'

    urdfs = [os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'urdf',
        'turtlebot3_' + TURTLEBOT3_MODEL + '_0'+ '.urdf'),
        os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'urdf',
        'turtlebot3_' + TURTLEBOT3_MODEL + '_1'+ '.urdf')]

    turtlebot3_gazebo_path = get_package_share_directory("turtlebot3_gazebo")

    models = [
        os.path.join(turtlebot3_gazebo_path, "models", "turtlebot3_" + TURTLEBOT3_MODEL + '_0', "model.sdf"),
        os.path.join(turtlebot3_gazebo_path, "models", "turtlebot3_" + TURTLEBOT3_MODEL + '_1', "model.sdf"),
    ]

    world_file_name = 'turtlebot3_worlds/' + TURTLEBOT3_MODEL + '.model'
    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_world.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzclient.launch.py")
        ),
    )

    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    spawn_points = [(-1.5, 0.5)]

    x, y = spawn_points

    turtlebot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "publish_frequency": 10.0}],
        # remappings=remappings,
        arguments=[urdfs[i]],
    )

    spawn_turtlebot3 = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-file",
            models[i],
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            "0.01",
        ],
        parameters=[{'use_sim_time': True}],
        output="screen",
    )

    async_slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='async_slam_toolbox_node',
        parameters=[{
            'use_sim_time': True,
            'odom_frame': '/odom',
            'base_frame': '/base_footprint',
            'scan_topic': 'scan',
            'map_frame':  '/map',
            'minimum_travel_distance': 0.3,
            'minimum_travel_heading': 0.3,
            'resolution': 0.05,
        }],
        
        output='screen',
        
    )  
    ld.add_action(turtlebot_state_publisher)
    ld.add_action(spawn_turtlebot3)
    ld.add_action(async_slam_toolbox)

    return ld
