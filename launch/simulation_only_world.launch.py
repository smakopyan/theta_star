import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    TURTLEBOT3_MODEL = 'waffle'
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    turtlebot3_gazebo_path = get_package_share_directory("theta_star")

    model = os.path.join(turtlebot3_gazebo_path, "models", "model.sdf")
    urdf = os.path.join(
            get_package_share_directory('theta_star'),
            'urdf',
            'turtlebot3_' + TURTLEBOT3_MODEL + '.urdf')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    world = os.path.join(
        get_package_share_directory('theta_star'),
        'worlds',
        'maze.sdf'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "publish_frequency": 10.0}],
            arguments=[urdf],
        )
    

    spawn_turtlebot_cmd =        spawn_turtlebot3 = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-file",
                model,
                "-entity",
                'tb',
                "-x",
                x_pose,
                "-y",
                y_pose,
                "-z",
                "0.01",
            ],
            output="screen",
        )
    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld