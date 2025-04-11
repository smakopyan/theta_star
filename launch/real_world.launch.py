import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )


    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': os.path.join(get_package_share_directory('theta_star'),
                                                   'maps', 'map.yaml'),
                    'map_frame': '/map',
                    'topic_name': "/map",
                    'use_sim_time': False 
                     },],
    )
    
    map_server_lifecyle=Node(package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            output='screen',
                parameters=[
                {'use_sim_time': False},
                {'autostart': True},
                {'node_names': ['map_server']},
                # {'bond_timeout': 0.5}
            ])
    

    rviz_config_dir = os.path.join(
        get_package_share_directory('theta_star'),
        'rviz',
        'slam.rviz')
    
    rviz_node = Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config_dir],
                    parameters=[{'use_sim_time': False}],
                    output='screen')
    
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[{
            'use_sim_time': False,
            'odom_model_type': 'diff-corrected',
            'laser_model_type': 'likelihood_field'
        }]
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )


    ld = LaunchDescription()
    ld.add_action(map_server)
    ld.add_action(map_server_lifecyle)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(amcl)
    ld.add_action(static_tf)
    ld.add_action(rviz_node)

    return ld
