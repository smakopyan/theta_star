import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

def generate_launch_description():
    ld = LaunchDescription()

    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    TURTLEBOT3_MODEL = 'waffle'
    urdf_file_name = 'turtlebot3_' + TURTLEBOT3_MODEL + '.urdf'

    urdf = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'urdf',
            'turtlebot3_' + TURTLEBOT3_MODEL + '.urdf')

    turtlebot3_gazebo_path = get_package_share_directory("turtlebot3_gazebo")

    model = os.path.join(turtlebot3_gazebo_path, "models", "turtlebot3_" + TURTLEBOT3_MODEL ,"model.sdf")
    
    declare_world_name = DeclareLaunchArgument(
        'world',
        default_value='test1.sdf',  
        choices=['world1.sdf', 'world2.sdf', 'world3.sdf', 'world4.sdf', 'turtlebot3_world.world', 'test.sdf', 'test1.sdf']
    )
    
    declare_map_name = DeclareLaunchArgument(
        'map',
        default_value='test.yaml',
        choices=['map1.yaml', 'map2.yaml', 'map3.yaml', 'map4.yaml', 'turtlebot3_map.yaml', 'map_test.yaml', 'test.yaml']
    )
        
    world_path = PathJoinSubstitution([
        get_package_share_directory('theta_star'),  
        'worlds',
        LaunchConfiguration('world')  
    ])
    
    map_path = PathJoinSubstitution([
        get_package_share_directory('theta_star'),
        'maps',
        LaunchConfiguration('map')
    ])

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world_path}.items(),
        )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzclient.launch.py")
        ),
    )
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_path,
                    'map_frame': '/map',
                    'topic_name': "/map",
                    'use_sim_time': True 
                     },],
    )
    
    map_server_lifecyle=Node(package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            output='screen',
                parameters=[
                {'use_sim_time': True},
                {'autostart': True},
                {'node_names': ['map_server']},
                # {'bond_timeout': 0.5}
            ])
    ld.add_action(declare_world_name)
    ld.add_action(declare_map_name)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(map_server)
    ld.add_action(map_server_lifecyle)

# test.yaml
    spawn_points = [(-0.7, 0.05), (2.0, 0.5)]
    last_action = None
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    for i in range(2):
        x, y = spawn_points[i]
        name = "tb" + str(i)
        namespace = "/tb" + str(i)

        turtlebot_state_publisher = Node(
            package="robot_state_publisher",
            namespace=namespace,
            executable="robot_state_publisher",
            output="screen",
            parameters=[{
                'frame_prefix': namespace + '/',
                "use_sim_time": True,
                "publish_frequency": 10.0}],
            # remappings=remappings,
            arguments=[urdf],
        )

        spawn_turtlebot3 = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-file",
                model,
                "-entity",
                name,
                "-robot_namespace",
                namespace,
                "-x",
                str(x),
                "-y",
                str(y),
                "-z",
                "0.01",
            ],
            output="screen",
        )
        if not last_action:
            ld.add_action(turtlebot_state_publisher)
            ld.add_action(spawn_turtlebot3)
        else:
            spawn_turtlebot3_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    on_exit=[spawn_turtlebot3, turtlebot_state_publisher],
                )
            )
            ld.add_action(spawn_turtlebot3_event)
        last_action = spawn_turtlebot3
    return ld
