from setuptools import find_packages, setup

package_name = 'theta_star'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/real_world.launch.py', 'launch/simulation_basic.launch.py', 'launch/simulation_only_world.launch.py', 'launch/simulation_slam.launch.py']),
        ('share/' + package_name + '/worlds', ['worlds/world1.sdf','worlds/world2.sdf', 'worlds/world3.sdf',
                                               'worlds/test1.sdf', 'worlds/turtlebot3_world.world', 'worlds/test.sdf']),
        ('share/' + package_name + '/maps', ['maps/map1.yaml','maps/test.yaml','maps/test.pgm', 'maps/test1.yaml','maps/test1.pgm', 'maps/map1.pgm', 'maps/map2.yaml', 'maps/map2.pgm',
                                             'maps/map3.yaml', 'maps/map3.pgm',
                                             'maps/map.yaml', 'maps/map.pgm',
                                             "maps/turtlebot3_map.yaml", "maps/turtlebot3_map.pgm"]),
        ('share/' + package_name + '/rviz', ['rviz/multi_robot_view.rviz', 'rviz/single_robot_view.rviz']),
        ('share/' + package_name + '/urdf', ['urdf/turtlebot3_waffle.urdf','urdf/turtlebot3_burger.urdf']),
        ('share/' + package_name + '/models', ['models/model.sdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sa',
    maintainer_email='satenikak@yandex.ru',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            'navigate = theta_star.multi_robot_navigator_example:main',
            'one_robot = theta_star.single_robot_navigator:main'
        ],
    },
)

