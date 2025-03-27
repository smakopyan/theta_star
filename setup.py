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
        ('share/' + package_name + '/launch', ['launch/world.launch.py', 'launch/world_slam_.launch.py', 'launch/only_world.launch.py']),
        ('share/' + package_name + '/worlds', ['worlds/world1.sdf','worlds/world2.sdf', 'worlds/world3.sdf',
                                               'worlds/test1.sdf', 'worlds/turtlebot3_world.world', 'worlds/test.sdf']),
        ('share/' + package_name + '/maps', ['maps/map1.yaml','maps/test.yaml','maps/test.pgm', 'maps/test1.yaml','maps/test1.pgm', 'maps/map1.pgm', 'maps/map2.yaml', 'maps/map2.pgm',
                                             'maps/map3.yaml', 'maps/map3.pgm',
                                             "maps/turtlebot3_map.yaml", "maps/turtlebot3_map.pgm"]),
        ('share/' + package_name + '/rviz', ['rviz/view.rviz']),
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
            'navigate = theta_star.example:main'
        ],
    },
)

