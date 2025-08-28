#!/usr/bin/env python3

import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def euler_to_quaternion(yaw):
    # roll = pitch = 0
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


def launch_static_transform(context, *args, **kwargs):
    x = float(LaunchConfiguration('x_pose').perform(context))
    y = float(LaunchConfiguration('y_pose').perform(context))
    yaw = float(LaunchConfiguration('yaw_pose').perform(context))

    qx, qy, qz, qw = euler_to_quaternion(yaw)

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=[
            str(x), str(y), '0.0',
            str(qx), str(qy), str(qz), str(qw),
            'map', 'odom'
        ]
    )
    return [static_tf_node]


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='5.5')
    y_pose = LaunchConfiguration('y_pose', default='-6.0')
    yaw_pose = LaunchConfiguration('yaw_pose', default='1.576')

    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_maze.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'true'}.items()
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
            'yaw_pose': yaw_pose
        }.items()
    )

    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'models'
        )
    )

    run_maze_pub_cmd = Node(
        package='turtlebot3_gazebo',
        executable='virya_test_node.py',
        name='virya_test_node',
        output='screen'
    )

    run_map_pub_cmd = Node(
        package='turtlebot3_gazebo',
        executable='map_publisher.py',
        name='virya_test_map_node',
        output='screen'
    )

    ld = LaunchDescription()

    # Add all actions
    ld.add_action(set_env_vars_resources)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(OpaqueFunction(function=launch_static_transform))
    ld.add_action(run_maze_pub_cmd)
    ld.add_action(run_map_pub_cmd)

    return ld
