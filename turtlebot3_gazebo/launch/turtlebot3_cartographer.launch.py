#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joep Tool, Hyungyu Kim

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir

def generate_launch_description():
    # Directories
    tb3_prefix = get_package_share_directory('turtlebot3_gazebo')

    # Launch Configurations
    cartographer_config_dir = LaunchConfiguration('cartographer_config_dir')
    configuration_basename = LaunchConfiguration('configuration_basename')
    resolution = LaunchConfiguration('resolution')
    publish_period = LaunchConfiguration('publish_period')
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    # Launch arguments (Declare)
    declare_cartographer_config_dir_cmd = DeclareLaunchArgument(
        'cartographer_config_dir',
        default_value=os.path.join(tb3_prefix, 'config'),
        description='Full path to config directory')

    declare_configuration_basename_cmd = DeclareLaunchArgument(
        'configuration_basename',
        default_value='turtlebot3_lds_2d.lua',
        description='Cartographer config file name')

    declare_resolution_cmd = DeclareLaunchArgument(
        'resolution',
        default_value='0.05',
        description='Resolution of occupancy grid')

    declare_publish_period_cmd = DeclareLaunchArgument(
        'publish_period',
        default_value='1.0',
        description='Occupancy grid publishing period')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time')

    declare_x_pose_cmd = DeclareLaunchArgument(
        'x_pose',
        default_value='-2.0',
        description='Initial X position of TurtleBot3')

    declare_y_pose_cmd = DeclareLaunchArgument(
        'y_pose',
        default_value='-0.5',
        description='Initial Y position of TurtleBot3')


    run_cartographer_cmd = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '--configuration_directory', cartographer_config_dir,
            '--configuration_basename', configuration_basename
        ]
    )

    occupancy_launch_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/occupancy_grid.launch.py']),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'resolution': resolution,
            'publish_period_sec': publish_period
        }.items()
    )

    # Final LaunchDescription
    ld = LaunchDescription()

    # Add declared arguments
    ld.add_action(declare_cartographer_config_dir_cmd)
    ld.add_action(declare_configuration_basename_cmd)
    ld.add_action(declare_resolution_cmd)
    ld.add_action(declare_publish_period_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_x_pose_cmd)
    ld.add_action(declare_y_pose_cmd)

    ld.add_action(run_cartographer_cmd)
    ld.add_action(occupancy_launch_cmd)

    return ld
