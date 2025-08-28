  ################################################################################
  # Set minimum required version of cmake, project name and compile options
  ################################################################################
  cmake_minimum_required(VERSION 3.5)
  project(turtlebot3_gazebo)

  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()

  if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
  endif()

  ################################################################################
  # Find ament packages and libraries for ament and system dependencies
  ################################################################################
  find_package(ament_cmake REQUIRED)
  find_package(ros_gz_sim REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(tf2 REQUIRED)

  ################################################################################
  # Build
  ################################################################################
  # Add Gazebo includes and libs if set in environment
  link_directories(
    ${GAZEBO_LIBRARY_DIRS}
    /usr/lib/x86_64-linux-gnu     # Adjust this path if your ignition libs are elsewhere
  )

  include_directories(
    include
    ${GAZEBO_INCLUDE_DIRS}
    /usr/include/ignition/transport11  # Include Ignition Transport headers
    /usr/include/ignition/msg8  
  )

  set(dependencies
    "geometry_msgs"
    "nav_msgs"
    "rclcpp"
    "sensor_msgs"
    "tf2"
  )

  set(EXEC_NAME "turtlebot3_drive")
  set(DRIVER_NAME "pub_pose")

  add_executable(${EXEC_NAME} src/turtlebot3_drive.cpp)
  ament_target_dependencies(${EXEC_NAME} ${dependencies})


  ################################################################################
  # Install
  ################################################################################
  install(TARGETS ${EXEC_NAME}
    DESTINATION lib/${PROJECT_NAME}
  )

  install(TARGETS)

  install(DIRECTORY launch models params rviz urdf worlds config
    DESTINATION share/${PROJECT_NAME}/
  )

  # Install Python scripts
  install(PROGRAMS
    scripts/virya_test_node.py
    scripts/map_publisher.py
    scripts/map.pgm
    scripts/map.yaml
    DESTINATION lib/${PROJECT_NAME}
  )

  install(DIRECTORY include/
    DESTINATION include/
  )

  ################################################################################
  # Macro for ament package
  ################################################################################
  ament_export_include_directories(include)
  ament_export_dependencies(gazebo_ros_pkgs)
  ament_export_dependencies(geometry_msgs)
  ament_export_dependencies(nav_msgs)
  ament_export_dependencies(rclcpp)
  ament_export_dependencies(sensor_msgs)
  ament_export_dependencies(tf2)
  ament_package()
