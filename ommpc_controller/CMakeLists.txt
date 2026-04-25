cmake_minimum_required(VERSION 3.0.2)
project(ommpc_controller)
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  dynamic_reconfigure
  traj_utils
)

find_package(Eigen3 REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/state.cfg
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES ommpc_controller
#  CATKIN_DEPENDS geometry_msgs nav_msgs quadrotor_msgs roscpp rospy std_msgs tf airsim_ros_pkgs
  CATKIN_DEPENDS dynamic_reconfigure geometry_msgs mavros nav_msgs roscpp rospy sensor_msgs std_msgs traj_utils
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}_example_node 
  src/ommpc_example.cpp
)

add_dependencies(${PROJECT_NAME}_example_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}_example_node
  ${catkin_LIBRARIES}
  osqp
)
