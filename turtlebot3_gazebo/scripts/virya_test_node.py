#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import math

class PoseAndGoalPublisher(Node):
    def __init__(self):
        super().__init__('pose_and_goal_publisher')
        self.get_logger().info('Pose and Goal Publisher Node started.')

        self.odom_pub = self.create_publisher(Odometry, 'virya_test/odom', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'virya_test/exit', 10)
        self.entry_pub = self.create_publisher(PoseStamped, 'virya_test/entry', 10)

        self.pose_sub = self.create_subscription(
            PoseArray,
            '/absolute_pose',
            self.pose_array_callback,
            10
        )

        self.last_position = (5.5000649787791165, -6.0124869804028584)
        self.last_time = None
        self.last_yaw = None

        self.start_pose = self.last_position
        self.current_closest_pose = None

        self.alpha = 0.2  # Smoothing factor for exponential moving average

        self.goal_timer = self.create_timer(1.0, self.publish_goals)
        self.odom_timer = self.create_timer(0.05, self.publish_odom)

    def get_yaw(self, orientation):
        q = orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def pose_array_callback(self, msg: PoseArray):
        ref_x = self.last_position[0]
        ref_y = self.last_position[1]

        min_dist = float('inf')
        best_pose = None

        for pose in msg.poses:
            if abs(pose.orientation.w - 1.0) < 1e-9:
                continue
            px, py = pose.position.x, pose.position.y
            dist = math.hypot(px - ref_x, py - ref_y)
            if dist < min_dist:
                min_dist = dist
                best_pose = pose

        if best_pose is None:
            return

        # Filtering: position jump
        pos = best_pose.position
        jump_dist = math.hypot(pos.x - ref_x, pos.y - ref_y)
        if jump_dist > 3.0:
            self.get_logger().warn(f"Rejected pose due to large jump: {jump_dist:.2f} m")
            return

        # Filtering: orientation jump
        yaw = self.get_yaw(best_pose.orientation)

        now = self.get_clock().now().to_msg()

        if self.current_closest_pose is None:
            stamped_pose = PoseStamped()
            stamped_pose.header.stamp = now
            stamped_pose.header.frame_id = 'map'
            stamped_pose.pose = best_pose
            self.current_closest_pose = stamped_pose
        else:
            cur = self.current_closest_pose.pose.position
            new = best_pose.position

            cur.x = self.alpha * new.x + (1 - self.alpha) * cur.x
            cur.y = self.alpha * new.y + (1 - self.alpha) * cur.y
            cur.z = self.alpha * new.z + (1 - self.alpha) * cur.z

            current_yaw = self.get_yaw(self.current_closest_pose.pose.orientation)
            smoothed_yaw = self.alpha * yaw + (1 - self.alpha) * current_yaw
            q = quaternion_from_euler(0, 0, smoothed_yaw)

            self.current_closest_pose.pose.orientation.x = q[0]
            self.current_closest_pose.pose.orientation.y = q[1]
            self.current_closest_pose.pose.orientation.z = q[2]
            self.current_closest_pose.pose.orientation.w = q[3]
            self.current_closest_pose.header.stamp = now
            self.current_closest_pose.header.frame_id = 'map'

        self.last_yaw = yaw

    def publish_odom(self):
        if self.current_closest_pose is None:
            return

        current_time = self.get_clock().now()
        pos = self.current_closest_pose.pose.position

        vx = vy = 0.0
        if self.last_position is not None and self.last_time is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                vx = (pos.x - self.last_position[0]) / dt
                vy = (pos.y - self.last_position[1]) / dt

        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose = self.current_closest_pose.pose
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.linear.z = 0.0

        self.odom_pub.publish(odom_msg)

        self.last_position = (pos.x, pos.y)
        self.last_time = current_time

    def publish_goals(self):
        now = self.get_clock().now().to_msg()
        q = quaternion_from_euler(0, 0, 0)

        # Exit goal
        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = 'map'
        goal.pose.position.x = -5.50
        goal.pose.position.y = 6.0
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]
        self.goal_pub.publish(goal)

        # Entry goal
        entry = PoseStamped()
        entry.header.stamp = now
        entry.header.frame_id = 'map'
        entry.pose.position.x = 5.50
        entry.pose.position.y = -6.0
        entry.pose.position.z = 0.0
        entry.pose.orientation.x = q[0]
        entry.pose.orientation.y = q[1]
        entry.pose.orientation.z = q[2]
        entry.pose.orientation.w = q[3]
        self.entry_pub.publish(entry)

def main(args=None):
    rclpy.init(args=args)
    node = PoseAndGoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
