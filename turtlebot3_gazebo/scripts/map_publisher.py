#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import cv2
import yaml
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose

class MapPublisher(Node):
    def __init__(self, map_folder_path: str):
        super().__init__('map_publisher')
        self.get_logger().info('Map Publisher Node has been started.')
        self.map_folder_path = map_folder_path
        self.load_map_data()

        self.map_publisher = self.create_publisher(OccupancyGrid, 'virya_test/map', 10)
        self.timer = self.create_timer(1.0, self.publish_map)

    def load_map_data(self):
        map_file_path = os.path.join(self.map_folder_path, 'map.pgm')
        map_meta_data_path = os.path.join(self.map_folder_path, 'map.yaml')

        # Load the map image
        self.map_image = cv2.imread(map_file_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            self.get_logger().error(f"Failed to load map image from {map_file_path}")
            return

        # Read and parse the map metadata
        with open(map_meta_data_path, 'r') as f:
            map_meta_data = yaml.safe_load(f)

        self.resolution = map_meta_data.get('resolution')
        self.origin = map_meta_data.get('origin', [0.0, 0.0, 0.0])
        self.width = self.map_image.shape[1]
        self.height = self.map_image.shape[0]

    def publish_map(self):
        # Convert grayscale image to occupancy grid format
        linear_map = []
        for row in range(self.height):
            for col in range(self.width):
                pixel = self.map_image[row][col]
                if pixel == 0:
                    linear_map.append(100)  # obstacle
                elif pixel == 255:
                    linear_map.append(0)    # free
                else:
                    linear_map.append(-1)   # unknown

        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"

        map_msg.info.map_load_time = self.get_clock().now().to_msg()
        map_msg.info.width = self.width
        map_msg.info.height = self.height
        map_msg.info.resolution = self.resolution

        origin = Pose()
        origin.position.x = self.origin[0]
        origin.position.y = self.origin[1]
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        map_msg.info.origin = origin

        map_msg.data = linear_map
        self.map_publisher.publish(map_msg)
        
def main(args=None):
    rclpy.init(args=args)
    script_dir = os.path.dirname(os.path.realpath(__file__)) # same dir as script
    node = MapPublisher(script_dir)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
