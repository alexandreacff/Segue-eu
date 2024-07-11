import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge

class DepthFilterNode(Node):

    def __init__(self):
        super().__init__('depth_filter_node')

        self.pub_pose = self.create_publisher(Image, 'tmp', 10)

        self.subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_filter_cb,
            10)
        
        self.publisher_ = self.create_publisher(Image, '/camera/depth/filtered_image', 10)
        self.bridge = CvBridge()

        # Configurando os filtros
        self.disparity_transform = rs.disparity_transform()
        self.inverse_disparity_transform = rs.disparity_transform(False)
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 1)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 3)
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def depth_filter_cb(self, msg):

        try:
            # Convertendo a mensagem ROS para uma imagem OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
            # depth_frame = rs.frame_from_image(depth_image)
            depth_frame = self.image_to_depth_frame(depth_image, 1280, 720)

            # Aplicando os filtros
            decimated_depth_frame = self.decimation.process(depth_frame)
            depth_frame = self.disparity_transform.process(decimated_depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.inverse_disparity_transform.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)

            # Convertendo de volta para um array numpy e para a mensagem ROS
            filtered_depth_image = np.asanyarray(depth_frame.get_data())
            self.get_logger().info(f'Tamanho imagem filtrada: {len(filtered_depth_image)}')

            filtered_msg = self.bridge.cv2_to_imgmsg(filtered_depth_image)
            self.publisher_.publish(filtered_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

