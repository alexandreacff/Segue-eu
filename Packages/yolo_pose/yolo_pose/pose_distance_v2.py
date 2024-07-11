import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2 as rs2

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
import numpy as np

from yolo_pose.utils import letterbox

class DistanceNode(Node):

    def __init__(self) -> None:
        super().__init__("pose_distance_node")

        self.cv_bridge = CvBridge()


        self.sub =  self.create_subscription(
            # Image, "/camera/aligned_depth_to_color/image_raw", self.image_depth_cb,
            Image, "/camera/depth/image_rect_raw", self.image_depth_cb,
            qos_profile_sensor_data
        )

        self._sub = self.create_subscription(
            Point, "/keypoint_center", self.kpts_cb,
            10
        )

        # self.sub_conf = self.create_subscription("/camera/confidence/image_rect_raw", Image, self.confidenceCallback)

        self.intrinsics = None
        self.pix = None
        self.pix_grade = None

        self.image = None

    def image_depth_cb(self, msg: Image) -> None:
        
        try:
            self.image = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
            # # pick one pixel among all the pixels with the closest range:
            
            rows, cols = self.image .shape
            pix = (rows // 2, cols // 2)
            # self.pix = pix


            if pix:
                distance = 0.001 * self.image[pix[1], pix[0]]
                self.get_logger().info(f' Pixel: %8.2f %8.2f.' % (pix[0], pix[1]))
                
                self.get_logger().info(f' Distance: %8.2f ' % (distance))


        except CvBridgeError as e:
            print(e)
            return
        
        except ValueError as e:
            return

    def kpts_cb(self, msg: Point) -> None:

        if self.image is not None:
            x = int(msg.x)
            y = int(msg.y)
            self.pix = [x, y]

    # def confidenceCallback(self, data):
    #     try:
    #         self.image = self.cv_bridge.imgmsg_to_cv2(data, data.encoding)
    #         grades = np.bitwise_and(self.image >> 4, 0x0f)
    #         if (self.pix):
    #             self.pix_grade = grades[self.pix[1], self.pix[0]]
    #     except CvBridgeError as e:
    #         print(e)
    #         return


def main():
    rclpy.init()
    node = DistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()