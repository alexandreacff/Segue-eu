import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np
import os

import torch

from sensor_msgs.msg import Image
from perception_msgs.msg import BoundingBoxes
from perception_msgs.msg import BoundingBox
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory


from yolo_pose.utils import letterbox

class KeypointsNode(Node):

    def __init__(self) -> None:
        super().__init__("kpts_pose_node")

        self.cv_bridge = CvBridge()

        self.kps_colors = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
              [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
              [255, 128, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255],
              [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255]]
        
        self.limb_colors = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
               [255, 51, 255], [255, 51, 255], [255, 51, 255], [255, 128, 0],
               [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
               [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0], [0, 255, 0]]
        
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]

        # subs
        self._sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data
        )

        self._sub = self.create_subscription(
            BoundingBoxes, '/yolov8/bounding_boxes', self.kpts_cb,
            10
        )

        self.pub_keys = self.create_publisher(Point, 'keypoint_center', 10)

        self.pub_pose = self.create_publisher(Image, 'image_pose', 10)

        self.image = None

    def get_scaled_keypoints(self, old_shape, target_shape, resized_keypoints):
        # Dimensões da imagem original
        original_height, original_width = target_shape
        
        # Dimensões da imagem redimensionada
        new_height, new_width = old_shape
        
        # Escalar keypoints de volta para as dimensões da imagem original
        scaled_keypoints = []
        for kp in resized_keypoints:
            x, y = kp
            scaled_x = float(x * original_width / new_width)
            scaled_y = float(y * original_height / new_height)
            scaled_keypoints.append((scaled_x, scaled_y))
        
        return scaled_keypoints


    def draw_keypoints(self, image, results, id):
            
        minha_confianca = []
        meus_keypoints = []
            
        for j in range(17):

            minha_confianca.append(results[j][2])
            meus_keypoints.append(results[j][0:2])

        meus_keypoints = self.get_scaled_keypoints((640, 640), (720, 1280), meus_keypoints)

        centro = (np.array(meus_keypoints[5]) + np.array(meus_keypoints[6]) + np.array(meus_keypoints[11]) + np.array(meus_keypoints[12])) / 4 
        x, y = centro
        cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)

        self.get_logger().info(f'ID: {id}')
        point = Point()
        point.x = float(x)
        point.y = float(y)

        self.pub_keys.publish(point)

        for j in range(len(meus_keypoints)):

            if j == 5 or j == 6 or j == 11 or j == 12:
                x, y = meus_keypoints[j]
                cv2.circle(image, (int(x), int(y)), 10, (0, 255, 0), -1)

            elif minha_confianca[j] > 0.7:
                x, y = meus_keypoints[j]
                cv2.circle(image, (int(x), int(y)), 10, (255, 0, 0), -1)

        return image
    
    def image_cb(self, msg: Image) -> None:


        self.image = self.cv_bridge.imgmsg_to_cv2(msg)
        # self.image, self.ratio, dwdh = letterbox(image, (640, 640))
        # self.dw, self.dh= int(dwdh[0]), int(dwdh[1])


    def kpts_cb(self, msg) -> None:

        if self.image is not None and len(msg.bounding_boxes) > 0:
            for box in msg.bounding_boxes:
                kpts = np.array(box.kp_pose).reshape((len(box.kp_pose)//3, 3))
                self.get_logger().info(f'{kpts.shape}')
                self.draw_keypoints(self.image, kpts, box.id)
            self.get_logger().info(f'{self.image.shape}')
            self.pub_pose.publish(self.cv_bridge.cv2_to_imgmsg(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)))


def main():
    rclpy.init()
    node = KeypointsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()