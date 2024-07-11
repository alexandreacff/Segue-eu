import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import time
import torch
import sys

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header

from ament_index_python.packages import get_package_share_directory


from yolo_pose.engine import TRTModule 
from yolo_pose.utils import blob, pose_postprocess, letterbox
from perception_msgs.msg import BoundingBoxes
from perception_msgs.msg import BoundingBox

class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_node")

        # params
        self.declare_parameter("engine", os.path.join(get_package_share_directory('yolo_pose'), 'config', 'yolov8n-pose.engine'))
        engine = self.get_parameter(
            "engine").get_parameter_value().string_value

        self.declare_parameter("device", "cuda")
        self.device = self.get_parameter(
            "device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self.cv_bridge = CvBridge()

        device = torch.device(self.device)
        self.yolo = TRTModule(engine, device)
        self.H, self.W = self.yolo.inp_info[0].shape[-2:]
        self.get_logger().info(f'Modelo pronto para inferêcias')
        self.get_logger().info(f'H: {self.H}, W: {self.W}')


        # subs
        self._sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data
        )

        # self.pub_keys = self.create_publisher(Float32MultiArray, 'keypoints_pose', 10)

        self.pub_boxes = self.create_publisher(BoundingBoxes, '/yolov8/bounding_boxes', 10)

    def yolox2bboxes_msgs(self, bboxes, kpts, scores, img_header: Header, image: np.ndarray) -> BoundingBoxes:
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        i = 0
        for i in range(len(scores)):
            one_box = BoundingBox()
            bbox = bboxes[i]
            # if < 0
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] < 0:
                bbox[2] = 0
            if bbox[3] < 0:
                bbox[3] = 0
            one_box.xmin = int(bbox[0])
            one_box.ymin = int(bbox[1])
            one_box.xmax = int(bbox[2])
            one_box.ymax = int(bbox[3])
            
            one_box.probability = float(scores[i])
            one_box.class_id = "Person"
            one_box.kp_pose = kpts[i].flatten().tolist()
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1
        
        return bboxes_msg

    def image_cb(self, msg: Image) -> None:

        # convert image + predict
        inicio = time.time()
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        cv_image, _, _ = letterbox(cv_image, (self.W, self.H))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        tensor = blob(cv_image, return_seg=False)
        tensor = torch.asarray(tensor, device=self.device)
        results = self.yolo(tensor)
        bboxes, scores, kpts = pose_postprocess(results.cpu().numpy(), self.threshold,
                                        0.5)

        bboxes_msg = self.yolox2bboxes_msgs(bboxes, kpts, scores, msg.header, cv_image)
        self.pub_boxes.publish(bboxes_msg)

        self.get_logger().info(f'Tempo de inferência: {time.time() - inicio}')
        
        # results = self.yolo.track(
        #     save=False,
        #     source=cv_image,
        #     verbose=False,
        #     stream=False,
        #     conf=self.threshold,
        #     device=self.device,
        #     tracker="bytetrack.yaml",
        #     iou=0.5,
        #     persist=True
        # )
        


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
