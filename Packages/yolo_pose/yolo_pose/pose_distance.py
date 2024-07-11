import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2 as rs2

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
import numpy as np
from rclpy.time import Time
import time
import math

from yolo_pose.utils import letterbox

class DistanceNode(Node):

    def __init__(self) -> None:
        super().__init__("pose_distance_node")

        self.cv_bridge = CvBridge()

        self.tmp = self.create_publisher(Image, 'tmp', 10)

        self.sub =  self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self.image_depth_cb,
            # Image, "/camera/depth/image_rect_raw", self.image_depth_cb,
            qos_profile_sensor_data
        )
        self.sub_info =  self.create_subscription(
            CameraInfo, "/camera/aligned_depth_to_color/camera_info", self.image_depthinfo_cb,
            # CameraInfo, "/camera/depth/camera_info", self.image_depthinfo_cb,
            qos_profile_sensor_data
        )

        self._sub = self.create_subscription(
             Point, 'keypoint_center', self.kpts_cb,
            10
        )

        self.pub_3d = self.create_publisher(PointStamped, 'follow_3d_point', 10)

        # self.sub_conf = self.create_subscription("/camera/confidence/image_rect_raw", Image, self.confidenceCallback)

        self.intrinsics = None
        self.pix = None
        self.pix_grade = None

        self.image = None

    def image_depth_cb(self, msg: Image) -> None:
        
        try:
            inicio = time.time()
            self.image = self.cv_bridge.imgmsg_to_cv2(msg, msg.encoding)
            # # pick one pixel among all the pixels with the closest range:
            # indices = np.array(np.where(self.image == self.image[self.image > 0].min()))[:,0]
            # pix = (indices[1], indices[0])
            # self.pix = pix 
            rows, cols = self.image.shape
            # self.get_logger().info(f'Shape imagem depth: {self.image.shape} ')
            # self.pix = (cols // 2 + 500, rows // 2 - 500)


            if self.intrinsics and self.pix and self.image.shape == (720, 1280):

                # Definir o tamanho da ROI
                roi_size = 10
                x, y = self.pix
                x_min = max(0, x - roi_size // 2)
                x_max = min(cols, x + roi_size // 2)
                y_min = max(0, y - roi_size // 2)
                y_max = min(rows, y + roi_size // 2)

                # Extrair a ROI
                roi = self.image[y_min:y_max, x_min:x_max]

                # Calcular a mediana dos valores de profundidade na ROI
                depth_median = np.median(roi[roi > 0])

                self.get_logger().info(f' Pixel: %8.2f %8.2f.' % (self.pix[0], self.pix[1]))
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [self.pix[0], self.pix[1]], depth_median)
                self.get_logger().info(f' Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2]))
                point_3d = PointStamped()
                point_3d.header.stamp = Time().to_msg()
                point_3d.header.frame_id = 'camera_link'
                point_3d.point.x = result[2]/1000
                point_3d.point.y = -result[0]/1000
                point_3d.point.z = -result[1]/1000
                self.pub_3d.publish(point_3d)   
                
                # Converter a imagem em escala de cinza para BGR para desenhar o retângulo
                imag = self.image.copy()
                self.get_logger().info(f' shape: {self.image.shape} ')

                # Desenhar o círculo no pixel de interesse
                cv2.circle(imag, (x, y), 5, (255, 0, 0), -1)

                # Desenhar a ROI na imagem
                # cv2.rectangle(imag, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                # Desenhar as bordas da ROI na imagem
                imag[y_min:y_max, x_min] = 10000  # Linha esquerda
                imag[y_min:y_max, x_max-1] = 10000  # Linha direita
                imag[y_min, x_min:x_max] = 10000  # Linha superior
                imag[y_max-1, x_min:x_max] = 10000  # Linha inferior
                # cv2.imwrite('src/yolo_pose/config/erro2.png', imag)


                self.get_logger().info(f'Distance: {math.sqrt(((point_3d.point.x )**2) + ((point_3d.point.y)**2) + ((point_3d.point.z)**2))}')
                self.tmp.publish(self.cv_bridge.cv2_to_imgmsg(imag))

            else:
                self.get_logger().info(f'Shape da imagem incorreto')
                self.get_logger().info(f'Image shape: {self.image.shape} ')


            self.get_logger().info(f'Tempo de inferência: {time.time() - inicio}')
        
        except:
            cv2.imwrite('src/yolo_pose/config/erro.png', self.image)
            return


    def image_depthinfo_cb(self, cameraInfo):

        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.get_logger().info(f'{cameraInfo.d}')
            try:
                self.intrinsics.coeffs = [i for i in cameraInfo.d]
            except:
                self.intrinsics.coeffs = [0, 0, 0, 0, 0]
        except CvBridgeError as e:
            print(e)
            return

    def kpts_cb(self, msg: Point) -> None:

        if self.image is not None:
            x = int(msg.x)
            y = int(msg.y)
            self.pix = (x, y)

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