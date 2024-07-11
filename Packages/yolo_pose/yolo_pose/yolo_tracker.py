import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np

import time
from dataclasses import dataclass

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header

from ament_index_python.packages import get_package_share_directory


from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 1000
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = True

class TrackerNode(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_node")

        self.cv_bridge = CvBridge()

        # create BYTETracker instance
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

        self.get_logger().info(f'Modelo pronto para inferêcias')


        # subs
        self._sub = self.create_subscription(
            BoundingBoxes, '/yolov8/bounding_boxes', self.tracker_cb,
            10
        )
        
        # subs
        self.sub_pose = self.create_subscription(
            Image, '/image_pose', self.image_cb,
            10
        )
        
        self.pub_boxes = self.create_publisher(BoundingBoxes, '/bytetrack/bounding_boxes', 10)

        self.pub_id = self.create_publisher(Image, '/id_image', 10)

        self.image = None


    # converts Detections into format that can be consumed by match_detections_with_tracks function
    def detections2boxes(self, detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]
        ))


    # converts List[STrack] into format that can be consumed by match_detections_with_tracks function
    def tracks2boxes(self, tracks) -> np.ndarray:
        return np.array([
            track.tlbr
            for track
            in tracks
        ], dtype=float)


    # matches our bounding boxes with predictions
    def match_detections_with_tracks(
        self,
        xyxy,
        tracks
    ):
        if not np.any(xyxy) or len(tracks) == 0:
            return np.empty((0,))

        tracks_boxes = self.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, xyxy)
        track2detection = np.argmax(iou, axis=1)

        tracker_ids = [None] * len(xyxy)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id

        return tracker_ids

    def tracker_cb(self, msg: BoundingBoxes) -> None:

        # convert image + predict
        scores = []
        xyxy = []

        inicio = time.time()
        print(msg)
        scores += [ [box.probability] for box in msg.bounding_boxes]
        # scores = np.array(scores)
        xyxy += [ [box.xmin ,box.ymin, box.xmax, box.ymax] for box in msg.bounding_boxes]   
        xyxy = np.array(xyxy)  

        # tracking detections
        if len(xyxy) > 0:
            tracks = self.byte_tracker.update(
                output_results= np.hstack((xyxy, scores)),
                img_info=(480,640,3),
                img_size=(480,640,3)
            )
            print(xyxy.shape)
            tracker_id = self.match_detections_with_tracks(xyxy = xyxy, tracks=tracks)
            
            for i in range(len(tracker_id)):
                if tracker_id[i] is not None:
                    msg.bounding_boxes[i].id = tracker_id[i]
                    if self.image is not None:
                        y = (msg.bounding_boxes[i].ymin + msg.bounding_boxes[i].ymax)/2
                        x = (msg.bounding_boxes[i].xmin + msg.bounding_boxes[i].xmax)/2
                        image = cv2.putText(img = self.image,
                            text =  f"Id: {int(tracker_id[i])}",
                            org = (int(x),int(y)),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 1,
                            color = (0, 0, 0),
                            thickness = 3
                            )
                        self.pub_id.publish(self.cv_bridge.cv2_to_imgmsg(image))
            self.pub_boxes.publish(msg)
            self.get_logger().info(f'Tempo de inferência: {time.time() - inicio}')
        

    def image_cb(self, msg: Image) -> None:


        self.image = self.cv_bridge.imgmsg_to_cv2(msg)


def main():
    rclpy.init()
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
