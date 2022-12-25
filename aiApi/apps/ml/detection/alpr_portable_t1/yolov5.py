from object_tracker.detector.obt import build_detector
from dotenv import load_dotenv
from os import getenv
from os.path import join
from object_tracker.utils.parser import get_config
import cv2
import os


class PortA:
    def __init__(self):
        load_dotenv(f'{os.path.dirname(os.path.realpath(__file__))}/object_tracker/.env')
        deep_sort_info = dict(REID_CKPT=join(getenv('project_root'), getenv('reid_ckpt')),
                              MAX_DIST=0.2,
                              MIN_CONFIDENCE=.3,
                              NMS_MAX_OVERLAP=0.5,
                              MAX_IOU_DISTANCE=0.7,
                              N_INIT=3,
                              MAX_AGE=10,
                              NN_BUDGET=100)
        deep_sort_dict = {'DEEPSORT': deep_sort_info}
        
        # get config to dict.
        cfg = get_config()
        cfg.merge_from_dict(deep_sort_dict)
        
        # load yolo detector
        self.detector = build_detector(cfg, False)
        
        # labels of coco 80 class for yolo
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                       'cow',
                       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                       'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard',
                       'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                       'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                       'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear',
                       'hair drier', 'toothbrush']

    def predict(self, img, mode=0):
        # prediction
        bbox_xywh, cls_conf, cls_ids = self.detector(img)

        results = []
        # mode json.
        if mode == 0:
            for i, _bbox_xywh in enumerate(bbox_xywh):
                results.append({
                    "confidence": float(cls_conf[i]),
                    "label": self.labels[int(cls_ids[i])],
                    "points": _bbox_xywh.tolist(),
                    "type": "rectangle"
                })
            return results
        # mode image.
        elif mode == 1:
            img = self.yolov5_draw_boxes(img, bbox_xywh, cls_conf, cls_ids)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, img_buf = cv2.imencode(".jpg", img)
            # return
            return img_buf.tobytes()

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    def _xywh_to_xyxy(self, bbox_xywh, img_shape):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), img_shape[1] - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), img_shape[0] - 1)
        return x1, y1, x2, y2

    def yolov5_draw_boxes(self, img1, bbox_xywh, cls_conf, cls_ids):
        bbox_xyxy = [self._xywh_to_xyxy(x, img1.shape[:2]) for x in bbox_xywh]
        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(ii) for ii in box]
            cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = '{} {:.2f}'.format(self.labels[int(cls_ids[i])], cls_conf[i])

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img1, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (255, 255, 255), -1)
            cv2.putText(img1, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 2)
        return img1

    def track_draw_boxes(self, img, output, offset=(0, 0)):
        croped = {}
        img_overlay = img.copy()
        img_crop = img.copy()
        for i, box in enumerate(output):
            x1, y1, x2, y2, identity = [int(ii) for ii in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # box text and bar
            color = self.compute_color_for_labels(identity)
            label = '{}{:d}'.format("", identity)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            croped['{}{:d}'.format("", identity)] = img_crop[y1:y2, x1:x2, :]
            
            cv2.circle(img_overlay, (x1 + int((x2 - x1) / 2), y2), 5, (0, 0, 255), 5)
            cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img_overlay, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img_overlay, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img_overlay, croped


if __name__ == '__main__':
    sg = PortA()
    img = cv2.imread("004545.jpg")
    sg.predict(img)
