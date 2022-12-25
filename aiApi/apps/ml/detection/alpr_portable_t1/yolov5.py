from object_tracker.detector.obt import build_detector
from dotenv import load_dotenv
from os import getenv
from os.path import join
from object_tracker.utils.parser import get_config
import cv2
import os


class PortA:
    def __init__(self):
        # print(os.getcwd(), os.path.dirname(os.path.realpath('__file__')), __file__, f'{__file__.replace("single_shot.py", "")}/object_tracker/.env')
        load_dotenv(f'{os.path.dirname(os.path.realpath(__file__))}/object_tracker/.env')
        # print(getenv('project_root'))
        deep_sort_info = dict(REID_CKPT=join(getenv('project_root'), getenv('reid_ckpt')),
                              MAX_DIST=0.2,
                              MIN_CONFIDENCE=.3,
                              NMS_MAX_OVERLAP=0.5,
                              MAX_IOU_DISTANCE=0.7,
                              N_INIT=3,
                              MAX_AGE=10,
                              NN_BUDGET=100)
        deep_sort_dict = {'DEEPSORT': deep_sort_info}
        cfg = get_config()
        cfg.merge_from_dict(deep_sort_dict)

        # self.alpr = ALPR_M(
        #     tc_detection_restore_path='m/yolov3-tc-detection-train-416_45000.ckpt',
        #     # tc_detection_restore_path='m/yolov3-tc-detection-train-416_45000-16.ckpt',
        #     # char_detect_restore_path='m/yolov3-voc-text-detector-train_76000.ckpt',
        #     plate_corner_detection_restore_path='m/t_mark_plate_m_01_final.h5',
        #     plate_detection_json_path='m/wpod-net.json',
        #     plate_detection_model_path='m/wpod-net.h5',
        #     plate_classification_restore_path='m/aa4-m2-224-plate_classify.h5',
        #     standalone_language_enable=True,
        #     standalone_language='thai',
        #     # standalone_language='malay',
        #     thai_ocr_restore_path='m/aa9-m10-thai_char_64.h5',
        #     thai_province_restore_path='m/aa6-m5-thai_province_256_64.h5',
        #     eng_ocr_restore_path='m/aa6-m-eng_char_64.h5',
        #     cambodia_province_restore_path='m/aa3_cambodia_province_256_64.h5',
        # )
        self.detector = build_detector(cfg, False)
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
        bbox_xywh, cls_conf, cls_ids = self.detector(img)

        # print(bbox_xywh)
        # img_overlay = self.yolov5_draw_boxes(img, bbox_xywh, cls_conf)
        results = []
        if mode == 0:
            for i, _bbox_xywh in enumerate(bbox_xywh):
                results.append({
                    "confidence": float(cls_conf[i]),
                    "label": self.labels[int(cls_ids[i])],
                    "points": _bbox_xywh.tolist(),
                    "type": "rectangle"
                })
            return results
        elif mode == 1:
            img = self.yolov5_draw_boxes(img, bbox_xywh, cls_conf, cls_ids)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, img_buf = cv2.imencode(".jpg", img)
            # return
            return img_buf.tobytes()
            # return "{}"
        # import json
        # return json.dumps(results)
        # print(json.dumps(results))
        # cv2.imshow("img_overlay", img_overlay)
        # cv2.waitKey(0)

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
            # croped.append(img[y1:y2, x1:x2, :].copy())
            # (x1, y1), (x2, y2)
            # if line_cross[1][1] <= y2 <= end_line[1][1]:
            cv2.circle(img_overlay, (x1 + int((x2 - x1) / 2), y2), 5, (0, 0, 255), 5)
            cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img_overlay, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img_overlay, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img_overlay, croped


if __name__ == '__main__':
    sg = PortA()
    img = cv2.imread("004545.jpg")
    sg.predict(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
