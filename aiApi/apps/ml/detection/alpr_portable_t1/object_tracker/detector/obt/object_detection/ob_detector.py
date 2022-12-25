import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, \
    non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.datasets import LoadWebcam, LoadImages
import cv2
import numpy as np
import os
import time

class yolov5:
    def __init__(self,
                 model_path: str = '/workspace/api-alpr/aiApi/apps/ml/detection/alpr_portable_t1/object_tracker/detector/obt/object_detection/yolov5/weights/yolov5m6.pt',
                 augment: bool = False,
                 conf_thres: float = 0.40,
                 # iou_thres: float = 0.213,
                 iou_thres: float = 0.45,
                 classes=None,
                 max_det: int = 1000,
                 agnostic_nms: bool = False,
                 imgsz: int = 640,
                 half: bool = False,
                 device: str = os.getenv("gpu")
                 ):
        set_logging()

        self.model_path = model_path
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms

        self.imgsz = imgsz
        print(os.getenv("gpu"))
        self.device = f'cuda:{os.getenv("gpu")}'

        #try:
        #   self.device = select_device(os.getenv("gpu"))
        #except:
        #   self.device = select_device(device)
        print(self.device)
        #self.half = half
        #self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.model_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.class_names = self.names
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
            next(self.model.parameters())))  # run once
        # if self.half:
        #     self.model.half()  # to FP16

    @staticmethod
    def xyxy_to_xywh(boxes_xyxy):
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xywh = boxes_xyxy.clone()
        elif isinstance(boxes_xyxy, np.ndarray):
            boxes_xywh = boxes_xyxy.copy()

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        return boxes_xywh

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    @torch.no_grad()
    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"

        # print(self.imgsz)
        # cv2.imshow("img1", img.shape)
        img, ratio, (dw, dh) = self.letterbox(ori_img, new_shape=self.imgsz, stride=self.stride)
        # cv2.imshow("img1", img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        img = np.ascontiguousarray(img)
        # print(img.shape)
        # print(img.max(), img.min())

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time.time()
        # print(img.shape)
        # print(img.max(), img.min())
        pred = self.model(img, augment=self.augment)[0]
        # print(pred.shape)
        pred = pred.cpu()
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        t2 = time.time()
        # print(pred)
        # print(t2-t1)
        for i, det in enumerate(pred):  # detections per image
            im0 = ori_img.copy()
            if len(det) > 0:
                # print(det.shape)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                bbox = self.xyxy_to_xywh(det[:, :4])
                cls_conf = det[:, 4]
                cls_ids = det[:, 5]
            else:
                bbox = torch.FloatTensor([]).reshape([0, 4])
                cls_conf = torch.FloatTensor([])
                cls_ids = torch.LongTensor([])
        return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

    @torch.no_grad()
    def predict(self, images_path, rtsp=False, crop_path: str = '', filter: int = 0, IMG_SHOW=True, CROP=False):
        # print(f'names: {names}')
        results = []
        if not rtsp:
            dataset = LoadImages(images_path, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadWebcam(images_path, img_size=self.imgsz, stride=self.stride)
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
            next(self.model.parameters())))  # run once
        count = 0
        fps = 0
        frame_skip = 1
        frame_count = 0
        ob_count = 0
        start = time.time()
        for path, img, im0s, vid_cap in dataset:
            # print(img.shape)
            count += 1
            frame_count += 1
            # cv2.imshow("test", im0s)
            # print(len(img))
            # print(type(im0s[0]))
            # print(img.shape, im0s[0].shape)
            # cv2.imshow("test", img[0])
            # cmd = cv2.waitKey(0)
            if frame_count < frame_skip:
                continue
            frame_count = 0
            # print(img.shape)
            cv2.imshow("img", im0s)
            cv2.waitKey(1)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time.time()
            # print(img.shape)
            # return None
            pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            t2 = time.time()
            # print(t2-t1)

            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                # print(len(im0))
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0  # for save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (cls, *xywh, conf)  # label format
                        # print(xywh)
                        c = int(cls)  # integer class
                        label = f'{self.names[c]} {conf:.2f}'
                        # print(f'[{c}]label: {label}')
                        if c == filter:
                            # print(xyxy)
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            # cv2.imshow("im", im0s[c1[1]:c2[1], c1[0]:c2[0], :])
                            # print(c1, c2)

                            # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                            if CROP:
                                cv2.imwrite(f'{crop_path}/{ob_count}.jpg', im0s[c1[1]:c2[1], c1[0]:c2[0], :])
                                ob_count += 1
                        if IMG_SHOW:
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1)
                    if IMG_SHOW:
                        cv2.imshow(p, im0)
                        cmd = cv2.waitKey(1)
                        if cmd == ord('q'):
                            break
            if (time.time() - start) >= 1:
                start = time.time()
                fps = count
                count = 0
                    # results.append({
                    #     'images_path': path,
                    #     'images': im0s,
                    #     'images_boxes': im0,
                    #     'results': det
                    # })
            # print(f'{s} Done. ({t2 - t1:.3f}s , {1/(t2 - t1)} fps, {fps})')

        return results
