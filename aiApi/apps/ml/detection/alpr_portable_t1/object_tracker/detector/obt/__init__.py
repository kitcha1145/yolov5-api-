


__all__ = ['build_detector']

# def build_detector(cfg, use_cuda):
#     from .YOLOv3 import YOLOv3
#     return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
#                     score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
#                     is_xywh=True, use_cuda=use_cuda)
def build_detector(cfg, use_cuda):
    import sys
    sys.path.append(f"{__file__}".replace("__init__.py", ""))

    from .object_detection.ob_detector import yolov5
    if use_cuda:
        return yolov5(device='0')
    return yolov5(device='cpu')
