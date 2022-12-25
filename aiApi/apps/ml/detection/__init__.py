__all__ = ['yolov5']


def yolov5():
    import sys, os
    # print(os.path.dirname(os.path.realpath(__file__)), __file__, f'{os.path.dirname(os.path.realpath(__file__))}/alpr_portable_t1')
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/alpr_portable_t1')
    from alpr_portable_t1.yolov5 import PortA
    return PortA()


