import tensorflow as tf
import layer


class Yolov3:
    def __init__(self, name, is_training, num_classes, anchors, norm='batch', image_size = 416):
        self.name = name
        self.is_training = is_training
        self.num_classes = num_classes
        self.norm = norm
        self.reuse = False
        self.anchors = anchors
        self.image_size = image_size

    def __call__(self, dark_out, dark_route_1, dark_route_2):
        with tf.variable_scope(self.name):
            large_obj_raw_detections, yolo_route = layer.yolo_1(dark_out, num_classes=self.num_classes,
                                                                reuse=self.reuse, is_training=self.is_training,
                                                                norm=self.norm)
            medium_obj_raw_detections, yolo_route = layer.yolo_2(yolo_route, dark_route_2,
                                                                 num_classes=self.num_classes,
                                                                 reuse=self.reuse,
                                                                 is_training=self.is_training,
                                                                 norm=self.norm)
            small_obj_raw_detections = layer.yolo_3(yolo_route, dark_route_1,
                                                    num_classes=self.num_classes,
                                                    reuse=self.reuse,
                                                    is_training=self.is_training,
                                                    norm=self.norm)

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return
