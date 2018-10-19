import tensorflow as tf
import layer


class Yolov3:
    def __init__(self, name, is_training, num_classes, anchors, norm='batch', image_size=416, training=True):
        self.name = name
        self.is_training = is_training
        self.num_classes = num_classes
        self.norm = norm
        self.reuse = False
        self.anchors = anchors
        self.image_size = image_size
        self.training = training

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

            large_obj_box_detections = layer.yolo_layer(large_obj_raw_detections, name='yolo_large_box',
                                                        anchors=self.anchors[6:9],
                                                        num_classes=self.num_classes,
                                                        image_size=self.image_size)

            medium_obj_box_detections = layer.yolo_layer(medium_obj_raw_detections, name='yolo_medium_box',
                                                         anchors=self.anchors[3:6],
                                                         num_classes=self.num_classes,
                                                         image_size=self.image_size)

            small_obj_box_detections = layer.yolo_layer(small_obj_raw_detections, name='yolo_small_box',
                                                        anchors=self.anchors[0:3],
                                                        num_classes=self.num_classes,
                                                        image_size=self.image_size)

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.training:
            with tf.variable_scope('reshape_yolo_large'):
                large_obj_box_detections = tf.reshape(large_obj_box_detections,
                                                      [-1,
                                                       large_obj_box_detections.shape[1]
                                                       * large_obj_box_detections.shape[2]
                                                       * large_obj_box_detections.shape[3],
                                                       self.num_classes + 5])

            with tf.variable_scope('reshape_yolo_medium'):
                medium_obj_box_detections = tf.reshape(medium_obj_box_detections,
                                                       [-1,
                                                        medium_obj_box_detections.shape[1]
                                                        * medium_obj_box_detections.shape[2]
                                                        * medium_obj_box_detections.shape[3],
                                                        self.num_classes + 5])

            with tf.variable_scope('reshape_yolo_small'):
                small_obj_box_detections = tf.reshape(small_obj_box_detections,
                                                      [-1,
                                                       small_obj_box_detections.shape[1]
                                                       * small_obj_box_detections.shape[2]
                                                       * small_obj_box_detections.shape[3],
                                                       self.num_classes + 5])

            return large_obj_box_detections, medium_obj_box_detections, small_obj_box_detections

        else:

            return large_obj_box_detections, medium_obj_box_detections, small_obj_box_detections, \
              large_obj_raw_detections, medium_obj_raw_detections, small_obj_raw_detections
