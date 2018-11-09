import tensorflow as tf
from dark_net import Darknet
from yolo_v3 import Yolov3
slim = tf.contrib.slim


class DETECTERSUBNET:
    def __init__(self, name, image_size, anchors,
                 batch_size=16,
                 num_anchors=9,
                 learning_rate=0.001,
                 num_classes=1,
                 num_id1=1,
                 num_id2=2,
                 num_id3=8,
                 num_id4=8,
                 num_id5=4,
                 norm='batch',
                 threshold=0.5,
                 max_num_boxes_per_image=20,
                 traininig=True,
                 confidence_score=0.7
                 ):
        self.confidence_score = confidence_score
        self.training = traininig
        self.name = name
        self.image_size = image_size
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        # self.max_num_boxes_per_image = ((self.image_size // 32) *
        #                                 (self.image_size // 32)
        #                                 + (self.image_size // 16)
        #                                 * (self.image_size // 16)
        #                                 + (self.image_size // 8)
        #                                 * (self.image_size // 8))
        self.max_num_boxes_per_image = max_num_boxes_per_image
        self.num_id1 = num_id1
        self.num_id2 = num_id2
        self.num_id3 = num_id3
        self.num_id4 = num_id4
        self.num_id5 = num_id5
        self.norm = norm
        self.threshold = threshold
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.num_anchors_per_detector = self.num_anchors // 3
        self.num_detectors_per_image = self.num_anchors_per_detector * ((self.image_size // 32) *
                                                                        (self.image_size // 32)
                                                                        + (self.image_size // 16)
                                                                        * (self.image_size // 16)
                                                                        + (self.image_size // 8)
                                                                        * (self.image_size // 8))

        # self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])

        self.Y_true_data = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.num_detectors_per_image, self.num_classes + 5])

        self.Y_true_boxes = tf.placeholder(dtype=tf.float32,
                                           shape=[None,
                                                  self.max_num_boxes_per_image * self.num_anchors_per_detector, 4])

        self.darknet = Darknet('darknet', is_training=self.is_training, num_id1=self.num_id1,
                               num_id2=self.num_id2, num_id3=self.num_id3,
                               num_id4=self.num_id4, num_id5=self.num_id5,
                               norm=self.norm)

        self.yolo_v3 = Yolov3('yolo_v3', is_training=self.is_training,
                              num_classes=self.num_classes, anchors=self.anchors,
                              norm=self.norm, image_size=self.image_size,
                              training=self.training)

    def yolo_v3_loss(self, yolo_out):

        def yolo_loss_for_each_scale(yolo_layer_outputs, conv_layer_outputs,
                                     yolo_true, yolo_true_boxes, ignore_thresh, anchors,
                                     num_classes=3, h=416, w=416, batch_size=16):

            def iou(yolo_out_pred, yolo_true_boxes_, shape_, batch_size_=16):
                yolo_true_boxes_ = tf.reshape(yolo_true_boxes_, [batch_size_, -1, 4])
                yolo_true_boxes_ = tf.expand_dims(yolo_true_boxes_, axis=1)
                true_coords_xy = yolo_true_boxes_[:, :, :, 0: 2]
                true_coords_wh = yolo_true_boxes_[:, :, :, 2: 4]
                true_up_left = true_coords_xy - true_coords_wh * 0.5
                true_down_right = true_coords_xy + true_coords_wh * 0.5

                true_area = true_coords_wh[:, :, :, 0] * true_coords_wh[:, :, :, 1]

                yolo_out_pred = tf.reshape(yolo_out_pred, [-1, shape_[1] * shape_[2] * shape_[3], 4])
                yolo_out_pred = tf.expand_dims(yolo_out_pred, axis=-2)
                pred_coords_xy = yolo_out_pred[:, :, :, 0: 2]
                pred_coords_wh = yolo_out_pred[:, :, :, 2: 4]
                pred_up_left = pred_coords_wh - pred_coords_wh * 0.5
                pred_down_right = pred_coords_xy + pred_coords_wh * 0.5

                pred_area = pred_coords_wh[:, :, :, 0] * pred_coords_wh[:, :, :, 1]

                intersects_up_left = tf.maximum(true_up_left, pred_up_left)
                intersects_down_right = tf.minimum(true_down_right, pred_down_right)

                intersects_wh = tf.maximum(intersects_down_right - intersects_up_left, 0.0)

                intersects_area = intersects_wh[:, :, :, 0] * intersects_wh[:, :, :, 1]

                iou_ = intersects_area / (pred_area + true_area - intersects_area)

                return tf.reduce_max(iou_, axis=-1)

            num_anchors = len(anchors)
            shape = yolo_layer_outputs.get_shape().as_list()

            yolo_out_pred_rela = yolo_layer_outputs[..., 0: 4] / tf.cast(tf.constant([w, h, w, h]), tf.float32)

            conv_layer_outputs = tf.reshape(conv_layer_outputs, [-1, shape[1], shape[2], shape[3], shape[4]])
            pred_conf = conv_layer_outputs[..., 4: 5]
            # pred_class = conv_layer_outputs[..., 5:]

            yolo_true = tf.reshape(yolo_true, [-1, shape[1], shape[2], shape[3], shape[4]])
            percent_x, percent_y, percent_w, percent_h, obj_mask, classes = tf.split(yolo_true,
                                                                                     [1, 1, 1, 1, 1, num_classes],
                                                                                     axis=-1)

            clustroid_x = tf.tile(tf.reshape(tf.range(shape[2], dtype=tf.float32), [1, -1, 1, 1]), [shape[2], 1, 1, 1])
            clustroid_y = tf.tile(tf.reshape(tf.range(shape[1], dtype=tf.float32), [-1, 1, 1, 1]), [1, shape[1], 1, 1])
            converted_x_true = percent_x * shape[2] - clustroid_x
            converted_y_true = percent_y * shape[1] - clustroid_y

            anchors = tf.constant(anchors, dtype=tf.float32)
            anchors_w = tf.reshape(anchors[:, 0], [1, 1, 1, num_anchors, 1])
            anchors_h = tf.reshape(anchors[:, 1], [1, 1, 1, num_anchors, 1])

            converted_w_true = tf.log((percent_w / anchors_w) * w)
            converted_h_true = tf.log((percent_h / anchors_h) * h)

            yolo_raw_box_true = tf.concat([converted_x_true, converted_y_true, converted_w_true, converted_h_true],
                                          axis=-1)
            yolo_raw_box_true = tf.where(tf.is_inf(yolo_raw_box_true),
                                         tf.zeros_like(yolo_raw_box_true),
                                         yolo_raw_box_true)

            box_loss_scale = 2 - yolo_true[..., 2: 3] * yolo_true[..., 3: 4]

            coords_xy_loss = (tf.nn.sigmoid_cross_entropy_with_logits(labels=yolo_raw_box_true[..., 0: 2],
                                                                      logits=conv_layer_outputs[..., 0: 2])
                              * obj_mask * box_loss_scale)
            coords_xy_loss = tf.reduce_sum(coords_xy_loss)

            coords_wh_loss = tf.square(yolo_raw_box_true[..., 2: 4]
                                       - conv_layer_outputs[..., 2: 4]) * 0.5 * obj_mask * box_loss_scale

            coords_wh_loss = tf.reduce_sum(coords_wh_loss)

            coords_loss = coords_xy_loss + coords_wh_loss

            box_iou = iou(yolo_out_pred_rela, yolo_true_boxes, shape, batch_size)

            ignore_mask = tf.cast(tf.less(box_iou, ignore_thresh * tf.ones_like(box_iou)), tf.float32)
            ignore_mask = tf.reshape(ignore_mask, [-1, shape[1], shape[2], num_anchors])
            ignore_mask = tf.expand_dims(ignore_mask, -1)

            back_loss = ((1 - obj_mask)
                         * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=pred_conf) * ignore_mask)
            back_loss = tf.reduce_sum(back_loss)

            fore_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=pred_conf)

            fore_loss = tf.reduce_sum(fore_loss)

            conf_loss = back_loss + fore_loss

            # cls_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=classes, logits=pred_class)
            # cls_loss = tf.reduce_sum(cls_loss)

            return coords_loss + conf_loss

        num_anchors_per_detector = len(self.anchors) // 3
        max_num_boxes_per_image = self.Y_true_boxes.shape[1] // 3

        num_large_detectors = int((self.image_size / 32) * (self.image_size / 32) * num_anchors_per_detector)
        num_medium_detectors = int((self.image_size / 16) * (self.image_size / 16) * num_anchors_per_detector)

        large_yolo_true_raw = self.Y_true_data[:, :num_large_detectors, :]
        medium_yolo_true_raw = self.Y_true_data[:, num_large_detectors: num_medium_detectors + num_large_detectors, :]
        small_yolo_true_raw = self.Y_true_data[:, num_medium_detectors + num_large_detectors:, :]

        large_yolo_true_boxes = self.Y_true_boxes[:, :max_num_boxes_per_image, :]
        medium_yolo_true_boxes = self.Y_true_boxes[:, max_num_boxes_per_image: 2*max_num_boxes_per_image, :]
        small_yolo_true_boxes = self.Y_true_boxes[:, 2*max_num_boxes_per_image:, :]

        yolo_layer_outputs_ = yolo_out[:3]
        conv_layer_outputs_ = yolo_out[3:]

        large_yolo_pred_boxes = yolo_layer_outputs_[0]
        medium_yolo_pred_boxes = yolo_layer_outputs_[1]
        small_yolo_pred_boxes = yolo_layer_outputs_[2]

        large_yolo_pred_raw = conv_layer_outputs_[0]
        medium_yolo_pred_raw = conv_layer_outputs_[1]
        small_yolo_pred_raw = conv_layer_outputs_[2]

        large_obj_loss = yolo_loss_for_each_scale(large_yolo_pred_boxes, large_yolo_pred_raw,
                                                  large_yolo_true_raw, large_yolo_true_boxes,
                                                  ignore_thresh=self.threshold,
                                                  anchors=self.anchors[num_anchors_per_detector*2:],
                                                  num_classes=self.num_classes, h=self.image_size,
                                                  w=self.image_size,
                                                  batch_size=self.batch_size
                                                  )

        medium_obj_loss = yolo_loss_for_each_scale(medium_yolo_pred_boxes, medium_yolo_pred_raw,
                                                   medium_yolo_true_raw, medium_yolo_true_boxes,
                                                   ignore_thresh=self.threshold,
                                                   anchors=self.anchors[num_anchors_per_detector:
                                                                        2*num_anchors_per_detector],
                                                   num_classes=self.num_classes, h=self.image_size,
                                                   w=self.image_size,
                                                   batch_size=self.batch_size
                                                   )

        small_obj_loss = yolo_loss_for_each_scale(small_yolo_pred_boxes, small_yolo_pred_raw,
                                                  small_yolo_true_raw, small_yolo_true_boxes,
                                                  ignore_thresh=self.threshold,
                                                  anchors=self.anchors[:num_anchors_per_detector],
                                                  num_classes=self.num_classes, h=self.image_size,
                                                  w=self.image_size,
                                                  batch_size=self.batch_size
                                                  )

        return (large_obj_loss + medium_obj_loss + small_obj_loss) / self.batch_size

    def yolo_v3_optimizer(self, yolo_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = self.learning_rate
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, name=name).
                minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step
        trainable_var_list = tf.trainable_variables()
        last_layer_var_list = [i for i in trainable_var_list if
                               i.shape[-1] == (5 + self.num_classes) * self.num_anchors_per_detector]
        last_layer_optimizer = make_optimizer(yolo_loss, last_layer_var_list)
        yolo_optimizer = make_optimizer(yolo_loss, trainable_var_list)

        return last_layer_optimizer, yolo_optimizer

    def coords_to_boxes(self, yolo_boxes_out, is_pred=True):

        x_pred, y_pred, w_pred, h_pred, confs_pred, classes_pred = tf.split(yolo_boxes_out,
                                                                            [1, 1, 1, 1, 1,
                                                                             self.num_classes],
                                                                            axis=-1)
        if is_pred:
            x_pred = x_pred / self.image_size
            y_pred = y_pred / self.image_size
            w_pred = w_pred / self.image_size
            h_pred = h_pred / self.image_size

        up_left_x = x_pred - w_pred / 2.0
        up_left_y = y_pred - h_pred / 2.0
        down_right_x = x_pred + w_pred / 2.0
        down_right_y = y_pred + h_pred / 2.0

        detections = tf.concat([up_left_x, up_left_y, down_right_x, down_right_y, confs_pred, classes_pred], axis=-1)

        return detections

    def draw_boxes(self, yolo_out):
        yolo_boxes_out = yolo_out[:3]

        yolo_boxes_out_large_obj = yolo_boxes_out[0]
        yolo_boxes_out_medium_obj = yolo_boxes_out[1]
        yolo_boxes_out_small_obj = yolo_boxes_out[2]

        yolo_boxes_out_large_obj = tf.reshape(yolo_boxes_out_large_obj, [self.batch_size, -1, self.num_classes + 5])
        yolo_boxes_out_medium_obj = tf.reshape(yolo_boxes_out_medium_obj, [self.batch_size, -1,
                                                                           self.num_classes + 5])
        yolo_boxes_out_small_obj = tf.reshape(yolo_boxes_out_small_obj, [self.batch_size, -1, self.num_classes + 5])

        yolo_boxes_out = tf.concat([yolo_boxes_out_large_obj, yolo_boxes_out_medium_obj, yolo_boxes_out_small_obj],
                                   axis=1)

        detections = self.coords_to_boxes(yolo_boxes_out)
        confs_pred = detections[:, :, 4]

        conf_mask = tf.cast(tf.expand_dims(tf.greater(confs_pred,
                                                      tf.ones_like(confs_pred) * self.confidence_score), -1),
                            tf.float32)
        predictions = detections * conf_mask

        pred_images = []

        for i in range(self.batch_size):
            conf_pred = predictions[i, :, 4]
            boxes_pred = predictions[i, :, 0: 4]
            up_left_x, up_left_y, down_right_x, down_right_y = tf.split(boxes_pred, [1, 1, 1, 1], axis=-1)

            boxes_pred = tf.concat([up_left_y, up_left_x, down_right_y, down_right_x], axis=-1)

            top_k_scores, top_k_indices = tf.nn.top_k(conf_pred, k=60)
            boxes_pred = tf.gather(boxes_pred, top_k_indices)

            desired_indices = tf.image.non_max_suppression(boxes_pred, top_k_scores, max_output_size=6)

            desired_boxes = tf.gather(boxes_pred, desired_indices)

            desired_boxes = tf.expand_dims(desired_boxes, axis=0)

            desired_boxes = tf.where(tf.less(desired_boxes, tf.zeros_like(desired_boxes)),
                                     tf.zeros_like(desired_boxes), desired_boxes)

            desired_boxes = tf.where(tf.greater(desired_boxes, tf.ones_like(desired_boxes)),
                                     tf.ones_like(desired_boxes), desired_boxes)

            pred_images.append(tf.image.draw_bounding_boxes(tf.expand_dims(self.X[i, :, :, :], axis=0), desired_boxes))

        pred_images = tf.concat(pred_images, axis=0)

        return pred_images

    def draw_true_boxes(self):

        yolo_true_data = self.coords_to_boxes(self.Y_true_data, is_pred=False)

        yolo_true_images = []

        for i in range(self.batch_size):
            conf_true = yolo_true_data[i, :, 4]
            boxes_true = yolo_true_data[i, :, 0: 4]

            up_left_x, up_left_y, down_right_x, down_right_y = tf.split(boxes_true, [1, 1, 1, 1], axis=-1)

            boxes_true = tf.concat([up_left_y, up_left_x, down_right_y, down_right_x], axis=-1)

            top_k_scores, top_k_indices = tf.nn.top_k(conf_true, k=20)

            boxes_true = tf.gather(boxes_true, top_k_indices)

            boxes_true = tf.expand_dims(boxes_true, axis=0)

            yolo_true_images.append(tf.image.draw_bounding_boxes(tf.expand_dims(self.X[i, :, :, :], axis=0),
                                                                 boxes_true))

        yolo_true_images = tf.concat(yolo_true_images, axis=0)

        return yolo_true_images

    def model(self, ipt):
        self.X = ipt

        dark_out, dark_route_1, dark_route_2 = self.darknet(self.X)

        yolo_out = self.yolo_v3(dark_out, dark_route_1, dark_route_2)

        loss = self.yolo_v3_loss(yolo_out)

        tf.summary.scalar('loss', loss)

        origin_images = self.draw_true_boxes()

        tf.summary.image('origin_images', origin_images)

        pred_images = self.draw_boxes(yolo_out)

        tf.summary.image('prediction_images', pred_images)

        return loss

    def out(self):
        dark_out, dark_route_1, dark_route_2 = self.darknet(self.X)

        yolo_out = self.yolo_v3(dark_out, dark_route_1, dark_route_2)

        return yolo_out
