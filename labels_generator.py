import numpy as np
import hyper_parameters
from os import scandir
import json_convert
import cv2
import random
import utils


def build_data(label, image_height, image_width, max_boxes=5):

    scale_h, scale_w = hyper_parameters.FLAGS.image_size / image_height, \
                       hyper_parameters.FLAGS.image_size / image_width

    boxes = np.array([np.array(label[0][key] + [1]) for key in label[0]])

    boxes_data = np.zeros((max_boxes, 5))
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
        boxes_data[:len(boxes)] = boxes

    return boxes_data


def get_detector_heatmap(image_ids, image_heights, image_widths, labels):

    def get_detector_heatmap_each_scale(boxes_data_, best_anchors_, anchors_mask, grid_size, num_classes):

        num_anchors = len(anchors_mask)
        boxes_data_shape = boxes_data_.shape

        best_anchors_mask = np.isin(best_anchors_, anchors_mask, invert=True)
        best_anchors_ = best_anchors_ * 1
        best_anchors_ -= min(anchors_mask)
        best_anchors_[best_anchors_mask] = 0

        boxes_data_mask = np.ones_like(best_anchors)
        boxes_data_mask[best_anchors_mask] = 0
        boxes_data_mask = np.expand_dims(boxes_data_mask, -1)
        boxes_data_ = boxes_data_ * boxes_data_mask

        i = np.floor(boxes_data_[:, :, 1] * grid_size[0]).astype('int32')
        j = np.floor(boxes_data_[:, :, 0] * grid_size[1]).astype('int32')

        boxes_data_ = boxes_data_.reshape([-1, boxes_data_.shape[-1]])
        best_anchors_ = best_anchors_.reshape([-1, 1])
        i = i.reshape([-1, 1])
        j = j.reshape([-1, 1])

        classes = boxes_data_[:, -1].reshape([-1]).astype(np.int)
        one_hot_array = np.zeros([boxes_data_.shape[0], num_classes])
        one_hot_array[np.arange(boxes_data_.shape[0]), classes - 1] = 1

        boxes_data_mask = boxes_data_[:, 2] > 0
        boxes_data_[boxes_data_mask, 4] = 1
        boxes_data_ = np.concatenate([boxes_data_, one_hot_array], axis=-1)

        y_true = np.zeros([boxes_data_shape[0] * int(grid_size[0]) * int(grid_size[1]) * num_anchors, 5 + num_classes])

        image_offset = np.repeat(np.linspace(0, y_true.shape[0], boxes_data_shape[0], endpoint=False, dtype=np.int),
                                 boxes_data_.shape[0] / boxes_data_shape[0]).reshape([-1, 1])
        grid_offset = num_anchors * (grid_size[0] * i + j)

        indexing_array = np.array(image_offset + grid_offset + best_anchors_, dtype=np.int32)
        indexing_array = indexing_array[boxes_data_mask, :]
        indexing_array = indexing_array.reshape([-1])

        y_true[indexing_array, :] = boxes_data_[boxes_data_mask]
        y_true = y_true.reshape(
            [boxes_data_shape[0], int(grid_size[0]) * int(grid_size[1]) * num_anchors, num_classes + 5])
        boxes_data_ = boxes_data_.reshape([boxes_data_shape[0], boxes_data_shape[1], -1])

        return y_true, boxes_data_[..., 0:4]

    anchors = utils.get_anchors(hyper_parameters.FLAGS.anchors_path)
    anchors = np.array(anchors, dtype=np.float32)

    boxes_data = []

    for i in range(len(image_ids)):
        label = labels[image_ids[i].decode('utf-8')]
        origin_height = image_heights[i]
        origin_width = image_widths[i]

        box = build_data(label, origin_height, origin_width,
                         max_boxes=hyper_parameters.FLAGS.max_num_boxes_per_image)
        boxes_data.append(box)

    boxes_data = np.array(boxes_data)

    boxes_xy = (boxes_data[:, :, 0:2] + boxes_data[:, :, 2:4]) // 2
    boxes_hw = boxes_data[:, :, 2:4] - boxes_data[:, :, 0:2]

    boxes_data[:, :, 0] = boxes_xy[..., 0] / hyper_parameters.FLAGS.image_size
    boxes_data[:, :, 1] = boxes_xy[..., 1] / hyper_parameters.FLAGS.image_size
    boxes_data[:, :, 2] = boxes_hw[..., 0] / hyper_parameters.FLAGS.image_size
    boxes_data[:, :, 3] = boxes_hw[..., 1] / hyper_parameters.FLAGS.image_size

    hw = np.expand_dims(boxes_hw, -2)
    anchors_broad = np.expand_dims(anchors, 0)

    anchor_maxes = anchors_broad / 2.
    anchor_mins = -anchor_maxes
    box_maxes = hw / 2.
    box_mins = -box_maxes
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_hw = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]
    box_area = hw[..., 0] * hw[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    best_anchors = np.argmax(iou, axis=-1)

    large_obj_image_size = hyper_parameters.FLAGS.image_size // 32
    medium_obj_image_size = hyper_parameters.FLAGS.image_size // 16
    small_obj_image_size = hyper_parameters.FLAGS.image_size // 8

    large_obj_detectors, large_obj_boxes = get_detector_heatmap_each_scale(boxes_data,
                                                                           best_anchors_=best_anchors,
                                                                           anchors_mask=[6, 7, 8],
                                                                           grid_size=(large_obj_image_size,
                                                                                      large_obj_image_size),
                                                                           num_classes=hyper_parameters.FLAGS.
                                                                           num_classes)
    medium_obj_detectors, medium_obj_boxes = get_detector_heatmap_each_scale(boxes_data,
                                                                             best_anchors_=best_anchors,
                                                                             anchors_mask=[3, 4, 5],
                                                                             grid_size=(medium_obj_image_size,
                                                                                        medium_obj_image_size),
                                                                             num_classes=hyper_parameters.FLAGS.
                                                                             num_classes)
    small_obj_detectors, small_obj_boxes = get_detector_heatmap_each_scale(boxes_data,
                                                                           best_anchors_=best_anchors,
                                                                           anchors_mask=[0, 1, 2],
                                                                           grid_size=(small_obj_image_size,
                                                                                      small_obj_image_size),
                                                                           num_classes=hyper_parameters.FLAGS.
                                                                           num_classes)

    yolo_true_data = np.concatenate([large_obj_detectors, medium_obj_detectors, small_obj_detectors], axis=1)
    yolo_true_boxes = np.concatenate([large_obj_boxes, medium_obj_boxes, small_obj_boxes], axis=1)

    heatmaps = [yolo_true_data, yolo_true_boxes]

    return heatmaps


if __name__ == '__main__':
    _labels = json_convert.load_label('data/label/keypoint_train_annotations_20170909.json')
    input_dir = '../MultiPoseNet-tensorflow/data/train'
    large_obj_image_size_ = hyper_parameters.FLAGS.image_size // 32
    medium_obj_image_size_ = hyper_parameters.FLAGS.image_size // 16
    small_obj_image_size_ = hyper_parameters.FLAGS.image_size // 8

    num_large_detections = large_obj_image_size_ * large_obj_image_size_ * hyper_parameters.FLAGS.num_anchors // 3
    num_medium_detections = medium_obj_image_size_ * medium_obj_image_size_ * hyper_parameters.FLAGS.num_anchors // 3
    num_small_detections = small_obj_image_size_ * small_obj_image_size_ * hyper_parameters.FLAGS.num_anchors // 3

    i_ = 0
    imgs = []
    img_ids = []
    img_heights = []
    img_widths = []

    images_name = []

    for img_file in scandir(input_dir):
        images_name.append(img_file.name)

    idx = list(range(len(images_name)))
    random.seed(54321)
    random.shuffle(idx)
    images_name = [images_name[i] for i in idx]
    for img_file in images_name:

        if img_file.endswith('.jpg'):
            img_ids.append(img_file[:-4])
            img = cv2.imread(input_dir + '/' + img_file, cv2.IMREAD_COLOR)
            imgs.append(img)
            height_, width_, _ = img.shape
            img_heights.append(height_)
            img_widths.append(width_)

        if i_ == 3:
            break

        i_ += 1

    heat_maps = get_detector_heatmap(img_ids, img_heights, img_widths, _labels)
    b = 0
    i_ = 0
    for img, width_, height_ in zip(imgs, img_widths, img_heights):
        # print(width, height)
        heat_map_large_obj = np.reshape(heat_maps[0][i_][:num_large_detections, :],
                                        [large_obj_image_size_, large_obj_image_size_,
                                         hyper_parameters.FLAGS.num_anchors // 3,
                                         hyper_parameters.FLAGS.num_classes + 5])

        _wh = heat_map_large_obj[:, :, :, 2: 4] * np.reshape([width_, height_], [1, 1, 1, 2])
        _centers = heat_map_large_obj[:, :, :, 0: 2] * np.reshape([width_, height_], [1, 1, 1, 2])

        _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
        _up_left = np.squeeze(_up_left, 0)
        _down_right = np.squeeze(_down_right, 0)
        _confs = heat_map_large_obj[:, :, :, 4]
        _cls = heat_map_large_obj[:, :, 0, 5]
        rows, cols = _confs.shape
        for j_ in range(cols):
            for k_ in range(rows):
                for l_ in range(hyper_parameters.FLAGS.num_anchors // 3):
                    if _confs[j_][k_][l_] >= 0.5:
                        print('center : (%d, ' % _centers[0][j_][k_][l_], '%d)' % _centers[0][j_][k_][l_])
                        print(_up_left[j_][k_][l_], _down_right[j_][k_][l_])
                        cv2.rectangle(img, (int(_up_left[j_][k_][l_]), int(_up_left[j_][k_][l_])),
                                      (int(_down_right[j_][k_][l_]), int(_down_right[j_][k_][l_])), (0, 0, 255),
                                      thickness=2)
        # cv2.rectangle(im, (10, 10), (110, 110), (0, 0, 255), thickness=2)
        i_ += 1
        cv2.imshow('image_large_obj' + str(i_), img)
    cv2.waitKey(0)
