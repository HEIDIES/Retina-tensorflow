import numpy as np
import math
import hyper_parameters
from os import scandir
import json_convert
import cv2


def get_detector_heatmap(image_ids, image_heights, image_widths, labels):
    heatmaps = []
    for k in range(5):
        heatmap = np.zeros(shape=[hyper_parameters.FLAGS.batch_size, hyper_parameters.FLAGS.image_size //
                                  int(8 * (2 ** k)),
                                  hyper_parameters.FLAGS.image_size // int(8 * (2 ** k)),
                                  hyper_parameters.FLAGS.num_anchors,
                                  hyper_parameters.FLAGS.num_classes +
                                  hyper_parameters.FLAGS.bbox_dims
                                  ],
                           dtype=np.float32)
        for i in range(len(image_ids)):
            label = labels[image_ids[i].decode('utf-8')]
            origin_height = image_heights[i]
            origin_width = image_widths[i]
            for key in label[0]:
                up_left_x = label[0][key][0]
                up_left_y = label[0][key][1]
                down_right_x = label[0][key][2]
                down_right_y = label[0][key][3]
                center_x = (up_left_x + down_right_x) / 2.0
                center_y = (up_left_y + down_right_y) / 2.0
                w_obj = down_right_x - up_left_x
                h_obj = down_right_y - up_left_y
                # print('frame id : %d' %i)
                # print('up_left_point : (%02f, ' %up_left_x, '%02f)\n' %up_left_y, 
                #  'down_right_point : (%02f, ' %down_right_x, '%02f)\n' %down_right_y, 
                #  'center_point : (%02f, ' %center_x, '%02f)\n' %center_y, 
                #  'obj_width : %02f\n' %w_obj, 
                #  'obj_height : %02f\n' %h_obj)

                # print(k)

                _current_size = hyper_parameters.FLAGS.image_size // (8 * (2 ** k))
                x = center_x / origin_width * _current_size
                y = center_y / origin_height * _current_size
                if x < 0.0 or x > _current_size or y < 0.0 or y > _current_size:
                    break
                # print(x, y)
                for j in range(hyper_parameters.FLAGS.num_anchors):
                    heatmap[i][int(y) - 1][int(x) - 1][j][0] = float(x)
                    heatmap[i][int(y) - 1][int(x) - 1][j][1] = float(y)
                    heatmap[i][int(y) - 1][int(x) - 1][j][2] = math.sqrt(w_obj / origin_width)
                    heatmap[i][int(y) - 1][int(x) - 1][j][3] = math.sqrt(h_obj / origin_height)
                    heatmap[i][int(y) - 1][int(x) - 1][j][4] = 1.0

        heatmaps.append(np.reshape(heatmap, [hyper_parameters.FLAGS.batch_size,
                                             hyper_parameters.FLAGS.image_size // int(8 * (2 ** k)),
                                             hyper_parameters.FLAGS.image_size // int(8 * (2 ** k)),
                                             hyper_parameters.FLAGS.num_anchors *
                                             (
                                                  hyper_parameters.FLAGS.num_classes +
                                                  hyper_parameters.FLAGS.bbox_dims
                                             )]))
    return heatmaps


if __name__ == '__main__':
    _labels = json_convert.load_label('data/label/keypoint_train_annotations_20170909.json')
    input_dir = 'data/train'
    i_ = 0
    imgs = []
    img_ids = []
    img_heights = []
    img_widths = []
    for img_file in scandir(input_dir):
        i_ += 1
        if img_file.name.endswith('.jpg') and img_file.is_file():
            img_ids.append(img_file.name[:-4])
            img = cv2.imread(img_file.path, cv2.IMREAD_COLOR)
            imgs.append(img)
            height_, width_, _ = img.shape
            img_heights.append(height_)
            img_widths.append(width_)

        if i_ == 3:
            break

    heat_maps = get_detector_heatmap(img_ids, img_heights, img_widths, _labels)
    b = 0
    i_ = 0
    for img, width_, height_ in zip(imgs, img_widths, img_heights):
        # print(width, height)
        heat_map = np.reshape(heat_maps[b][i_], [hyper_parameters.FLAGS.image_size // int(8 * (2 ** b)),
                                                 hyper_parameters.FLAGS.image_size // int(8 * (2 ** b)),
                                                 hyper_parameters.FLAGS.num_anchors,
                                                 hyper_parameters.FLAGS.num_classes +
                                                 hyper_parameters.FLAGS.bbox_dims])
        _wh = np.square(heat_map[:, :, 0, 2: 4]) * np.reshape([width_, height_], [1, 1, 1, 2])
        current_size = hyper_parameters.FLAGS.image_size // (8 * (2 ** b))
        _centers = heat_map[:, :, 0, 0: 2] * np.reshape([width_, height_], [1, 1, 1, 2]) / current_size
        print(current_size)
        _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
        _up_left = np.squeeze(_up_left, 0)
        _down_right = np.squeeze(_down_right, 0)
        _confs = heat_map[:, :, 0, 4]
        rows, cols = _confs.shape
        for j_ in range(cols):
            for k_ in range(rows):
                if _confs[j_][k_] >= 0.5:
                    # print('center : (%d, ' %_centers[0][j][k][0], '%d)' %_centers[0][j][k][1])
                    # print(_up_left[j][k], _down_right[j][k])
                    cv2.rectangle(img, (int(_up_left[j_][k_][0]), int(_up_left[j_][k_][1])),
                                  (int(_down_right[j_][k_][0]), int(_down_right[j_][k_][1])), (0, 0, 255),
                                  thickness=2)
        # cv2.rectangle(im, (10, 10), (110, 110), (0, 0, 255), thickness=2)
        i_ += 1
        cv2.imshow('image' + str(i_), img)
    cv2.waitKey(0)
