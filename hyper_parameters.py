import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 416, 'image size, default: 256')
tf.flags.DEFINE_float('learning_rate', 1e-3,
                      'initial learning rate for Adam, default: 0.0001')
tf.flags.DEFINE_integer('num_id1', 1, 'number of dark_net_id1')
tf.flags.DEFINE_integer('num_id2', 2, 'number of dark_net_id2')
tf.flags.DEFINE_integer('num_id3', 8, 'number of dark_net_id3')
tf.flags.DEFINE_integer('num_id4', 8, 'number of dark_net_id4')
tf.flags.DEFINE_integer('num_id5', 4, 'number of dark_net_id5')
tf.flags.DEFINE_integer('max_num_boxes_per_image', 20, 'max number of boxes per image')
tf.flags.DEFINE_string('X', 'data/tfrecords/train.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/image.tfrecords')
tf.flags.DEFINE_string('labels_file', 'data/label/keypoint_train_annotations_20170909.json',
                       'labels file for training, default: data/label/keypoint_train_annotations_20170911.json')
tf.flags.DEFINE_integer('num_anchors', 3, 'the number of anchors, default: 9')
tf.flags.DEFINE_integer('bbox_dims', 4, 'the dimensions of bbox')
tf.flags.DEFINE_integer('num_classes', 1, 'the number of classes')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_float('ignore_threshold', 0.5, 'Impacts how the loss is calculated. '
                                               'Must be between zero and one, and the default is set to 0.5.')
