import tensorflow as tf
import ops
import numpy as np


def retina7(ipt, name='retina7', reuse=False, is_training=True):
    with tf.variable_scope(name):
        ipt = tf.nn.relu(ipt)
        return ops.conv2d(ipt, 256, 3, 1, 2, norm=None, activation=None,
                          reuse=reuse, is_training=is_training, name='c1s2k256', use_bias=True,
                          kernel_initializer=None)


def retina6(ipt, name='retina6', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 256, 3, 1, 2, norm=None, activation=None,
                          reuse=reuse, is_training=is_training, name='c1s2256', use_bias=True,
                          kernel_initializer=None)


def retina5(ipt, name='retina5', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                          reuse=reuse, is_training=is_training, name='c1s1256', use_bias=True,
                          kernel_initializer=None)


def retina4(ipt1, ipt2, name='retina4', reuse=False, is_training=True,
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)


def retina3(ipt1, ipt2, name='retina3', reuse=False, is_training=True,
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    shape = ipt2.get_shape().as_list()
    with tf.variable_scope(name):
        c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c1s1k256', use_bias=True,
                              kernel_initializer=None)
        upsample = tf.image.resize_images(ipt2, [shape[1] * 2, shape[2] * 2],
                                          method=resize_method)
        return tf.add(c1s1k256, upsample)


def retina_class_subnet(ipt, num_anchors, name='retina_class_subnet', reuse=False, is_training=True):
    with tf.variable_scope(name):
        c3s1k256 = ops.conv2d(ipt, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_1', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        c3s1k256 = ops.conv2d(c3s1k256, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_2', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        c3s1k256 = ops.conv2d(c3s1k256, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_3', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        c3s1k256 = ops.conv2d(c3s1k256, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_4', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        retina_class_subnet_output = ops.conv2d(c3s1k256, num_anchors, 3, 1, 1, norm=None, activation=tf.nn.relu,
                                                reuse=reuse, is_training=is_training, name='output', use_bias=True,
                                                kernel_initializer=None, weights_std=0.01, bias_init=-np.log(99))
        return retina_class_subnet_output


def retina_bboxreg_subnet(ipt, num_anchors, name='retina_regbbox_subnet', reuse=False, is_training=True):
    with tf.variable_scope(name):
        c3s1k256 = ops.conv2d(ipt, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_1', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        c3s1k256 = ops.conv2d(c3s1k256, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_2', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        c3s1k256 = ops.conv2d(c3s1k256, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_3', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        c3s1k256 = ops.conv2d(c3s1k256, 256, 3, 1, 1, norm=None, activation=tf.nn.relu,
                              reuse=reuse, is_training=is_training, name='c3s1k256_4', use_bias=True,
                              kernel_initializer=None, weights_std=0.01)
        retina_bboxreg_subnet_output = ops.conv2d(c3s1k256, num_anchors * 4, 3, 1, 1, norm=None, activation=tf.nn.relu,
                                                  reuse=reuse, is_training=is_training, name='output', use_bias=True,
                                                  kernel_initializer=None, weights_std=0.01, bias_init=-np.log(99))
        return retina_bboxreg_subnet_output
