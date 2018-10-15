import tensorflow as tf
import layer


class Darknet:
    def __init__(self, name, is_training,
                 num_id1=1, num_id2=2, num_id3=8, num_id4=8, num_id5=4,
                 norm='batch'):
        self.name = name
        self.is_training = is_training
        self.num_id1 = num_id1
        self.num_id2 = num_id2
        self.num_id3 = num_id3
        self.num_id4 = num_id4
        self.num_id5 = num_id5
        self.norm = norm
        self.reuse = False

    def __call__(self, ipt):
        with tf.variable_scope(self.name):
            c3s1k32 = layer.c3s1k32(ipt, reuse=self.reuse, is_training=self.is_training, norm=self.norm)
            c3s2k64 = layer.c3s2k64(c3s1k32, reuse=self.reuse, is_training=self.is_training, norm=self.norm)

            dark_net_conv_1 = []
            for i in range(self.num_id1):
                dark_net_conv_1.append(layer.dark_net_conv_1(dark_net_conv_1[-1] if i
                                                             else c3s2k64, i,
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k128 = layer.c3s2k128(dark_net_conv_1[-1], reuse=self.reuse, is_training=self.is_training,
                                      norm=self.norm)

            dark_net_conv_2 = []
            for i in range(self.num_id2):
                dark_net_conv_2.append(layer.dark_net_conv_2(dark_net_conv_2[-1] if i
                                                             else  c3s2k128, i,
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k256 = layer.c3s2k256(dark_net_conv_2[-1], reuse=self.reuse, is_training=self.is_training,
                                      norm=self.norm)

            dark_net_conv_3 = []
            for i in range(self.num_id3):
                dark_net_conv_3.append(layer.dark_net_conv_3(dark_net_conv_3[-1] if i
                                                             else c3s2k256, i,
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k512 = layer.c3s2k512(dark_net_conv_3[-1], reuse=self.reuse, is_training=self.is_training,
                                      norm=self.norm)

            dark_net_route_1 = dark_net_conv_3[-1]

            dark_net_conv_4 = []
            for i in range(self.num_id5):
                dark_net_conv_4.append(layer.dark_net_conv_4(dark_net_conv_4[-1] if i
                                                             else c3s2k512, i,
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k1024 = layer.c3s2k1024(dark_net_conv_4[-1], reuse=self.reuse, is_training=self.is_training,
                                        norm=self.norm)

            dark_net_route_2 = dark_net_conv_4[-1]

            dark_net_conv_5 = []
            for i in range(self.num_id5):
                dark_net_conv_5.append(layer.dark_net_conv_5(dark_net_conv_5[-1] if i
                                                             else c3s2k1024, i,
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            dark_out = dark_net_conv_5[-1]

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return dark_out, dark_net_route_1, dark_net_route_2


