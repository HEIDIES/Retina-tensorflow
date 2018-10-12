import tensorflow as tf
from model import DETECTERSUBNET
from reader import Reader
from datetime import datetime
import os
import logging
import json_convert
# import numpy as np
import labels_generator
import hyper_parameters
import anchors_generator
train_mode = True


def train_retina_subnet():
    if hyper_parameters.FLAGS.load_model_retina is not None:
        checkpoints_dir = "checkpoints/" + hyper_parameters.FLAGS.load_model_retina.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    labels = json_convert.load_label(hyper_parameters.FLAGS.labels_file)
    anchors = anchors_generator.generate_anchors()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        retina = DETECTERSUBNET('retina_subnet',
                                hyper_parameters.FLAGS.image_size_retina,
                                anchors,
                                batch_size=hyper_parameters.FLAGS.batch_size,
                                num_anchors=hyper_parameters.FLAGS.num_anchors,
                                learning_rate=hyper_parameters.FLAGS.learning_rate_retina,
                                gamma=hyper_parameters.FLAGS.gamma,
                                alpha=hyper_parameters.FLAGS.alpha
                                )

        res50_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        saver_res50 = tf.train.Saver(res50_var_list)

        reg_loss, confs_loss = retina.model()
        retina_subnet_output = retina.out()
        optimizer = retina.retina_subnet_optimizer(reg_loss, confs_loss)

        saver = tf.train.Saver()

        reader = Reader(hyper_parameters.FLAGS.X, batch_size=hyper_parameters.FLAGS.batch_size,
                        image_size=hyper_parameters.FLAGS.image_size)
        x, image_ids, image_heights, image_widths = reader.feed()

        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        if train_mode is True:
            with graph.as_default():
                summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(checkpoints_dir, graph)

            with tf.Session(graph=graph, config=config) as sess:
                if hyper_parameters.FLAGS.load_model_retina is not None:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    restore = tf.train.import_meta_graph(meta_graph_path)
                    restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                    step = int(meta_graph_path.split("-")[2].split(".")[0])
                else:
                    sess.run(tf.global_variables_initializer())
                    saver_res50.restore(sess, hyper_parameters.FLAGS.pretrained_model_checkpoints)
                    step = 0

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    while not coord.should_stop() and step < 100000:

                        images, img_ids, img_widths, img_heights = sess.run([x, image_ids,
                                                                             image_heights, image_widths])
                        heatmaps = labels_generator.get_detector_heatmap(img_ids, img_heights, img_widths, labels)
                        _, reg_loss_val, confs_loss_val, summary = sess.run([optimizer,
                                                                            reg_loss, confs_loss,
                                                                            summary_op],
                                                                            feed_dict={retina.X: images,
                                                                                       retina.Y_3: heatmaps[0],
                                                                                       retina.Y_4: heatmaps[1],
                                                                                       retina.Y_5: heatmaps[2],
                                                                                       retina.Y_6: heatmaps[3],
                                                                                       retina.Y_7: heatmaps[4]})
                        train_writer.add_summary(summary, step)
                        train_writer.flush()
                        if (step + 1) % 100 == 0:
                            logging.info('-----------Step %d:-------------' % (step + 1))
                            logging.info('  reg_loss   : {}'.format(reg_loss_val))
                            logging.info('  confs_loss   : {}'.format(confs_loss_val))

                        if step % 10000 == 0:
                            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                            logging.info("Model saved in file: %s" % save_path)

                        step += 1

                except KeyboardInterrupt:
                    logging.info('Interrupted')
                    coord.request_stop()

                except Exception as e:
                    coord.request_stop(e)

                finally:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                    # When done, ask the threads to stop.
                    coord.request_stop()
                    coord.join(threads)


def main(unused_argv):
    train_retina_subnet()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
