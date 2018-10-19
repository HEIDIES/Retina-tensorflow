import tensorflow as tf
from model import DETECTERSUBNET
from reader import Reader
from datetime import datetime
import os
import logging
import json_convert
import utils
import labels_generator
import hyper_parameters
import numpy as np
train_mode = True


def train_yolo_v3():
    if hyper_parameters.FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + hyper_parameters.FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    labels = json_convert.load_label(hyper_parameters.FLAGS.labels_file)
    anchors = utils.get_anchors(hyper_parameters.FLAGS.anchors_path)
    anchors = np.array(anchors, dtype=np.float32)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        yolo_v3 = DETECTERSUBNET('yolo_v3',
                                 image_size=hyper_parameters.FLAGS.image_size,
                                 anchors=anchors,
                                 batch_size=hyper_parameters.FLAGS.batch_size,
                                 num_anchors=hyper_parameters.FLAGS.num_anchors,
                                 learning_rate=hyper_parameters.FLAGS.learning_rate,
                                 num_classes=hyper_parameters.FLAGS.num_classes,
                                 num_id1=hyper_parameters.FLAGS.num_id1,
                                 num_id2=hyper_parameters.FLAGS.num_id2,
                                 num_id3=hyper_parameters.FLAGS.num_id3,
                                 num_id4=hyper_parameters.FLAGS.num_id4,
                                 num_id5=hyper_parameters.FLAGS.num_id5,
                                 norm=hyper_parameters.FLAGS.norm,
                                 threshold=hyper_parameters.FLAGS.threshold,
                                 max_num_boxes_per_image=hyper_parameters.FLAGS.max_num_boxes_per_image
                                 )

        loss = yolo_v3.model()
        last_layer_optimizer, yolo_optimizer = yolo_v3.yolo_v3_optimizer(loss)

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
                if hyper_parameters.FLAGS.load_model is not None:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    restore = tf.train.import_meta_graph(meta_graph_path)
                    restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                    step = int(meta_graph_path.split("-")[2].split(".")[0])
                else:
                    sess.run(tf.global_variables_initializer())
                    step = 0

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    while not coord.should_stop() and step < 200000:

                        images, img_ids, img_heights, img_widths = sess.run([x, image_ids,
                                                                             image_heights, image_widths])
                        heatmaps = labels_generator.get_detector_heatmap(img_ids, img_heights, img_widths, labels)

                        optimizer = yolo_optimizer

                        if step < 10000 // 3:
                            optimizer = last_layer_optimizer
                        _, loss_val, summary = sess.run([optimizer, loss, summary_op],
                                                        feed_dict={yolo_v3.X: images,
                                                                   yolo_v3.Y_true_data: heatmaps[0],
                                                                   yolo_v3.Y_true_boxes: heatmaps[1]}
                                                        )

                        train_writer.add_summary(summary, step)
                        train_writer.flush()
                        if (step + 1) % 100 == 0:
                            logging.info('-----------Step %d:-------------' % (step + 1))
                            logging.info('  loss   : {}'.format(loss_val))

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
    train_yolo_v3()
    vars(unused_argv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
