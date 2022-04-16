import os

import numpy as np
import tensorflow.compat.v1 as tf
import PIL.Image as Image

from object_detection.utils import label_map_util, visualization_utils

tf.disable_v2_behavior()

PATH_TO_LABELS = os.path.join('mscoco_label_map.pbtxt')
NUM_CLASSES = 90
TEST_IMAGE_PATHS = [os.path.join('test_images', 'image{}.jpg').format(i) for i in range(1, 3)]
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
PATH_TO_CKPT = os.path.join('.', MODEL_NAME, 'frozen_inference_graph.pb')


def main(argv=None):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=NUM_CLASSES,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    jpeg = tf.placeholder(tf.string)
    decode = tf.image.decode_jpeg(jpeg)

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        for image_path in TEST_IMAGE_PATHS:
            with tf.gfile.GFile(image_path, 'rb') as f:
                data = f.read()
            image = sess.run(decode, feed_dict={jpeg: data})

            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes = graph.get_tensor_by_name('detection_boxes:0')
            scores = graph.get_tensor_by_name('detection_scores:0')
            classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')

            boxes, scores, classes, num_detections = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(image, axis=0)}
            )
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True
            )
            Image.fromarray(image).show()


if __name__ == "__main__":
    tf.app.run()
