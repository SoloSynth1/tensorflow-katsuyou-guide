import os
import re
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def get_data():
    train, test = [], []
    topdir = os.path.join('images_background_small2', 'Japanese_(katakana)')
    regexp = re.compile(r'character(\d+)')
    for dirpath, _, files in os.walk(topdir, followlinks=True):
        match = regexp.search(dirpath)
        if match is None:
            continue
        label = int(match.group(1)) - 1
        data = [(label, os.path.join(dirpath, file)) for file in files]

        random.shuffle(data)
        num_train = int(len(data) * 0.8)
        train += data[:num_train]
        test += data[num_train:]
    return train, test


def get_batch(data_list, shuffle=False):
    list_len = len(data_list)
    labels, paths = zip(*data_list)
    queue = tf.train.slice_input_producer([labels, paths])
    label = queue[0]
    png = tf.read_file(queue[1])
    image = tf.image.decode_png(png, channels=1)
    image.set_shape([105, 105, 1])
    image = tf.image.per_image_standardization(image)
    if shuffle:
        return tf.train.shuffle_batch(
            [image, label],
            batch_size=32,
            capacity=list_len * 2 + 3 * 32,
            min_after_dequeue=list_len * 2
        )
    else:
        return tf.train.batch([image, label], list_len)


def inference(inputs, reuse=False):
    """
    :param inputs: [batch_size, height, width, channels] の Tensor
    :param reuse: 変数を再利用するか否か
    :return: 推論結果の [batch_size, 47] の Tensor
    """
    with tf.variable_scope('conv1', reuse=reuse):
        weight1 = tf.get_variable(
            'w', [3, 3, 1, 16],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias1 = tf.get_variable(
            'b', shape=[16],
            initializer=tf.zeros_initializer()
        )
        conv1 = tf.nn.conv2d(inputs, weight1, [1, 2, 2, 1], 'VALID')
        out1 = tf.nn.relu(tf.add(conv1, bias1))

    pool1 = tf.nn.max_pool(out1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    with tf.variable_scope('conv2', reuse=reuse):
        weight2 = tf.get_variable(
            'w', [3, 3, 16, 24],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias2 = tf.get_variable(
            'b', shape=[24],
            initializer=tf.zeros_initializer()
        )
        conv2 = tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], 'VALID')
        out2 = tf.nn.relu(tf.add(conv2, bias2))

    pool2 = tf.nn.max_pool(out2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    with tf.variable_scope('conv3', reuse=reuse):
        weight3 = tf.get_variable(
            'w', [3, 3, 24, 36],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias3 = tf.get_variable(
            'b', shape=[36],
            initializer=tf.zeros_initializer()
        )
        conv3 = tf.nn.conv2d(pool2, weight3, [1, 1, 1, 1], 'VALID')
        out3 = tf.nn.relu(tf.add(conv3, bias3))

    pool3 = tf.nn.max_pool(out3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    reshape = tf.reshape(pool3, [pool3.get_shape()[0].value, -1])

    with tf.variable_scope('fully_connect', reuse=reuse):
        weight4 = tf.get_variable(
            'w', [5 * 5 * 36, 47],
            initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
        bias4 = tf.get_variable(
            'b', shape=[47],
            initializer=tf.zeros_initializer()
        )
        out4 = tf.add(tf.matmul(reshape, weight4), bias4)

    return out4


def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )
    return tf.reduce_mean(cross_entropy)


def training(loss):
    optimizer = tf.train.AdamOptimizer(0.001)
    return optimizer.minimize(loss)


def main():
    train, test = get_data()
    train_images, train_labels = get_batch(train, shuffle=True)
    train_logits = inference(train_images)
    losses = loss(train_labels, train_logits)
    train_op = training(losses)

    test_images, test_labels = get_batch(test, shuffle=False)
    test_logits = inference(test_images, reuse=True)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), tf.to_int64(test_labels))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())
        for i in range(300):
            _, loss_value, accuracy_value = sess.run([train_op, losses, accuracy])
            print('step {:3d}: {:5f} ({:3f})'.format(i+1, loss_value, accuracy_value * 100.0))

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
