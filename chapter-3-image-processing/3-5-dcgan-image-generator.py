import os
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def batch(batch_size=32):
    paths = []
    topdir = os.path.join('images_background_small2', 'Japanese_(katakana)')
    for dirpath, _, files in os.walk(topdir, followlinks=True):
        paths += [os.path.join(dirpath, file) for file in files]

    queue = tf.train.slice_input_producer([paths])
    png = tf.read_file(queue[0])
    image = tf.image.decode_png(png, channels=1)
    image = tf.image.resize_images(image, [32, 32])
    image = tf.subtract(tf.divide(image, 127.5), 1.0)
    return tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity= len(paths) + 3 * batch_size,
        min_after_dequeue=len(paths)
    )


def generator(inputs, batch_size):
    with tf.variable_scope('g'):
        with tf.variable_scope('reshape'):
            weight0 = tf.get_variable(
                'w', [10, 4 * 4 * 36],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias0 = tf.get_variable(
                'b', [4 * 4 * 36],
                initializer=tf.zeros_initializer()
            )
            fc0 = tf.add(tf.matmul(inputs, weight0), bias0)
            out0 = tf.reshape(fc0, [batch_size, 4, 4, 36])

        with tf.variable_scope('conv_transpose1'):
            weight1 = tf.get_variable(
                'w', [5, 5, 24, 36],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias1 = tf.get_variable(
                'b', [24],
                initializer=tf.zeros_initializer()
            )
            deconv1 = tf.nn.conv2d_transpose(out0, weight1, [batch_size, 8, 8, 24], [1, 2, 2, 1])
            out1 = tf.add(deconv1, bias1)

        with tf.variable_scope('conv_transpose2'):
            weight2 = tf.get_variable(
                'w', [5, 5, 16, 24],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias2 = tf.get_variable(
                'b', [16],
                initializer=tf.zeros_initializer()
            )
            deconv2 = tf.nn.conv2d_transpose(out1, weight2, [batch_size, 16, 16, 16], [1, 2, 2, 1])
            out2 = tf.add(deconv2, bias2)

        with tf.variable_scope('conv_transpose3'):
            weight3 = tf.get_variable(
                'w', [5, 5, 1, 16],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias3 = tf.get_variable(
                'b', [1],
                initializer=tf.zeros_initializer()
            )
            deconv3 = tf.nn.conv2d_transpose(out2, weight3, [batch_size, 32, 32, 1], [1, 2, 2, 1])
            out3 = tf.add(deconv3, bias3)

    return tf.nn.tanh(out3)


def discriminator(inputs, reuse=False):
    with tf.variable_scope('d'):
        with tf.variable_scope('conv1', reuse=reuse):
            weight1 = tf.get_variable(
                'w', [5, 5, 1, 16],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias1 = tf.get_variable(
                'b', [16],
                initializer=tf.zeros_initializer
            )
            conv1 = tf.nn.conv2d(inputs, weight1, [1, 2, 2, 1], 'SAME')
            out1 = tf.nn.relu(tf.add(conv1, bias1))

        with tf.variable_scope('conv2', reuse=reuse):
            weight2 = tf.get_variable(
                'w', [5, 5, 16, 24],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias2 = tf.get_variable(
                'b', [24],
                initializer=tf.zeros_initializer
            )
            conv2 = tf.nn.conv2d(out1, weight2, [1, 2, 2, 1], 'SAME')
            out2 = tf.nn.relu(tf.add(conv2, bias2))

        with tf.variable_scope('conv3', reuse=reuse):
            weight3 = tf.get_variable(
                'w', [5, 5, 24, 36],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias3 = tf.get_variable(
                'b', [36],
                initializer=tf.zeros_initializer
            )
            conv3 = tf.nn.conv2d(out2, weight3, [1, 2, 2, 1], 'SAME')
            out3 = tf.nn.relu(tf.add(conv3, bias3))

        reshape = tf.reshape(out3, [out3.get_shape()[0].value, -1])

        with tf.variable_scope('fully_connect', reuse=reuse):
            weight4 = tf.get_variable(
                'w', [4 * 4 * 36, 2],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias4 = tf.get_variable(
                'b', [2],
                initializer=tf.zeros_initializer()
            )
            out4 = tf.add(tf.matmul(reshape, weight4), bias4)

    return out4


def main():
    batch_size = 32
    inputs = tf.random_normal([batch_size, 10])
    real = batch(batch_size)
    fake = generator(inputs, batch_size)
    real_logits = discriminator(real)
    fake_logits = discriminator(fake, reuse=True)
    g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.ones([batch_size], dtype=tf.int64),
        logits=fake_logits
    ))
    d_loss = tf.reduce_sum([
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([batch_size], dtype=tf.int64),
            logits=fake_logits
        )),
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([batch_size], dtype=tf.int64),
            logits=real_logits
        ))
    ])

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
    g_train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss, var_list=g_vars)
    d_train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=d_vars)

    generated = tf.concat(tf.split(fake, batch_size)[:8], 2)
    generated = tf.divide(tf.add(tf.squeeze(generated, axis=0), 1.0), 2.0)
    generated = tf.image.convert_image_dtype(generated, tf.uint8)
    output_img = tf.image.encode_png(generated)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())

        for i in range(50001):
            _, _, g_loss_value, d_loss_value = sess.run([g_train_op, d_train_op, g_loss, d_loss])
            print('step {:5d}: g = {:.4f}, d = {:.4f}'.format(i+1, g_loss_value, d_loss_value))

            if i % 100 == 0:
                img = sess.run(output_img)
                with open(os.path.join(os.path.dirname(__file__), '{:05d}.png'.format(i)), 'wb') as f:
                    f.write(img)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
