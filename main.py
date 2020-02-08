from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sonnet as snt
import i3d
import random
import os
from multiprocessing import Pool
import time

from ucf101_dataset import make_dataset, get_key_vid
from ucf101_dataset import load_rgb_frames
import videotransforms
from tensorflow.python.framework import graph_util

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 64

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
}

train_split = "./ucf101/jsons/ucf101.sp1.customer.json"
dataset_root = "/home/pytorch/DataSet/ucf_rgb"
train_depth_mode = "random"
category = 101
depth_size = _SAMPLE_VIDEO_FRAMES
depth_stride = 10

def make_datasets(split):
    """split = training or testing"""
    return make_dataset(train_split, split, dataset_root, category, depth_size)

global_train_dataset = make_datasets("training")
global_test_dataset = make_datasets("testing")
global_vcCorp = videotransforms.CenterCrop(224)
global_vcRandomCorp = videotransforms.RandomCrop(224)
global_vcRandomFlip = videotransforms.RandomHorizontalFlip()
def get_one_batch_data_single_read(index):
    vid, labels, dur, sample_frame = global_train_dataset[index]
    start_index = random.randint(0, sample_frame - depth_size)
    imgs = load_rgb_frames(dataset_root, vid, start_index + 1, depth_size)
    imgs = global_vcRandomFlip(global_vcRandomCorp(imgs))
    return imgs

def get_one_batch_labels_single_read(index):
    _, labels, _, _ = global_train_dataset[index]
    labels = labels[:, 0:64] #all 64 depth have the same value
    return labels

def get_one_batch_testdata_single_read(index):
    vid, labels, dur, sample_frame = global_test_dataset[index]
    start_index = random.randint(0, sample_frame - depth_size)
    imgs = load_rgb_frames(dataset_root, vid, start_index + 1, depth_size)
    imgs = global_vcCorp(imgs)
    return imgs

def get_one_batch_testlabels_single_read(index):
    _, labels, _, _ = global_test_dataset[index]
    labels = labels[:, 0:64] #all 64 depth have the same value
    return labels

def get_one_batch_data_multiprocess_v2(phase, iterator, batchsize):
    if phase == "training":
        dataset = global_train_dataset
    else:
        dataset = global_test_dataset
    if iterator + batchsize > len(dataset):
        iterator = 0
    pool = Pool(processes=batchsize)

    chunks = [iterator+i for i in range(batchsize)]
    if phase == "training":
        imgs_batch = pool.map(get_one_batch_data_single_read, chunks)
        labels_batch = pool.map(get_one_batch_labels_single_read, chunks)
    else:
        imgs_batch = pool.map(get_one_batch_testdata_single_read, chunks)
        labels_batch = pool.map(get_one_batch_testlabels_single_read, chunks)
    pool.close()
    pool.join()
    return np.asarray(imgs_batch), np.asarray(labels_batch), iterator+batchsize

def get_one_batch_data(phase, iterator, batchsize):
    imgs_batch = None
    labels_batch = None
    if phase == "training":
        dataset = global_train_dataset
    else:
        dataset = global_test_dataset
    for _ in range(batchsize):
        if iterator >= len(dataset):
            iterator = 0
        vid, labels, dur, sample_frame = dataset[iterator]
        start_index = random.randint(0, sample_frame - depth_size)
        imgs = load_rgb_frames(dataset_root, vid, start_index + 1, depth_size)
        #vcCorp = videotransforms.CenterCrop(224)
        if phase == "training":
            imgs = global_vcRandomFlip(global_vcRandomCorp(imgs))
        else:
            imgs = global_vcCorp(imgs)
        imgs = np.reshape(imgs, [1, depth_size, 224, 224, 3])
        if imgs_batch is None:
            imgs_batch = imgs
        else:
            imgs_batch = np.concatenate((imgs_batch, imgs))
        labels = labels[:, 0:64]
        labels = np.reshape(labels, [1, category, depth_size])
        if labels_batch is None:
            labels_batch = labels
        else:
            labels_batch = np.concatenate((labels_batch, labels))
        # print(imgs_batch.shape)
        iterator += 1
    return imgs_batch, labels_batch, iterator

def test_read():
    iterator = 0
    batchsize = 32
    start = time.time()
    imgs_batch, labels_batch, iterator = get_one_batch_data_multiprocess_v2("training", iterator, batchsize)
    end = time.time()
    print(end-start)
    print(imgs_batch.shape)
    print(labels_batch.shape)

def get_one_image(vid, dataset=global_train_dataset):
    imgs = None
    for item in dataset:
        #For testing
        if item[0] == vid:
            vid, labels, dur, sample_frame = item
            start_index = 0
            imgs = load_rgb_frames(dataset_root, vid, start_index + 1, depth_size)
            # Reshape images to 224
            imgs = global_vcCorp(imgs)
            imgs = np.reshape(imgs, [1,depth_size,224,224,3])
            labels = labels[:, 0:64]
            labels = np.reshape(labels, [1,category,depth_size])[:, :, 0]
    return imgs, labels

def train():
    is_training = False
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3), name="rgb_input")
    Y_ = tf.placeholder(tf.float32, [None, 101, depth_size], name="Y_")
    Y_2 = tf.placeholder(tf.float32, [None, 101], name="Y_2")
    with tf.variable_scope('RGB'):
        # Method1
        # First construct the graph with 400 calss
        # Then replace 400 with 101 othwise you cann't restore the models
        # Method2
        # move the endpoint forward
        # Then create the rgb_logits manually
        rgb_model = i3d.InceptionI3d(
            num_classes=400, spatial_squeeze=True, final_endpoint='Mixed_5c')
        rgb_Mixed_5c, _ = rgb_model(
            rgb_input, is_training=is_training, dropout_keep_prob=1.0)

    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    net = tf.nn.avg_pool3d(rgb_Mixed_5c, ksize=[1, 2, 7, 7, 1],
                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    net = tf.nn.dropout(net, 1.0)

    logits_u3d = i3d.Unit3D(output_channels=101,
                      kernel_shape=[1, 1, 1],
                      activation_fn=None,
                      use_batch_norm=False,
                      use_bias=True,
                      name='Conv3d_0c_1x1')(net, is_training=is_training)

    logits = tf.squeeze(logits_u3d, [2, 3], name='SpatialSqueeze')

    #transpose_logits = tf.transpose(logits, perm=[0, 2, 1])
    transpose_logits = tf.transpose(logits, perm=[2, 1, 0])
    resize_logits = tf.image.resize(transpose_logits, [101, 64])
    transpose_logits2 = tf.transpose(resize_logits, perm=[2, 0, 1])

    model_logits = tf.reduce_mean(logits, axis=1)
    model_predictions = tf.nn.softmax(model_logits, name="Ys")

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=transpose_logits2, labels=Y_))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=transpose_logits2, labels=Y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=3).minimize(loss)

    # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_predictions, labels=Y_2))
    loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_predictions, labels=Y_2))
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=3).minimize(loss2)

    with tf.Session() as sess:
        # Init the new add layer's parameteters
        sess.run(tf.global_variables_initializer())
        # Restore the parameters of all the exsits layers
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])

        iterator = 0
        epoch_num = 5
        batchsize = 32
        steps = int(len(global_train_dataset)/batchsize)

        for epoch in range(epoch_num):
            for step in range(steps):
                feed_dict = {}
                feed_dict[rgb_input], feed_dict[Y_], iterator = get_one_batch_data_multiprocess_v2("training", iterator, batchsize)
                feed_dict[Y_2] = feed_dict[Y_][:, :, 0]  # All labels have the same value in depth direction

                # optimizer, training 5 epoch, lr=3, inference accuracy 3386/3776=0.896
                # optimizer2, training 5 epoch, lr=3, inference accuracy
                if step % 5 == 0:
                    step_loss, _ = sess.run([loss2, optimizer2], feed_dict=feed_dict)
                    print("step:{}, loss:{}".format(step, step_loss))
                else:
                    sess.run(optimizer2, feed_dict=feed_dict)

        if os.path.isdir("./fine_tune_model") is False:
            os.makedirs("./fine_tune_model")
            os.makedirs("./fine_tune_model/pb")
        saver = tf.train.Saver()
        saver.save(sess, "./fine_tune_model/model")

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Ys'])
        with tf.gfile.FastGFile("./fine_tune_model/pb/model.pb", mode='wb') as f:
            f.write(constant_graph.SerializeToString())

def inference():
    vid_key = get_key_vid(train_split)
    with tf.Session() as sess:
        path = './fine_tune_model/pb/model.pb'
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # for node in graph_def.node:
            #     print("node name is: {} \t node op is: {}".format(node.name, node.op))
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            Ys = sess.graph.get_tensor_by_name('Ys:0')
            X = sess.graph.get_tensor_by_name('rgb_input:0')

            test_one_image = False
            if test_one_image:
                total = 0
                correct = 0
                for item in global_test_dataset:
                    vid = item[0]
                    #vid = "Biking/v_Biking_g01_c03"
                    imgs, labels = get_one_image(vid, global_test_dataset)

                    res = sess.run(Ys, feed_dict={X:imgs})
                    print("------------------")
                    print(np.argmax(res))
                    print(vid_key[np.argmax(res)])
                    print(np.argmax(labels))
                    print(vid_key[np.argmax(labels)])
                    total += 1
                    if np.argmax(res) == np.argmax(labels):
                        correct += 1
                print("Total:{}, correct:{}".format(total, correct))
            else:
                iterator = 0
                batchsize = 32
                steps = int(len(global_test_dataset) / batchsize)
                total = 0
                correct = 0
                print("total steps: {}, images: {}".format(steps, steps*batchsize))
                for step in range(steps):
                    imgs, labels, iterator = get_one_batch_data_multiprocess_v2("testing", iterator, batchsize)
                    labels = labels[:, :, 0] # All labels have the same value in depth direction
                    res = sess.run(Ys, feed_dict={X: imgs})
                    for bs in range(batchsize):
                        total += 1
                        # print("------------------")
                        # print(np.argmax(res[bs]))
                        # print(vid_key[np.argmax(res[bs])])
                        # print(np.argmax(labels[bs]))
                        # print(vid_key[np.argmax(labels[bs])])
                        if np.argmax(res[bs]) == np.argmax(labels[bs]):
                            correct += 1
                    print("Total:{}, correct:{}".format(total, correct))

if __name__ == "__main__":
    #test_read()
    train()
    #inference()
    #test_400class()
    # vid_key = get_key_vid(train_split)
    # for key in vid_key:
    #     print("{}:{}".format(key, vid_key[key]))
