import numpy as np
import tensorflow as tf
import random
import os
from multiprocessing import Pool
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
    tf.compat.v1.disable_eager_execution()
    rgb_saver = tf.train.import_meta_graph('./fine_tune_model/model.ckpt.meta')
    with tf.Session() as sess:
        # Restore the parameters of all the exsits layers
        rgb_saver.restore(sess, "./fine_tune_model/model.ckpt")
        loss = sess.graph.get_tensor_by_name('loss:0')
        rgb_input = sess.graph.get_tensor_by_name('rgb_input:0')
        Y_ = sess.graph.get_tensor_by_name('Y_:0')

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=3).minimize(loss)

        iterator = 0
        epoch_num = 5
        batchsize = 32
        steps = int(len(global_train_dataset)/batchsize)

        for epoch in range(epoch_num):
            for step in range(steps):
                feed_dict = {}
                feed_dict[rgb_input], feed_dict[Y_], iterator = get_one_batch_data_multiprocess_v2("training", iterator, batchsize)
                #feed_dict[Y_2] = feed_dict[Y_] # All labels have the same value in depth direction
                if step % 5 == 0:
                    step_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                    print("step:{}, loss:{}".format(step, step_loss))
                else:
                    sess.run(optimizer, feed_dict=feed_dict)

        if os.path.isdir("./fine_tune_model_tf2") is False:
            os.makedirs("./fine_tune_model_tf2")
            os.makedirs("./fine_tune_model_tf2/pb")
        saver = tf.train.Saver()
        saver.save(sess, "./fine_tune_model_tf2/model.ckpt")

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Ys'])
        with tf.gfile.FastGFile("./fine_tune_model_tf2/pb/model.pb", mode='wb') as f:
            f.write(constant_graph.SerializeToString())

def inference():
    vid_key = get_key_vid(train_split)
    with tf.Session() as sess:
        path = './fine_tune_model_tf2/pb/model.pb'
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
