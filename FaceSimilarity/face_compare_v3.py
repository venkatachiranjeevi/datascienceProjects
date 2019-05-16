import argparse
import json
import sys
import tensorflow as tf
import numpy as np

from scipy import misc
from FaceSimilarity import facenet
from FaceSimilarity.align import detect_face


sess = tf.Session()
facenet.load_model("20180402-114759.pb")
graph = tf.get_default_graph()
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


def compare_faces(image_files, image_size=160, margin=44, gpu_memory_fraction=1.0, threshold=0.88):

    images, images_names = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)


    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    nrof_images = len(images)
    base_image_index = images_names.index("base_image")

    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, images_names[i]))
    print('')

    res={}

    for i in range(nrof_images):
        if i!= base_image_index:
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[base_image_index,:], emb[i,:]))))
            if dist > float(threshold):
                out = {"distance": str(dist) ,'SAME' : "No"}
            else:
                out = {"distance": str(dist) ,'SAME' : "Yes"}

            res[images_names[i]] = out
    return res


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_list = []
    image_names=[]
    for image_name,image_data in image_paths.items():
        img_size = np.asarray(image_data.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(image_data, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          #image_paths.remove(image)
          print("can't detect face, remove ", image_name)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = image_data[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        image_names.append(image_name)
    images = np.stack(img_list)
    return images,image_names


def detect_face_in_image(image, gpu_memory_fraction=1.0):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)