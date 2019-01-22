import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import cv2
import time

#defining path of my training and testing dataset (local path)
ROOT= "A:"
training_imgs = os.path.join(ROOT, "/DataMining/training")
test_imgs = os.path.join(ROOT, "/DataMining/testing")

#function to load images and their classes 
def data_load(dir_path):

    dirs = [dir for dir in os.listdir(dir_path) 
                   if os.path.isdir(os.path.join(dir_path, dir))]
    classes = []
    img = []
    for d in dirs:
        label_dir = os.path.join(dir_path, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            img.append(skimage.data.imread(f))
            classes.append(int(d))
    return img, classes

#first load training data for preprocessing and training
images, classes = data_load(training_imgs)
nclasses = len(set(classes))
print("The number of unique classes are: {0}\nCount of images in all classes: {1}".format(nclasses, len(images)))



#function to show an image from each class before preprocessing
def sample_from_class(images, classes):
    tot_class = set(classes)
    plt.figure(figsize=(20, 20))
    i = 1
    for label in tot_class:
        img = images[classes.index(label)]
        plt.subplot(10, 7, i)
        plt.axis('off')
        plt.title("class {0}, Num imgs {1}".format(label, classes.count(label)))
        i += 1
        _ = plt.imshow(img)
    plt.show()

print("\n\nImages from each class before preprocessing")
sample_from_class(images, classes)

print("\nPre-processing the data")

print("\nImages after resizing")
#resize image to resolution of 64*64
resized_imgs = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images]
sample_from_class(resized_imgs, classes)


ready_classes = np.array(classes)
ready_images = np.array(resized_imgs)

def net_architecture(img_ph):    
    mymean = 0
    mydev = 0.1
    
    #convolutional layer 1
    weights_conv1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mymean, stddev = mydev), name="weights_conv1")
    bias_conv1 = tf.Variable(tf.zeros(6),name="bias_conv1")
    layer1_out   = tf.nn.conv2d(img_ph, weights_conv1, strides=[1, 1, 1, 1], padding='VALID') + bias_conv1

    #activation function
    layer1_out = tf.nn.relu(layer1_out)

    #pooling - reduction from 28*28 to 14*14
    layer1_out = tf.nn.max_pool(layer1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #convolutional layer 2
    weights_conv2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mymean, stddev = mydev),name="weights_conv2")
    bias_conv2 = tf.Variable(tf.zeros(16),name="bias_conv2")
    conv2_out   = tf.nn.conv2d(layer1_out, weights_conv2, strides=[1, 1, 1, 1], padding='VALID') + bias_conv2
    
    #activation function
    conv2_out = tf.nn.relu(conv2_out)

    #pooling - reduction from 10*10 to 5*5
    conv2_out = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #flattening
    fc0   = flatten(conv2_out)
    
    #fully connected layer 3
    weights_fullyconn = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mymean, stddev = mydev), name="weights_fullyconn")
    bias_fullyconn = tf.Variable(tf.zeros(120),name="bias_fullyconn")
    fullyconn_out   = tf.matmul(fc0, weights_fullyconn) + bias_fullyconn
    
    #activation function
    fullyconn_out    = tf.nn.relu(fullyconn_out)

    #fully connected layer 4
    fullyconn2_weights  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mymean, stddev = mydev),name="fullyconn2_weights")
    fullyconn2_bias  = tf.Variable(tf.zeros(84),name="fullyconn2_bias")
    fullyconn2_out    = tf.matmul(fullyconn_out, fullyconn2_weights) + fullyconn2_bias
    
    #activation function
    fullyconn2_out    = tf.nn.relu(fullyconn2_out)

    
    #dropout
    hidden_layer = tf.nn.dropout(fullyconn2_out, keep_prob)
    
    #layer 5 fully connected
    fullyconn3_weights  = tf.Variable(tf.truncated_normal(shape=(84, nclasses), mean = mymean, stddev = mydev),name="fullyconn3_weights")
    fullyconn3_bias  = tf.Variable(tf.zeros(nclasses),name="fullyconn3_bias")
    logits = tf.matmul(fullyconn2_out, fullyconn3_weights) + fullyconn3_bias
    
    return logits

graph = tf.Graph()
with graph.as_default():
    placeholder_img = tf.placeholder(tf.float32, [None, 32, 32, 3])
    placeholder_classes = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)

    logits = net_architecture(placeholder_img);
    classify_labels = tf.argmax(logits, 1)

    err = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=placeholder_classes))

    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(err)

    init = tf.global_variables_initializer()

session = tf.Session(graph=graph)

#initialize session
_ = session.run([init])

#train using adamoptimizer to minimize error. alpha = 0.001
start = time.time()
for i in range(201):
    _, err_value = session.run([train, err], 
                                feed_dict={placeholder_img: ready_images, placeholder_classes: ready_classes})
    if i % 10 == 0:
        print("error: ", err_value)
end = time.time()
tot = end - start
print("\nTime taken for training the model is {0}\n".format(tot))

#Select 20 images and check for correctness of classification
try_random_20 = random.sample(range(len(resized_imgs)), 20)
img_to_be_classified = [resized_imgs[i] for i in try_random_20]
sample_classes = [classes[i] for i in try_random_20]
#run classification
classify = session.run([classify_labels], 
                        feed_dict={placeholder_img: img_to_be_classified})[0]
#plot the results with image
classiedfig = plt.figure(figsize=(20, 20))
for img in range(len(img_to_be_classified)):
    actual_class = sample_classes[img]
    after_classification = classify[img]
    plt.subplot(10, 2,1+img)
    plt.axis('off')
    val="true" if actual_class == after_classification else 'false'
    plt.text(80, 20, "actual:        {0}\nclassified: {1}\ntruth: {2}".format(actual_class, after_classification,val), 
             fontsize=12)
    plt.imshow(img_to_be_classified[img])

test_images, test_classes = data_load(test_imgs)
#preprocess the test images
resized_test = [skimage.transform.resize(image, (32, 32), mode='constant')
                 for image in test_images]
sample_from_class(resized_test, test_classes)


classify = session.run([classify_labels], 
                        feed_dict={placeholder_img: resized_test})[0]


correct_matches = sum([int(actual == classified) for actual,classified in zip(test_classes, classify)])
print("\nThe correct matches were {0} out of a total of {1} test images.".format(correct_matches,len(test_classes)))
correctness = (correct_matches / len(test_classes))*100
print("The accuracy of the system is: {0} percent".format(correctness))

session.close()