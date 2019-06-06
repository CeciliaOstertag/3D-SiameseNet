import keras
from keras import Sequential, Model
from keras.layers import Conv3D, AveragePooling3D, BatchNormalization, ZeroPadding3D, Dropout, Activation, Flatten, Dense, Input, concatenate, Lambda, UpSampling3D
from keras import models
from keras import backend as K
import tensorflow as tf
import os
import sys
import gc
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import nibabel as nib
import skimage
import scipy
from scipy.ndimage import zoom
from random import shuffle
import glob
from keras.utils.training_utils import multi_gpu_model
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
###### TF RECORDS UTILITY FUNCTIONS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
def _read_from_tfrecord(example_proto):
    feature = {
        'train/original': tf.FixedLenFeature([], tf.string),
        'train/dup': tf.FixedLenFeature([], tf.string),
        'train/mask': tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_example([example_proto], features=feature)

    original_1d = tf.decode_raw(features['train/original'], tf.float64)
    dup_1d = tf.decode_raw(features['train/dup'], tf.float64)
    mask_1d = tf.decode_raw(features['train/mask'], tf.float64)

    zmax = 150
    xmax = 205
    ymax = 216
    xmax = xmax//2
    ymax = ymax//2
    zmax = zmax//2

    original_restored = tf.reshape(original_1d, [xmax, ymax, zmax],name="r1")
    dup_restored = tf.reshape(dup_1d, [xmax, ymax, zmax],name="r2")
    mask_restored = tf.reshape(mask_1d, [xmax, ymax, zmax],name="r3")
    
    return original_restored, dup_restored, mask_restored
    
###### 

def plotExampleImage(image,title):
	fig = plt.figure(figsize=(10,2))
	plt.title(title)
	cols = 3
	rows = 1
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	proj1 = np.mean(volume, axis=1)
	proj2 = np.mean(volume, axis=2)
	ax1 = fig.add_subplot(rows, cols, 1)
	ax1.title.set_text("axis 0")
	plt.imshow(proj0,cmap="gray") 
	ax2 = fig.add_subplot(rows, cols, 2)
	ax2.title.set_text("axis 1")
	plt.imshow(proj1,cmap="gray")
	ax3 = fig.add_subplot(rows, cols, 3)
	ax3.title.set_text("axis 2")
	plt.imshow(proj2,cmap="gray")
	
def saveExampleImage(image,title):
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	cv2.imwrite(title+".jpg",proj0)


def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)
	
def layer_to_visualize(layer,img_to_visualize):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.asarray(convolutions)
    print("Shape ",convolutions.shape)
    for i in range(min(10,convolutions.shape[2])): #convolutions.shape[2]
    	conv = convolutions[0,0,i,:,:,:]
    	conv = np.squeeze(conv)
    	plotExampleImage(conv,"conv "+str(i))
    plt.show() 
    
   
    
class DataGenerator(keras.utils.Sequence):
	def __init__(self, dataset, next, sess, nbsamples, batchsize, augment=False):
		self.dataset = dataset
		self.next = next
		self.sess = sess
		self.nbsamples = nbsamples
		self.batchsize = batchsize
		self.augment = augment
		self.i = 1

	def __len__(self):
		'Denotes the number of batches per epoch'
		return self.nbsamples // self.batchsize

	def __getitem__(self, index):
		'Generate one batch of data'
		zmax = 150
		xmax = 205
		ymax = 216
		xmax = xmax//2
		ymax = ymax//2
		zmax = zmax//2
		batch = self.sess.run(self.next)
		originals = batch[0].reshape((-1, 1, xmax, ymax, zmax)).astype(np.float16)
		duplicates = batch[1].reshape((-1, 1, xmax, ymax, zmax)).astype(np.float16)
		masks = batch[2].reshape((-1, 1, xmax, ymax, zmax)).astype(np.float16)

		"""
		if self.augment == True:
			for i in range(imgs.shape[0]):
				angle = np.random.randint(0, 6)
				neg = np.random.randint(0,2)
				if neg == 1:
					angle = - angle
				flip = np.random.randint(0,2)
				imgs[i,:,:,:] = scipy.ndimage.interpolation.rotate(imgs[i,:,:,:], angle, axes=(1,2), reshape=False)
				imgs2[i,:,:,:] = scipy.ndimage.interpolation.rotate(imgs2[i,:,:,:], angle, axes=(1,2), reshape=False)
				if flip == 1:
					imgs[i,:,:,:] = np.flip(imgs[i,:,:,:], axis=2)	
					imgs2[i,:,:,:] = np.flip(imgs2[i,:,:,:], axis=2)		
		imgs = imgs.reshape((-1, 1, xmax, ymax, zmax))
		imgs2 = imgs2.reshape((-1, 1, xmax, ymax, zmax))
		labels = batch[0].reshape((-1,))
		ptids = batch[5].reshape((-1,))
		print(self.i)
		"""
		self.i += 1
		return [originals,duplicates], masks

####
train_filename = 'groundTruths.tfrecords'
epochs=600
batch_size = 2 # erreur de segmentation avec un batch size trop grand sur cpu
data_path = tf.placeholder(dtype=tf.string, name="tfrecord_file")
#print(train_filename.dtype)

with tf.device('/cpu:0'):
	dataset = tf.data.TFRecordDataset(data_path)
	dataset = dataset.map(_read_from_tfrecord).shuffle(100,None,False)
	dataval = dataset.take(40)
	dataval = dataval.repeat().batch(batch_size) #don't shuffle validation data ?
	datatrain = dataset.skip(40)
	datatrain = datatrain.shuffle(buffer_size=100).repeat().batch(batch_size)

gc.collect()

##########################


nb_classes=2

#####

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_image_data_format("channels_first")
#K.set_floatx("float16")
zmax = 150
xmax = 205
ymax = 216
xmax = xmax//2
ymax = ymax//2
zmax = zmax//2

reg = 0.0001

with tf.device('/cpu:0'):
	left_input = Input((1,xmax,ymax,zmax),name="left_input")
	right_input = Input((1,xmax,ymax,zmax),name="right_input")
	inputs = Input((1,xmax,ymax,zmax))

	x = BatchNormalization(axis=1, momentum=0.99)(inputs)

	c1 = Conv3D(16, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=1, momentum=0.99)(c1)
	o1 = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = AveragePooling3D(pool_size=3, strides=2, padding="same")(o1)

	c2 = Conv3D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=1, momentum=0.99)(c2)
	o2 = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = AveragePooling3D(pool_size=3, strides=2, padding="same")(o2)

	c3 = Conv3D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=1, momentum=0.99)(c3)
	o3 = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = AveragePooling3D(pool_size=3, strides=2, padding="same")(o3)

	enc = Conv3D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)

	x = BatchNormalization(axis=1, momentum=0.99)(enc)
	res = keras.layers.LeakyReLU(alpha=0.01)(x)

	model1 = Model(inputs=inputs, outputs=o1)
	model2 = Model(inputs=inputs, outputs=o2)
	model3 = Model(inputs=inputs, outputs=o3)
	model = Model(inputs=inputs, outputs=res)
	
	o1_l = model1(left_input)
	o1_r = model1(right_input)
	d1 = keras.layers.subtract([o1_l, o1_r])
	d1 = Lambda (K.abs)(d1)
	d1 = keras.layers.ZeroPadding3D(padding=((1,1), (2,2), (2,3)))(d1)
	
	o2_l = model2(left_input)
	o2_r = model2(right_input)
	d2 = keras.layers.subtract([o2_l, o2_r])
	d2 = Lambda (K.abs)(d2)
	d2 = keras.layers.ZeroPadding3D(padding=((0,1), (1,1), (1,1)))(d2)
	
	o3_l = model3(left_input)
	o3_r = model3(right_input)
	d3 = keras.layers.subtract([o3_l, o3_r])
	d3 = Lambda (K.abs)(d3)
	d3 = keras.layers.ZeroPadding3D(padding=((0,0), (0,1), (0,1)))(d3)

	encoded_l = model(left_input)
	encoded_r = model(right_input)
	diff = keras.layers.subtract([encoded_l, encoded_r])
	diff = Lambda (K.abs)(diff)
	x = Conv3D(32, 3, padding="same", activation = keras.layers.LeakyReLU(alpha=0.01))(diff)
	x = BatchNormalization(axis=1, momentum=0.99)(x)
	x = keras.layers.LeakyReLU(alpha=0.01)(x)
	
	x = UpSampling3D((2, 2, 2))(x)
	x = keras.layers.add([x, d3])
	x = Conv3D(32, 3, padding="same", activation = keras.layers.LeakyReLU(alpha=0.01))(x)	
	x = BatchNormalization(axis=1, momentum=0.99)(x)
	x = keras.layers.LeakyReLU(alpha=0.01)(x)
	
	x = UpSampling3D((2, 2, 2))(x)
	x = keras.layers.add([x, d2])
	x = Conv3D(16, 3, padding="same", activation = keras.layers.LeakyReLU(alpha=0.01))(x)	
	x = BatchNormalization(axis=1, momentum=0.99)(x)
	x = keras.layers.LeakyReLU(alpha=0.01)(x)
	
	x = UpSampling3D((2, 2, 2))(x)
	x = keras.layers.add([x, d1])
	x = Conv3D(16, 3, padding="same", activation = keras.layers.LeakyReLU(alpha=0.01))(x)	
	x = BatchNormalization(axis=1, momentum=0.99)(x)
	x = keras.layers.LeakyReLU(alpha=0.01)(x)
	
	x = Conv3D(1, 3, padding="same", activation = keras.layers.LeakyReLU(alpha=0.01))(x)	
	x = BatchNormalization(axis=1, momentum=0.99)(x)
	x = keras.layers.Softmax()(x)
	recon = keras.layers.Cropping3D(cropping=((1, 1), (2, 2), (2, 3)))(x)
	
	
	siamese_net = Model(inputs=[left_input,right_input],outputs=recon)

print("network ok")
gc.collect()
parallel_model = multi_gpu_model(siamese_net,gpus=2)

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000000001, amsgrad=False)
parallel_model.compile(optimizer=adam,
	          loss='categorical_crossentropy',
	          metrics=['mean_squared_logarithmic_error'])
#siamese_net.summary()
gc.collect()

iter1 = datatrain.make_initializable_iterator()
next1 = iter1.get_next()
iter2 = dataval.make_initializable_iterator()
next2 = iter2.get_next()
sess.run(iter1.initializer, feed_dict={data_path: train_filename})
sess.run(iter2.initializer, feed_dict={data_path: train_filename})
print("init iterateur ok")


batch0 = sess.run(next2)
imgtest = batch0[0][0,:,:,:]
imgtest2 = batch0[1][0,:,:,:]
imgtest = imgtest.reshape((-1, 1, xmax, ymax, zmax))
imgtest2 = imgtest2.reshape((-1, 1, xmax, ymax, zmax))
train_gen = DataGenerator(datatrain, next1, sess, 100, batch_size, True)
val_gen = DataGenerator(dataval, next2, sess, 40, batch_size, False)	
history = parallel_model.fit_generator(train_gen, epochs=epochs, verbose=2, validation_data=val_gen, max_queue_size=40, workers=12, use_multiprocessing=False, callbacks=[keras.callbacks.CSVLogger("log_change.csv", separator=',', append=True)])

gc.collect()
	
#siamese_net.save("multimodal.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Val loss'], loc='upper left')
plt.show()

plt.plot(history.history['mean_squared_logarithmic_error'])
plt.plot(history.history['val_mean_squared_logarithmic_error'])
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['Train log err', 'Val log err'], loc='upper left')
plt.show()

siamese_net.save("change.h5")
plotExampleImage(np.squeeze(imgtest),"input1")
plotExampleImage(np.squeeze(imgtest2),"input2")
plt.show()

layer_output = recon
activation_model = Model(inputs=[left_input,right_input], outputs=layer_output) # Creates a model that will return these outputs, given the model input
activation = activation_model.predict_generator(val_gen,steps=1)
plotExampleImage(np.squeeze(activation[0,:,:,:]),"recon")
plt.show()	

siamese_net.save("change.h5")
sess.close()
