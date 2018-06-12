


import numpy as np 
import os

import keras
import keras.backend as K
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)


from keras.layers import Dense, Conv2DTranspose, Conv2D
from keras.layers import Input, UpSampling2D, Activation, Add, concatenate
from keras.models import Model, Sequential

def residual_block(layer_input, filters=512, down_filter=False, normalization=False):
	"""Residual block described in paper"""
	d1 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
	if normalization:
		# d = InstanceNormalization()(d)
		d1 = BatchNormalization(momentum=0.8)(d1) # 6/6/2018: use it for CT #  6/5/2018: remove it for MNIST
	d1 = Activation('relu')(d1)
	d2 = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d1)
	if normalization:
		# d = InstanceNormalization()(d)
		d2 = BatchNormalization(momentum=0.8)(d2) # 6/6/2018: use it for CT #  6/5/2018: remove it for MNIST
	if down_filter:
		d = Add()([d1, d2])
	else:
		d = Add()([d2, layer_input])
	return d

if __name__ == "__main__":
	print("Start")
	features_map = 256
	i = Input(shape=(1,1,features_map))
	img = Input(shape=(128,128,1))
	# o = Conv2DTranspose(features_map, kernel_size=1, strides=4, padding='same')(i)
	# o2 = UpSampling2D()(i)
	# o2 = Conv2D(features_map, )
	r = Conv2DTranspose(features_map, kernel_size=1, strides=4, padding='same')(i)
	r = Conv2D(features_map, kernel_size=3, strides=1, padding='same')(r)
	r = UpSampling2D()(r)
	r = residual_block(r, filters=features_map)
	r = UpSampling2D()(r)
	r = residual_block(r, filters=128, down_filter=True)
	r = UpSampling2D()(r)
	r = residual_block(r, filters=64, down_filter=True)
	r = UpSampling2D()(r)
	r = residual_block(r, filters=32, down_filter=True)
	r = UpSampling2D()(r)
	r = keras.layers.concatenate([r, img])
	print(K.int_shape(r))
	# 
	r = residual_block(r, filters=16, down_filter=True)
	o = Conv2D(1, kernel_size=3, padding='same', activation='tanh')(r)

	# r = Conv2D(128, kernel_size=3, strides=1, padding='same')(r)
	# r = Conv2D(128, kernel_size=3, strides=1, padding='same')(r)
	
	

	model = Model(inputs=[i, img], outputs=o)
	model.summary()