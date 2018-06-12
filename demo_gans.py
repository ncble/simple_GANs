"""

author: Lu Lin
date: 8/6/2018
"""


import numpy as np 
import os

import keras
import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)


from keras.layers import Dense, Conv2DTranspose, Conv2D, Reshape, Flatten
from keras.layers import Input, UpSampling2D, Activation, Add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
def reset_session():
	
	gpu_fraction=0.2
	K.get_session().close()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	K.set_session(sess)

def build_generator():
	i = Input(shape=(128,)) # noise vector size=128
	l = Dense(100, activation="relu")(i)
	l = Dense(50, activation="relu")(l)
	l = Dense(20, activation="relu")(l)
	l = Dense(12, activation="tanh")(l)
	o = Reshape((2,2,3))(l) # image 2x2 with 3 channels: shape (2,2, 3) 

	model = Model(inputs=i, outputs=o)
	return model

def build_critic():
	i = Input(shape=(2,2,3)) # noise vector size=128
	l = Conv2DTranspose(10, kernel_size=2, strides=2, activation="relu")(i) # double image size to 4x4
	l = LeakyReLU(alpha=0.2)(l)
	l = Conv2DTranspose(10, kernel_size=2, strides=2, activation="relu")(l) # double image size to 4x4
	l = LeakyReLU(alpha=0.2)(l)
	o = Dense(1, activation=None)(Flatten()(l))
	model = Model(inputs=i, outputs=o)
	return model

def wasserstein_loss(y_gt, y_pre):
	return -K.mean(y_gt*y_pre)


class SimpleGANs(object):
	"""docstring for SimpleGANs"""
	def __init__(self):
		super(SimpleGANs, self).__init__()
		# self.arg = arg
	def load_data(self):
		return

	def build_model(self):
		noise = Input(shape=(128,))
		img = Input(shape=(2,2,3))
		self.Generator = build_generator()
		self.Critic = build_critic()
		self.Generator.name = "Generator"
		self.Critic.name = "Critic"
		
		fake = self.Generator(noise)
		fake_score = self.Critic(fake)
		real_score = self.Critic(img)
		self.combined_critic = Model(inputs=[noise, img], outputs=[fake_score, real_score])
		self.Generator.trainable = False
		self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss], optimizer="sgd", metrics=["acc"])
		# combined_critic.name = "Critic"
		
		self.Generator.trainable = True
		self.Critic.trainable = False

		self.combined_generator = Model(inputs=noise, outputs=[fake_score])
		self.combined_generator.compile(loss=[wasserstein_loss], optimizer="sgd", metrics=["acc"])
		# combined_generator.name = "Generator"
		
	def summary(self):
		self.Critic.summary()
		self.combined_critic.summary()
		self.combined_generator.summary()

	def save_weights(self):
		self.combined_generator.save_weights("./model_weights.h5")

	def load_only_generator_weights(self):
		print("Loading pretrained weights (only for G)...")
		self.combined_generator.load_weights("./model_weights.h5")
		g_weights = self.Generator.get_weights()
		# reset_session()
		K.clear_session()
		self.build_model()
		self.Generator.set_weights(g_weights)
		print("+ Done.")
	def fake(self):
		np.random.seed(17)
		z = np.random.random((1,128))
		print(self.Generator.predict(z))

	def train(self):
		np.random.seed(17)
		z = np.random.random((10,128))
		score = np.ones((10,1))
		# for _ in range(5):
		history = self.combined_generator.train_on_batch(z, score)
		print(history)




if __name__ == "__main__":
	print("Start")
	gan = SimpleGANs()
	gan.build_model()
	# gan.summary()
	# gan.save_weights()
	gan.load_only_generator_weights()
	gan.summary()
	# gan.fake()
	# gan.train()