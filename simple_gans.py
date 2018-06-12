"""

author: Lu Lin
date: 8/6/2018
"""


import numpy as np 
import os, sys, glob
from functools import partial
import keras
import keras.backend as K
import tensorflow as tf

def reset_session(gpu_fraction=0.1):
	
	
	K.get_session().close()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	# K.set_session(sess)
	return sess

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

# reset_session()
# sess = reset_session(gpu_fraction=0.1)
# K.set_session(sess)
from keras.layers import Dense, Conv2DTranspose, Conv2D, Reshape, Flatten
from keras.layers import Input, UpSampling2D, Activation, Add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from keras.layers.merge import _Merge
from time import time
from DLalgors import _DLalgo
from keras.utils import plot_model
from keras.initializers import he_normal, he_uniform

def wasserstein_loss(y_gt, y_pre):
	return -K.mean(y_gt*y_pre)
def my_critic_acc(y_true, y_pred):
	sign = K.less(K.zeros(1), y_true*y_pred)
	return K.mean(sign)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight, singular_value=1, method="one_side"):
	gradients = K.gradients(y_pred, averaged_samples)[0]
	gradients_sqr = K.square(gradients)
	gradients_sqr_sum = K.sum(gradients_sqr,
							  axis=np.arange(1, len(gradients_sqr.shape)))
	gradient_l2_norm = K.sqrt(gradients_sqr_sum)
	# compute lambda * (1 - ||grad||)^2 still for each single sample
	if method == "two_sides":
		gradient_penalty = gradient_penalty_weight * K.square(singular_value - gradient_l2_norm) # Exp2 WGAN-GP
	elif method == "one_side":
		gradient_penalty = gradient_penalty_weight * K.square(K.relu(gradient_l2_norm-singular_value))
	# return the mean as loss over all the batch samples
	return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
	def __init__(self):
		super(RandomWeightedAverage, self).__init__()
	def _merge_function(self, inputs):
		weights = K.random_uniform((1, 1, 1))
		return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class my_generator(object):
	"""docstring for my_generator"""
	def __init__(self, X):
		super(my_generator, self).__init__()
		self.X = X
		self.N = len(self.X)
		self.current = 0
	def preprocessing(self):
		# self.mean = np.mean(self.X,axis=0)
		# self.std = np.mean(self.X,axis=0)
		

		### Minimax normalization
		self.max = np.max(self.X, axis=0)
		self.min = np.min(self.X, axis=0)
		self.range = (self.max - self.min)
		self.X = (self.X-self.min)/ self.range
		self.X = 2*self.X-1
	def next(self, batch_size=32):
		## assert self.N > batch size !!
		if self.current + batch_size < self.N:
			A = self.X[self.current: self.current + batch_size]
			self.current = self.current + batch_size
			return A
		else:
			res = self.N - self.current 
			A = self.X[self.current:]
			B = self.X[0:(batch_size - res)]
			
			self.current = batch_size - res
			return np.vstack([A, B])
	def shuffle(self):
		np.random.shuffle(self.X)


		
class SimpleGANs(_DLalgo):
	"""docstring for SimpleGANs"""
	def __init__(self, dim=2, algo="WGAN-GP"):
		super(SimpleGANs, self).__init__()
		self.dim = dim
		self.opt_D_config = {"lr":5*1e-5, "beta_1": 0.0, "beta_2": 0.9}
		self.opt_G_config = {"lr": 1e-4, "beta_1":0.0, "beta_2":0.9}
		self.optimizerD = Adam(**self.opt_D_config)
		self.optimizerG = Adam(**self.opt_G_config)
		self.algo = algo #"WGAN-GP" # "WGAN", "JS", "Hinge-JS"
		self.noise_size = 128
		self.gp_method = "two_sides" # "two_sides" #"one_side"
		self.singular_value = 1.
		self.GRADIENT_PENALTY_WEIGHT = 1. # Exp3 1.0
	
		self.critic_steps = 5
		self.use_He_initialization = True
		self.my_initializer = lambda :he_normal() if self.use_He_initialization else "glorot_uniform" # TODO

		
		# self.arg = arg
	def load_data(self, source, target, shuffle=True):
		print("Loading dataset from npy files...")
		self.dataA = np.load(source)
		self.dataB = np.load(target)
		self.source_name = source.split("/")[-1]
		self.target_name = target.split("/")[-1]
		if shuffle:
			np.random.shuffle(self.dataA)
			np.random.shuffle(self.dataB)

		print("+ Done.")
		
		print("Building data generator...")
		self.dataA = my_generator(self.dataA)
		self.dataB = my_generator(self.dataB)
		print("+ Done.")
		print("Preprocessing dataset...")
		self.dataA.preprocessing()
		self.dataB.preprocessing()
		print("Source data has {} samples.".format(self.dataA.N))
		print("Target data has {} samples.".format(self.dataB.N))
		print("+ Done.")

	def build_generator(self, features_size=2):
		i = Input(shape=(128,), name="noise") # noise vector size=128
		condition = Input(shape=(features_size, ), name="conditional_vec")


		l = Dense(100, activation="relu", kernel_initializer=self.my_initializer())(i)
		l = concatenate([l, condition], axis=-1)
		l = Dense(50, activation="relu", kernel_initializer=self.my_initializer())(l)
		l = Dense(20, activation="relu", kernel_initializer=self.my_initializer())(l)
		o = Dense(features_size, activation="tanh", kernel_initializer=self.my_initializer())(l)
		# o = Reshape((2,2,3))(l) # image 2x2 with 3 channels: shape (2,2, 3) 

		model = Model(inputs=[i, condition], outputs=o)
		return model

	def build_critic(self, features_size=2):
		i = Input(shape=(features_size, ), name="features_vec") # noise vector size=128
		
		# l = Dense(100, activation="relu")(i)
		l = Dense(100, activation=None, kernel_initializer=self.my_initializer())(i)
		l = LeakyReLU(alpha=0.2)(l)
		# l = Dense(100, activation="relu")(l)
		l = Dense(100, activation=None, kernel_initializer=self.my_initializer())(l)
		l = LeakyReLU(alpha=0.2)(l)
		l = Dense(50, activation=None, kernel_initializer=self.my_initializer())(l)
		l = LeakyReLU(alpha=0.2)(l)
		# l = Dense(50, activation="relu")(l)
		# l = Conv2DTranspose(10, kernel_size=2, strides=2, activation="relu")(i) # double image size to 4x4
		# l = LeakyReLU(alpha=0.2)(l)
		# l = Conv2DTranspose(10, kernel_size=2, strides=2, activation="relu")(l) # double image size to 4x4
		
		o = Dense(1, activation=None, kernel_initializer=self.my_initializer())(l)
		model = Model(inputs=i, outputs=o)
		return model
	def build_model(self, fromdir=None):
		if fromdir is None:
			noise = Input(shape=(self.noise_size,), name="noise")
			source = Input(shape=(self.dim,), name="source")
			target = Input(shape=(self.dim, ), name="target")
			self.Generator = self.build_generator(features_size=self.dim)
			self.Critic = self.build_critic(features_size=self.dim)
			self.Generator.name = "Generator"
			self.Critic.name = "Critic"
			
			fake = self.Generator([noise, source])
			fake_score = self.Critic(fake)
			real_score = self.Critic(target)
			
			
			# if self.algo == "WGAN":
			# elif self.algo == "WGAN-GP":
			# elif self.algo == "JS":
			# elif self.algo == "Hinge-JS":

			if self.algo == "WGAN":
				self.combined_critic = Model(inputs=[noise, source, target], outputs=[fake_score, real_score])
				self.Critic.trainable = True
				self.Generator.trainable = False
				self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss], 
											optimizer=self.optimizer, 
											metrics=[my_critic_acc])
			elif self.algo == "WGAN-GP":
				
				avg = RandomWeightedAverage()([target, fake])
				avg_output_dummy = self.Critic(avg)

				partial_gp_loss = partial(gradient_penalty_loss,
					  averaged_samples=avg,
					  gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT,
					  singular_value=self.singular_value, 
					  method=self.gp_method)
				partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error


				self.combined_critic = Model(inputs=[noise, source, target], outputs=[fake_score, real_score, avg_output_dummy])
				self.Critic.trainable = True
				self.Generator.trainable = False
				self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], 
											optimizer=self.optimizerD, 
											metrics=[my_critic_acc])
			

			self.Generator.trainable = True
			self.Critic.trainable = False
			self.combined_generator = Model(inputs=[noise, source], outputs=[fake_score])
			self.combined_generator.compile(loss=[wasserstein_loss], 
											optimizer=self.optimizerG, 
											metrics=["acc"])

			# import ipdb; ipdb.set_trace()
			
		else:
			## fromdir = ["./critic.h5", "./generator.h5", "./D_model.h5", "./G_model.h5"]

			### Method 1: Re-compile
			# self.Critic = load_model("./critic.h5", custom_objects={"wasserstein_loss":wasserstein_loss})
			# self.Generator = load_model("./generator.h5", custom_objects={"wasserstein_loss":wasserstein_loss})
			# self.combined_critic = load_model("./D_model.h5", custom_objects={"wasserstein_loss":wasserstein_loss})
			# self.combined_generator = load_model("./G_model.h5", custom_objects={"wasserstein_loss":wasserstein_loss})

			# self.combined_critic.get_layer("Critic").trainable = True
			# self.combined_critic.get_layer("Generator").trainable = False
			# self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss], optimizer="sgd", metrics=["acc"])
			# self.combined_generator.get_layer("Generator").trainable = True
			# self.combined_generator.get_layer("Critic").trainable = False
			# self.combined_generator.compile(loss=[wasserstein_loss], optimizer="sgd", metrics=["acc"])

			### Method 2  # Note: a bite strange in Summary(), it seems that self.Generator and self.Critic are not shared ...
				
			# self.combined_critic = load_model("./D_model.h5", custom_objects={"wasserstein_loss":wasserstein_loss})
			# self.combined_generator = load_model("./G_model.h5", custom_objects={"wasserstein_loss":wasserstein_loss})
			# self.Critic = self.combined_critic.get_layer("Critic")
			# self.Generator = self.combined_critic.get_layer("Generator")
			# self.Critic.trainable = True
			# self.Generator.trainable = False
			# self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss], optimizer="sgd", metrics=["acc"])
			# self.Generator.trainable = True
			# self.Critic.trainable = False
			# self.combined_generator.compile(loss=[wasserstein_loss], optimizer="sgd", metrics=["acc"])

			### Method 3 (best way to guarantee)
			noise = Input(shape=(self.noise_size,), name="noise")
			source = Input(shape=(self.dim,), name="source")
			target = Input(shape=(self.dim, ), name="target")
			self.Critic = load_model(os.path.join(fromdir, "critic.h5"), custom_objects={"wasserstein_loss":wasserstein_loss})
			self.Generator = load_model(os.path.join(fromdir, "generator.h5"), custom_objects={"wasserstein_loss":wasserstein_loss})
			fake = self.Generator([noise, source])
			fake_score = self.Critic(fake)
			real_score = self.Critic(target)

			if self.algo == "WGAN":
				self.combined_critic = Model(inputs=[noise, source, target], outputs=[fake_score, real_score])
				self.Critic.trainable = True
				self.Generator.trainable = False
				self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss], 
											optimizer=self.optimizer, 
											metrics=[my_critic_acc])
			elif self.algo == "WGAN-GP":
				
				avg = RandomWeightedAverage()([target, fake])
				avg_output_dummy = self.Critic(avg)

				partial_gp_loss = partial(gradient_penalty_loss,
					  averaged_samples=avg,
					  gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT,
					  singular_value=self.singular_value, 
					  method=self.gp_method)
				partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error


				self.combined_critic = Model(inputs=[noise, source, target], outputs=[fake_score, real_score, avg_output_dummy])
				self.Critic.trainable = True
				self.Generator.trainable = False
				self.combined_critic.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], 
											optimizer=self.optimizerD, 
											metrics=[my_critic_acc])
			self.Generator.trainable = True
			self.Critic.trainable = False
			self.combined_generator = Model(inputs=[noise, source], outputs=[fake_score])
			self.combined_generator.compile(loss=[wasserstein_loss], optimizer="sgd", metrics=["acc"])
		
	def write_tensorboard_graph(self, to_dir="./logs", save_png2dir="./Model_graph"):
		if not os.path.exists(save_png2dir):
			os.makedirs(save_png2dir)
		# tensorboard = keras.callbacks.TensorBoard(log_dir=to_dir, histogram_freq=0, write_graph=True, write_images=False, write_grads=False)
		# tensorboard.set_model(self.combined_GS)
		# tensorboard.set_model(self.Critic)
		# tensorboard.set_model(self.Generator)
		try:
			plot_model(self.Critic, to_file=os.path.join(save_png2dir, "critic.png"))
			plot_model(self.Generator, to_file=os.path.join(save_png2dir, "generator.png"))
		except:
			pass
	def summary(self):
		print("="*50)
		print("Critic summary:")
		self.Critic.summary()
		print("="*50)
		print("Critic model summary:")
		self.combined_critic.summary()
		print("="*50)
		print("Generator model summary:")
		self.combined_generator.summary()

	def save_weights(self):
		self.combined_generator.save_weights("./model_weights.h5")
		
	def save_model(self, save2dir=None):
		if not os.path.exists(save2dir):
			os.makedirs(save2dir)

		self.Critic.save(os.path.join(save2dir, "critic.h5"))
		self.Generator.save(os.path.join(save2dir, "generator.h5"))
		self.combined_critic.save(os.path.join(save2dir, "D_model.h5"))
		self.combined_generator.save(os.path.join(save2dir, "G_model.h5"))

	def reset_history_in_folder(self, dirpath):
		for filepath in glob.glob(os.path.join(dirpath, "*.csv")):
			os.remove(filepath)

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
		z = np.random.random((1,self.noise_size))
		source = np.random.random((1, self.dim))
		print(self.Generator.predict([z, source]))

	def train(self, iterations, batch_size=32, 
		savefig2dir="./output/figures/WGAN", 
		savehis2dir="./output/history/WGAN", 
		saveWeights2dir="./weights/WGAN",
		plot_range=(-60,60)):
		self.batch_size = batch_size
		if not os.path.exists(savefig2dir):
			os.makedirs(savefig2dir)
		if not os.path.exists(savehis2dir):
			os.makedirs(savehis2dir)
		if not os.path.exists(saveWeights2dir):
			os.makedirs(saveWeights2dir)

		self.save_config(save2path=os.path.join(saveWeights2dir, "config.dill"))
		config_message = self.print_config(return_message=True)
		with open(os.path.join(saveWeights2dir, "config.txt"), "w") as file:
			file.write(config_message)

		st = time()
		for iteration in range(iterations):
			
			if (iteration % 100 == 0): #  and (iteration>0) # plot initial clouds to see if they have non-zeros overlap area !
				Z = np.random.normal(0,1, (self.dataA.N, self.noise_size))
				# B = self.dataB.X * (self.dataB.range) + self.dataB.min
				B = self.dataB.X * (self.dataB.range/2) + self.dataB.min + (self.dataB.range/2)

				Y = self.Generator.predict([Z, self.dataA.X])
				# Y = Y*self.dataA.range + self.dataA.min
				# Y = Y* (self.dataA.range/2) + self.dataA.min + (self.dataA.range/2)
				Y = Y* (self.dataB.range/2) + self.dataB.min + (self.dataB.range/2)
				limite = plot_range
				plt.figure()
				plt.title("Iter {}".format(iteration))
				plt.xlim(limite)
				plt.ylim(limite)
				plt.scatter(B[:,0], B[:,1], label="target", color='red')
				plt.scatter(Y[:,0], Y[:,1], label="fake", alpha=0.5, color='green')
				plt.legend(loc="best")
				plt.savefig(os.path.join(savefig2dir, "iter_{}.png".format(iteration)))
				plt.close()

			for _ in range(self.critic_steps):
				z = np.random.normal(0,1, (batch_size,self.noise_size))
				# inputs=[noise, source, target], outputs=[fake_score, real_score]
				x_source = self.dataA.next(batch_size=batch_size)
				x_target = self.dataB.next(batch_size=batch_size)
				pos = np.ones((batch_size, 1))
				neg = -pos
				if self.algo == "WGAN":
					d_history = self.combined_critic.train_on_batch([z, x_source, x_target], [neg, pos])
				elif self.algo == "WGAN-GP":
					dummy_y = pos #np.zeros((batch_size, 1)) # it's useless
					d_history = self.combined_critic.train_on_batch([z, x_source, x_target], [neg, pos, dummy_y])					

			z = np.random.normal(0,1, (batch_size,self.noise_size))
			g_history = self.combined_generator.train_on_batch([z, x_source], [pos])		

			if (iteration % 50 == 0) and (iteration>0):

				if self.algo == "WGAN":
					message = "{} : [D - loss: {:.3f}, (-) loss: {:.3f}, (+) loss: {:.3f}, (-) acc: {:.2f}%, (+) acc: {:.2f}%], [G - loss: {:.3f} (D_loss_fake)]".format(iteration, 
									d_history[0],
									d_history[1],
									d_history[2],
									100*d_history[3],
									100*d_history[4],
									g_history[0])
				elif self.algo == "WGAN-GP":
					message = "{} : [D - loss: {:.4f}, GP-loss: {:.4f}, (-) acc: {:.2f}%, (+) acc: {:.2f}%], [G - loss: {:.3f} (D_loss_fake)]".format(iteration, 
									d_history[0],
									d_history[3],
									100*d_history[4],
									100*d_history[5],
									g_history[0])

				et = time()
				message = message + " ({:.2f} s)".format(et-st)
				print(message)
				st = et
				
				with open(os.path.join(savehis2dir, "D_Losses.csv"), "ab") as csv_file:
						np.savetxt(csv_file, np.array(d_history).reshape(1,-1), delimiter=",")
				with open(os.path.join(savehis2dir, "G_Losses.csv"), "ab") as csv_file:
					np.savetxt(csv_file, np.array(g_history).reshape(1,-1), delimiter=",")
	def deploy(self):
		return


class DataDistribution(object):
	"""docstring for DataDistribution"""
	def __init__(self, dim=2):
		super(DataDistribution, self).__init__()
		self.dim = dim
		
	def create0(self, n_samples, seed=None, show=False):
		"""
		Studnet distribution ? TODO
		"""
		rng = np.random.RandomState(42)
		S = rng.standard_t(1.5, size=(n_samples, self.dim))
		S[:, 0] *= 2
		A = np.array([[1, 1], [0, 2]])  # Mixing matrix
		X = np.dot(S, A.T)  # Generate observations
		if show:
			plt.figure()
			plt.xlim((-5,5))
			plt.ylim((-5,5))
			plt.scatter(X[:,0], X[:,1])
			plt.show()
			plt.close()
		return X
	def create1(self, n_samples, center=np.zeros(2), ratio=[1,2], seed=None, show=False):
		"""
		Multivariate gaussian distribution

		"""
		np.random.seed(seed)
		mean = np.array(center)#np.zeros(self.dim)
		A = np.array(ratio)+ 0.01*(2*np.random.random((self.dim, self.dim))-1)
		# A = np.diag(np.arange(1,100,2)[:self.dim]) + 0.005*np.random.random((self.dim, self.dim))
		cov = (A+A.T)/2
		X = np.random.multivariate_normal(mean, cov, n_samples)
		
		if show:
			plt.figure()
			plt.xlim((-5,5))
			plt.ylim((-5,5))
			plt.scatter(X[:,0], X[:,1])
			plt.show()
		return X

	def create2(self, n_samples, n_clusters=6, seed=None, show=False):
		"""
		Classic: 6 clusters


		"""
		np.random.seed(seed)
		rotation60 = np.array([[0.5, -np.sqrt(3)/2],[np.sqrt(3)/2, 0.5]])
		center = np.ones(2)*20
		cluster_buffer = []


		for _ in range(n_clusters):
			# center = (2*np.random.random(2)-1)*80.
			center = rotation60.dot(center)
			ratio = np.random.choice(5,2, replace=False)
			X = self.create1(n_samples, center=center, ratio=ratio)
			cluster_buffer.append(X)
			
		X = np.vstack(cluster_buffer) 

		if show:
			limite = (-60,60)
			plt.figure()
			plt.title("{} clusters of {} samples each".format(n_clusters, n_samples))
			plt.xlim(limite)
			plt.ylim(limite)
			plt.scatter(X[:,0], X[:,1], s=10, alpha=0.8)
			plt.savefig("{}_clusters.png".format(n_clusters))
			plt.show()
		return X
	def create3(self, n_samples, seed=None, show=False):
		"""
		Gaussian distribution: Source data

		"""
		np.random.seed(seed)
		X = 10*np.random.randn(n_samples, 2)
		if show:
			limite = (-60,60)
			plt.figure()
			plt.title("Gaussian distribution")
			plt.xlim(limite)
			plt.ylim(limite)
			plt.scatter(X[:,0], X[:,1], s=10, alpha=0.8)
			plt.savefig("normal_source.png")
			plt.show()
		return X

	def create4(self, n_samples, n_classes=100, amplitude=(-100,100), seed=None, show=False):
		"""
		Grid (multimodel distribution) clouds: 100 clouds

		"""
		n_split = int(np.sqrt(n_classes))
		np.random.seed(seed)
		grid_axis_x = np.linspace(amplitude[0], amplitude[1], n_split)
		grid_axis_y = np.linspace(amplitude[0], amplitude[1], n_split)
		# rotation60 = np.array([[0.5, -np.sqrt(3)/2],[np.sqrt(3)/2, 0.5]])
		# center = np.ones(2)*20
		cluster_buffer = []
		
		for x in grid_axis_x:
			for y in grid_axis_y:
				# center = (2*np.random.random(2)-1)*80.
				center = np.array([x, y])
				ratio = np.random.choice(5,2, replace=False)
				X = self.create1(n_samples, center=center, ratio=ratio)
				cluster_buffer.append(X)
			
		X = np.vstack(cluster_buffer) 

		if show:
			limite = (-120, 120) #(-60,60)
			plt.figure()
			plt.title("Grid: {} clusters".format(n_classes))
			plt.xlim(limite)
			plt.ylim(limite)
			plt.scatter(X[:,0], X[:,1], s=10, alpha=0.8)
			plt.savefig("grid_{}.png".format(n_classes))
			plt.show()
			plt.close()
		return X

	def create5(self,n_samples, n_classes=50, seed=None, show=False):
		"""
		Spiral multi-model clusters

		"""
		np.random.seed(seed)
		rotation30 = np.array([[np.sqrt(3)/2, -0.5],[0.5, np.sqrt(3)/2]])
		center = np.ones(2)*70
		r_decay = 0.995
		cluster_buffer = []

		for i in range(n_classes):
			center = center*(r_decay**i)
			center = rotation30.dot(center)
			ratio = np.random.choice(5,2, replace=False)
			X = self.create1(n_samples, center=center, ratio=ratio)
			cluster_buffer.append(X)
		X = np.vstack(cluster_buffer)

		if show:
			limite = (-100, 100) #(-60,60)
			plt.figure()
			plt.title("Spiral: {} clusters".format(n_classes))
			plt.xlim(limite)
			plt.ylim(limite)
			plt.scatter(X[:,0], X[:,1], s=10, alpha=0.8)
			plt.savefig("spiral_{}.png".format(n_classes))
			plt.show()
			plt.close()
		return X
		return 




if __name__ == "__main__":
	print("Start")


	gan = SimpleGANs(dim=2, algo="WGAN-GP")
	gan.build_model()
	# gan.build_model(fromdir="./weights/WGAN-GP/Exp1_bis")
	gan.summary()
	# gan.load_data("normal_source.npy", "grid_100.npy", shuffle=True)
	gan.load_data("normal_source.npy", "6_clusters.npy", shuffle=True)



	try:
		EXP_NUM = "Exp23"
		gan.write_tensorboard_graph(to_dir="./weights/WGAN-GP/{}/board".format(EXP_NUM), save_png2dir="./weights/WGAN-GP/{}".format(EXP_NUM))
		gan.reset_history_in_folder("./output/history/WGAN-GP/{}".format(EXP_NUM))
		gan.train(100000, batch_size=128, 
			savefig2dir="./output/figures/WGAN-GP/{}".format(EXP_NUM), 
			savehis2dir="./output/history/WGAN-GP/{}".format(EXP_NUM), 
			saveWeights2dir="./weights/WGAN-GP/{}".format(EXP_NUM),
			plot_range=(-60,60))
	except KeyboardInterrupt:
		gan.save_model("./weights/WGAN-GP/{}_bis".format(EXP_NUM))
		sys.exit(0)



	# import ipdb; ipdb.set_trace()
	# gan.save_weights()
	# gan.load_only_generator_weights()

	# gan.fake()
	# gan.train()



	######### Save config by hand ###########
	# EXP_NUM = "Exp4"
	# gan.write_tensorboard_graph(save_png2dir="./weights/WGAN-GP/{}".format(EXP_NUM))
	# gan.save_config(save2path="./weights/WGAN-GP/{}/config.dill".format(EXP_NUM))
	# config_message = gan.print_config(return_message=True)
	# with open("./weights/WGAN-GP/{}/config.txt".format(EXP_NUM), "w") as file:
	# 	file.write(config_message)
	##########################################


	############# Data generation ###############
	# data = DataDistribution(2)
	
	# X = data.create2(500, show=True, seed=17)
	# np.save("6_clusters.npy", X)

	# X = data.create3(3000, show=True, seed=17)
	# np.save("normal_source.npy", X)

	# X = data.create4(50, show=True, seed=17)
	# np.save("grid_100.npy", X)
	
	# X = data.create5(60, show=True, seed=17)
	# np.save("spiral_50.npy", X)