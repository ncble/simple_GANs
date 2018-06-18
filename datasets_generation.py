"""

author: Lu Lin
date: 14/6/2018
"""
import os, sys, glob
import socket
machine_name = socket.gethostname()
print("="*50)
print("Machine name: {}".format(machine_name))
print("="*50)
if machine_name == "lulin-QX-350-Series":
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="1"
	sys.path.append("/home/lulin/na4/my_packages")
import numpy as np
import matplotlib.pyplot as plt

import utils
from draw import draw_clouds

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

	def create_mask1(self, X, show=True):
		shape = X.shape # (N, self.dim)
		Y = np.empty(shape)
		dist = np.linalg.norm(X, axis=1) # L2-distance 

		##### change here to have different mask distribution #####
		## For normal distribution
		Y[dist>15] = np.ones(2) ## Mask (1,1)
		Y[(dist<15)*(dist>10)] = np.array([0,1])
		Y[(dist<10)*(dist>6)] = np.array([1,0])
		Y[(dist<6)] = np.zeros(2)


		## For spiral 50
		# Y[dist>50] = np.ones(2) ## Mask (1,1)
		# Y[(dist<50)*(dist>25)] = np.array([0,1])
		# Y[(dist<25)*(dist>10)] = np.array([1,0])
		# Y[(dist<10)] = np.zeros(2)

		if show:
			my_label = Y[:,0]*2+Y[:,1]
			my_label = my_label.astype(int)
			print("Counting...")
			for i in range(4):
				print("Number of 'class' {}: {}".format(i, np.sum(my_label==i)))



			# limite = 60
			# draw_clouds(X, labels=my_label, title="Distribution with masks", 
			# 	save_to="./normal_masks.png", 
			# 	show=show, 
			# 	xlim=(-limite,limite), 
			# 	ylim=(-limite,limite))


			limite = 120
			draw_clouds(X, labels=my_label, title="Distribution with masks", 
				save_to="./spiral_masks.png", 
				show=show, 
				xlim=(-limite,limite), 
				ylim=(-limite,limite))
			

		return Y







if __name__ == "__main__":
	print("Start")



	############# Data generation ###############
	# if not os.path.exists("./data"):
	# 	os.makedirs("./data")
	data = DataDistribution(2)
	
	# X = data.create2(500, show=True, seed=17)
	# np.save("./data/6_clusters.npy", X)

	# X = data.create3(3000, show=False, seed=17)
	# np.save("./data/normal_source.npy", X)

	# X = data.create4(50, show=True, seed=17)
	# np.save("./data/grid_100.npy", X)
	
	X = data.create5(60, show=False, seed=17)
	# np.save("./data/spiral_50.npy", X)


	Y = data.create_mask1(X, show=True)
	
	np.save("./data/spiral_masks2.npy", Y)
	# np.save("./data/normal_masks.npy", Y)



	# X2 = data.create5(60, show=False, seed=17)
	# my_label = Y[:,0]*2+Y[:,1]
	# my_label = np.round(my_label).astype(int)
	# limite = 120
	# draw_clouds([X,X2], labels=[my_label, my_label+4], title="Distribution with masks", 
	# 	save_to=None, 
	# 	show=True, 
	# 	xlim=(-limite,limite), 
	# 	ylim=(-limite,limite))