


import numpy as np 
import matplotlib.pyplot as plt
import os, sys
sys.path.append("/home/lulin/na4/my_packages")
sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
from utils import generator



def make_gif(dirpath="../samples/exp0", save2path="../demo/test_imageio.gif"):
	import imageio
	import cv2
	dirname = "/".join(save2path.split("/")[:-1])
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	collections = []
	for file in generator(root_dir=dirpath, file_type='png', file_label_fun=None, stop_after = None, verbose=1):
		collections.append(file)
		# print(file)
	collections.sort()
	collections = sorted(collections, key=lambda x:int(x.split("/")[-1].split(".")[0]))
	# print(collections)
	print("Reading images...")
	collections = [cv2.imread(x) for x in collections[100:-100]]
	# collections = list(map(lambda file:cv2.imread(file), collections))
	print("+ Done.")
	print("Making Gif...")
	imageio.mimsave(save2path, collections)
	print("+ Done.")
	# import ipdb; ipdb.set_trace()


if __name