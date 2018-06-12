

import os
import numpy as np 
import matplotlib.pyplot as plt




def plot_D_statistic(history, show=False, cut=None, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	if cut is not None:
		history = history[:cut]
	length = len(history)
	xaxis_scale = np.arange(0, length*50, 50)
	plt.figure()
	plt.title("Critic loss")
	plt.plot(xaxis_scale, history[:, 0], label="WGAN-GP loss")
	plt.plot(xaxis_scale, history[:, 2], label="real imgs loss")
	plt.plot(xaxis_scale, history[:, 1], label="fake imgs loss")
	plt.plot(xaxis_scale, history[:, 3], label="GP penalization loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_loss.png"))
	if show:
		plt.show()
	plt.close()

	plt.figure()
	plt.title("Critic accuracy")
	plt.plot(xaxis_scale, history[:, 5], label="Critic acc (real)")
	plt.plot(xaxis_scale, history[:, 4], label="Critic acc (fake)")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_acc.png"))
	if show:
		plt.show()
	plt.close()

def plot_G_statistic(history, show=False, cut=None, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	if cut is not None:
		history = history[:cut]
	length = len(history)
	# last_100_mean_seg_acc = np.mean(history[-100:, 4])
	xaxis_scale = np.arange(0, length*50, 50)
	plt.figure()
	plt.title("Generator loss")
	plt.plot(xaxis_scale, history[:, 0], label="WGAN loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "generator_loss.png"))
	if show:
		plt.show()
	plt.close()

	# plt.figure()
	# plt.title("Segmenter accuracy")
	# # plt.set_xticks(np.arange(0, length*10, 10))
	# best_dice = np.max(history[:, -1])
	# plt.plot(xaxis_scale, history[:, 4], label="Seg acc (train)")
	# plt.plot(xaxis_scale, 1.-history[:, 2], label="Seg dice (train)")
	# plt.plot(xaxis_scale, history[:, -2]/100, label="Seg dice (current)")
	# plt.plot(xaxis_scale, (best_dice/100)*np.ones(length) , label="Best mean dice {:.2f}%".format(best_dice))
	# plt.plot(xaxis_scale, history[:, -1]/100, label="Seg dice (test)")
	# # plt.plot(last_100_mean_cls_acc*np.ones(length), label="Last mean acc: {}".format(last_100_mean_cls_acc))

	# plt.xlabel("Iteration")
	# plt.legend(loc="best")
	# plt.savefig(os.path.join(save2dir, "segmenter_acc.png"))
	# if show:
	# 	plt.show()
	# plt.close()
if __name__ == "__main__":
	print("Start")

	algo = "WGAN-GP"
	# exp_name = "Exp3(one_side)"
	exp_name = "Exp1"
	
	D_hist = np.loadtxt(open("./output/history/WGAN-GP/{}/D_Losses.csv".format(exp_name), "r"), delimiter=",")
	plot_D_statistic(D_hist, show=False, save2dir="./output/history/WGAN-GP/{}".format(exp_name))
	G_hist = np.loadtxt(open("./output/history/WGAN-GP/{}/G_Losses.csv".format(exp_name), "r"), delimiter=",")
	plot_G_statistic(G_hist, save2dir="./output/history/WGAN-GP/{}".format(exp_name))