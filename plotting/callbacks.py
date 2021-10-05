import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd
from scipy.stats import pearsonr

import seaborn as sns
import sys
sys.path.append("../..")


from gp_functions import rbf_covariance
from scipy.stats import multivariate_normal as mvnpy
from util import polar_warp

from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances


def callback_oned(model, X, Y, X_aligned, data_expression_ax, latent_expression_ax, prediction_ax=None, X_test=None, Y_pred=None, Y_test_true=None, X_test_aligned=None, F_samples=None):
	model.eval()
	markers = [".", "+", "^"]
	colors = ["blue", "orange"]

	if model.fixed_view_idx is not None:
		curr_idx = model.view_idx["expression"][model.fixed_view_idx]
		X_aligned["expression"][curr_idx] = torch.tensor(X[curr_idx].astype(np.float32))
		

	data_expression_ax.cla()
	latent_expression_ax.cla()
	

	data_expression_ax.set_title("Observed data")
	latent_expression_ax.set_title("Aligned data")
	

	data_expression_ax.set_xlabel("Spatial coordinate")
	latent_expression_ax.set_xlabel("Spatial coordinate")
	
	data_expression_ax.set_ylabel("Outcome")
	latent_expression_ax.set_ylabel("Outcome")
	

	data_expression_ax.set_xlim([-10, 10])
	latent_expression_ax.set_xlim([-10, 10])

	for vv in range(model.n_views):

		view_idx = model.view_idx["expression"]

		data_expression_ax.scatter(
			X[view_idx[vv], 0],
			Y[view_idx[vv], 0],
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=100,
			c="blue",
		)
		data_expression_ax.scatter(
			X[view_idx[vv], 0],
			Y[view_idx[vv], 1],
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=100,
			c="orange",
		)
		latent_expression_ax.scatter(
			# model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
			X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
			Y[view_idx[vv], 0],
			c="blue",
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=100,
		)
		latent_expression_ax.scatter(
			# model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
			X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
			Y[view_idx[vv], 1],
			c="orange",
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=100,
		)
		# latent_expression_ax.scatter(
		# 	model.Xtilde.detach().numpy()[vv, :, 0],
		# 	model.delta_list.detach().numpy()[vv][:, 0],
		# 	c="red",
		# 	label="View {}".format(vv + 1),
		# 	marker="^",
		# 	s=100,
		# )

		if F_samples is not None:
			latent_expression_ax.scatter(
				X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
				F_samples.detach().numpy()[view_idx[vv], 0],
				c="red",
				marker=markers[vv],
				s=100,
			)
			latent_expression_ax.scatter(
				X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
				F_samples.detach().numpy()[view_idx[vv], 1],
				c="green",
				marker=markers[vv],
				s=100,
			)

	
	if prediction_ax is not None:

		prediction_ax.cla()
		prediction_ax.set_title("Predictions")
		prediction_ax.set_xlabel("True outcome")
		prediction_ax.set_ylabel("Predicted outcome")
		
		### Plots the warping function
		# prediction_ax.scatter(
		# 	X[view_idx[vv], 0],
		# 	X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
		# 	label="View {}".format(vv + 1),
		# 	marker=markers[vv],
		# 	s=100,
		# 	c="blue",
		# )
		# prediction_ax.scatter(
		# 	model.Xtilde.detach().numpy()[vv, :, 0],
		# 	model.delta_list.detach().numpy()[vv][:, 0],
		# 	c="red",
		# 	label="View {}".format(vv + 1),
		# 	marker="^",
		# 	s=100,
		# )
		latent_expression_ax.scatter(
			X_test_aligned["expression"].detach().numpy()[:, 0],
			Y_pred.detach().numpy()[:, 0],
			c="blue",
			label="Prediction",
			marker="s",
			s=100,
		)
		latent_expression_ax.scatter(
			X_test_aligned["expression"].detach().numpy()[:, 0],
			Y_pred.detach().numpy()[:, 1],
			c="orange",
			label="Prediction",
			marker="s",
			s=100,
		)
		prediction_ax.scatter(
			Y_test_true[:, 0],
			Y_pred.detach().numpy()[:, 0],
			c="black",
			s=100,
		)
		prediction_ax.scatter(
			Y_test_true[:, 1],
			Y_pred.detach().numpy()[:, 1],
			c="black",
			s=100,
			marker="^",
		)
		
		

	data_expression_ax.legend()
	plt.draw()
	plt.pause(1 / 60.0)



def callback_twod(model, X, Y, data_expression_ax, latent_expression_ax):
	model.eval()
	markers = [".", "+", "^"]
	colors = ["blue", "orange"]

	data_expression_ax.cla()
	latent_expression_ax.cla()
	data_expression_ax.set_title("Observed data")
	latent_expression_ax.set_title("Aligned data")

	curr_view_idx = model.view_idx["expression"]

	for vv in range(model.n_views):

		data_expression_ax.scatter(
			X[curr_view_idx[vv], 0],
			X[curr_view_idx[vv], 1],
			c=Y[curr_view_idx[vv], 0],
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=200,
		)
		latent_expression_ax.scatter(
			model.G_means["expression"].detach().numpy()[curr_view_idx[vv], 0],
			model.G_means["expression"].detach().numpy()[curr_view_idx[vv], 1],
			c=Y[curr_view_idx[vv], 0],
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=200,
		)
		latent_expression_ax.scatter(
			model.Xtilde.detach().numpy()[vv][:, 0],
			model.Xtilde.detach().numpy()[vv][:, 1],
			c="red",
			label="View {}".format(vv + 1),
			marker=markers[vv],
			s=200,
		)

	plt.draw()
	plt.pause(1 / 60.0)



