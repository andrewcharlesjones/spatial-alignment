import pandas as pd
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../../../data/codex"
data = pd.read_csv(pjoin(DATA_DIR, "codex_mrl_expression.csv"))  # , nrows=200)

marker_names = data.columns.values[1:-8]
sample_names = data.sample_Xtile_Ytile.str.split("_").str[0].values
sample_names_unique = np.unique(sample_names)

sample1_idx = np.where(sample_names == "BALBc-3")[0]
sample2_idx = np.where(sample_names == "BALBc-2")[0]

data_sample1 = data.iloc[sample1_idx, :]
data_sample2 = data.iloc[sample2_idx, :]

xtilespan = 1344
ytilespan = 1008

def tile_spatial_coordinates(data_df):

	if "xcoord" in data_df.columns or "ycoord" in data_df.columns:
		raise Exception("DataFrame already contains scaled coordinates.")

	tile_nums_split = data_df.sample_Xtile_Ytile.str.split("_")
	x_tile_nums = tile_nums_split.str[1].str[1:].values.astype(float)
	y_tile_nums = tile_nums_split.str[2].str[1:].values.astype(float)
	xcoords = (x_tile_nums - 1) * xtilespan + data_df["X.X"].values
	ycoords = (y_tile_nums - 1) * ytilespan + data_df["Y.Y"].values
	data_df["xcoord"] = xcoords
	data_df["ycoord"] = ycoords
	

tile_spatial_coordinates(data_sample1)
tile_spatial_coordinates(data_sample2)

# plt.scatter(data_sample1.xcoord, data_sample1.ycoord)
# plt.show()
# import ipdb; ipdb.set_trace()

normalized_data1 = data_sample1[marker_names].values.copy()
keep_idx = np.where((np.abs(normalized_data1) >= 10_000).sum(1) == 0)[0]
data_sample1 = data_sample1.iloc[keep_idx]
normalized_data2 = data_sample2[marker_names].values.copy()
keep_idx = np.where((np.abs(normalized_data2) >= 10_000).sum(1) == 0)[0]
data_sample2 = data_sample2.iloc[keep_idx]

# import ipdb

# ipdb.set_trace()
for marker in marker_names:
	plt.figure(figsize=(10, 5))

	plt.subplot(121)
	plt.title("Slice 1")
	curr_data = data_sample1[marker].values
	curr_data = (curr_data - curr_data.mean()) / curr_data.std()
	plt.scatter(
		data_sample1["xcoord"],
		data_sample1["ycoord"],
		c=data_sample1[marker],
		s=1,
		marker="s",
	)

	plt.subplot(122)
	plt.title("Slice 2")
	curr_data = data_sample2[marker].values
	curr_data = (curr_data - curr_data.mean()) / curr_data.std()
	plt.scatter(
		data_sample2["xcoord"],
		data_sample2["ycoord"],
		c=data_sample2[marker],
		s=1,
		marker="s",
	)
	plt.show()
import ipdb

ipdb.set_trace()
