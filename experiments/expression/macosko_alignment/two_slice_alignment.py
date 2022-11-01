#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import join as pjoin
from scipy.io import mmread
import scanpy as sc
import anndata
import squidpy as sq
from gpsa import VariationalGPSA, rbf_kernel
import torch
import sys
sys.path.append("../../../../paste")
from src.paste import PASTE, visualization
import scanpy as sc


# In[2]:


def rotate_90deg_counterclockwise(coords):
    y_coords_reflected = -coords[:, 1] - coords[:, 1].max()
    return np.vstack([y_coords_reflected, coords[:, 0]]).T


# In[3]:


DATA_DIR = "../../../data/mouse_brain_slideseq/12_allMTXs_CCF/"


# ## Load data

# In[4]:


gene_list = pd.read_table(pjoin(DATA_DIR, "01_Gene_List.txt"), header=None).values.squeeze()

metadata_slice1 = pd.read_table(pjoin(DATA_DIR, "MBASS_d1_metadata.tsv"))
metadata_slice2 = pd.read_table(pjoin(DATA_DIR, "MBASS_d3_metadata.tsv"))

barcodes_slice1 = pd.read_table(pjoin(DATA_DIR, "MBASS_d1_barcodes.txt"), header=None).values.squeeze()
barcodes_slice2 = pd.read_table(pjoin(DATA_DIR, "MBASS_d3_barcodes.txt"), header=None).values.squeeze()

data_sparse_slice1 = mmread(pjoin(DATA_DIR, "MBASS_d1_matrix.mtx"))
data_sparse_slice2 = mmread(pjoin(DATA_DIR, "MBASS_d3_matrix.mtx"))

data_slice1 = pd.DataFrame(data_sparse_slice1.toarray(), index=gene_list, columns=barcodes_slice1)
data_slice2 = pd.DataFrame(data_sparse_slice2.toarray(), index=gene_list, columns=barcodes_slice2)


# ## Manually rotate one slice so they have same orientation

# In[19]:


rotated_coords_slice1 = rotate_90deg_counterclockwise(rotate_90deg_counterclockwise(rotate_90deg_counterclockwise(metadata_slice1[["Original_x", "Original_y"]].values)))
X_slice1 = rotated_coords_slice1[~metadata_slice1.isOutsideCCF]
X_slice2 = metadata_slice2.loc[~metadata_slice2.isOutsideCCF, ["Original_x", "Original_y"]].values


# In[20]:


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_slice1[:, 0], X_slice1[:, 1], s=.05) #, c=np.log(Y_slice1 + 1))
plt.title("Slice 1")
# plt.axis("off")

plt.subplot(122)
plt.scatter(X_slice2[:, 0], X_slice2[:, 1], s=.05) #, c=np.log(Y_slice2 + 1))
plt.title("Slice 2")
# plt.axis("off")
plt.show()


# ## Filter for spots inside the tissue and create AnnData objects

# In[21]:


n_spots = 10_000
n_genes = 10_000

in_slice_idx = np.where(~metadata_slice1.isOutsideCCF.values)[0]
in_slice_idx = np.random.choice(in_slice_idx, size=n_spots, replace=False)

X_slice1 = rotated_coords_slice1[in_slice_idx]
Y_slice1 = data_slice1.transpose().iloc[in_slice_idx, :n_genes]
Y_slice1 = Y_slice1.loc[:, Y_slice1.sum(0) > 0]
anndata_slice1 = anndata.AnnData(Y_slice1)
anndata_slice1.obsm['spatial'] = X_slice1

in_slice_idx = np.where(~metadata_slice2.isOutsideCCF.values)[0]
in_slice_idx = np.random.choice(in_slice_idx, size=n_spots, replace=False)

X_slice2 = metadata_slice2[["Original_x", "Original_y"]].values[in_slice_idx]
Y_slice2 = data_slice2.transpose().iloc[in_slice_idx, :n_genes]
Y_slice2 = Y_slice2.loc[:, Y_slice2.sum(0) > 0]
anndata_slice2 = anndata.AnnData(Y_slice2)
anndata_slice2.obsm['spatial'] = X_slice2


# ## Compute spatial autocorrelation (Moran's I) for each gene

# In[22]:


sq.gr.spatial_neighbors(anndata_slice1)
sq.gr.spatial_autocorr(
    anndata_slice1,
    mode="moran",
)

sq.gr.spatial_neighbors(anndata_slice2)
sq.gr.spatial_autocorr(
    anndata_slice2,
    mode="moran",
)


# In[23]:


plt.figure(figsize=(10, 5))
plt.subplot(121)
moran_genenames_sorted_slice1 = anndata_slice1.uns["moranI"].index.values #.astype(int)
plt.hist(anndata_slice1.uns["moranI"].I.values, 30)
plt.xlabel("Moran's I")
plt.ylabel("Count")
plt.title("Slice 1")
plt.xlim([-0.025, 0.35])

plt.subplot(122)
moran_genenames_sorted_slice2 = anndata_slice2.uns["moranI"].index.values #.astype(int)
plt.hist(anndata_slice2.uns["moranI"].I.values, 30)
plt.xlabel("Moran's I")
plt.ylabel("Count")
plt.title("Slice 2")
plt.xlim([-0.025, 0.35])
plt.show()


# In[24]:


gene_idx_to_plot = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(anndata_slice1.obsm["spatial"][:, 0], anndata_slice1.obsm["spatial"][:, 1], c=np.log(anndata_slice1[:, moran_genenames_sorted_slice1[gene_idx_to_plot]].X.squeeze() + 1), s=5)
plt.title("Slice 1")
plt.axis("off")

plt.subplot(122)
plt.scatter(anndata_slice2.obsm["spatial"][:, 0], anndata_slice2.obsm["spatial"][:, 1], c=np.log(anndata_slice2[:, moran_genenames_sorted_slice1[gene_idx_to_plot]].X.squeeze() + 1), s=5)
plt.title("Slice 1")
plt.axis("off")
plt.show()


# In[25]:


genenames_to_keep = np.intersect1d(moran_genenames_sorted_slice1[:30], moran_genenames_sorted_slice2[:30])
anndata_slice1 = anndata_slice1[:, genenames_to_keep]
anndata_slice2 = anndata_slice2[:, genenames_to_keep]


# ## Start alignment

# In[14]:


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


# In[15]:


n_views = 2
anndata_all_slices = anndata_slice1.concatenate(anndata_slice2)
n_samples_list = [anndata_all_slices[anndata_all_slices.obs.batch == str(ii)].shape[0] for ii in range(n_views)]

X1 = np.array(anndata_all_slices[anndata_all_slices.obs.batch == "0"].obsm["spatial"])
X2 = np.array(anndata_all_slices[anndata_all_slices.obs.batch == "1"].obsm["spatial"])
Y1 = np.log(np.array(anndata_all_slices[anndata_all_slices.obs.batch == "0"].X) + 1)
Y2 = np.log(np.array(anndata_all_slices[anndata_all_slices.obs.batch == "1"].X) + 1)

Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X1 = scale_spatial_coords(X1)
X2 = scale_spatial_coords(X2)
X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

view_idx = [
    np.arange(X1.shape[0]),
    np.arange(X1.shape[0], X1.shape[0] + X2.shape[0]),
]

x = torch.from_numpy(X).float().clone()
y = torch.from_numpy(Y).float().clone()


data_dict = {
    "expression": {
        "spatial_coords": x,
        "outputs": y,
        "n_samples_list": n_samples_list,
    }
}


# In[16]:


n_spatial_dims = 2
m_X_per_view = 20
m_G = 20
N_LATENT_GPS = {"expression": None}
N_EPOCHS = 10
PRINT_EVERY = 10


# In[17]:


device = "cuda" if torch.cuda.is_available() else "cpu"

model = VariationalGPSA(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    fixed_warp_kernel_variances=np.ones(n_views) * 1e-3,
    # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    fixed_view_idx=0,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means


# In[18]:


for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0 or t == N_EPOCHS - 1:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

aligned_coords = G_means["expression"].detach().numpy()
aligned_coords_slice1 = aligned_coords[view_idx["expression"][0]]
aligned_coords_slice2 = aligned_coords[view_idx["expression"][1]]


# In[ ]:


plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.scatter(X1[:, 0], X1[:, 1], c=Y1[:, 0], s=5)
plt.title("Slice 1")
# plt.axis("off")

plt.subplot(222)
plt.scatter(X2[:, 0], X2[:, 1], c=Y2[:, 0], s=5)
plt.title("Slice 1")
# plt.axis("off")

plt.subplot(223)
plt.scatter(aligned_coords_slice1[:, 0], aligned_coords_slice1[:, 1], c=Y1[:, 0], s=5)
plt.title("Slice 1")
# plt.axis("off")

plt.subplot(224)
plt.scatter(aligned_coords_slice2[:, 0], aligned_coords_slice2[:, 1], c=Y2[:, 0], s=5)
plt.title("Slice 1")
# plt.axis("off")
plt.show()


# In[ ]:


pi12 = PASTE.pairwise_align(anndata_slice1, anndata_slice2, alpha=0.1)


# In[1]:


slices = [anndata_slice1, anndata_slice2]
pis = [pi12]
new_slices = visualization.stack_slices_pairwise(slices, pis)


# In[32]:


new_slices


# In[ ]:




