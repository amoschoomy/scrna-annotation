# %%
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as anndata
import seaborn as sbn
import matplotlib.pyplot as plt
# %%
adata = sc.read_h5ad('Group_6.h5ad')
adata
# %%
# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# Identify ribosomal genes (replace 'RPS' and 'RPL' with the actual prefixes used in your dataset)
adata.var['rb'] = adata.var_names.str.startswith(('RPS', 'RPL'))

# Calculate QC metrics for both mitochondrial and ribosomal genes
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'rb'], percent_top=None, log1p=False, inplace=True)
# %%
adata.obs
# %%
adata.var
# %%
# Calculate Pearson correlation
from scipy.stats import pearsonr

correlation, _ = pearsonr(adata.obs['total_counts'], adata.obs['n_genes_by_counts'])

print('Pearson correlation: %.3f' % correlation)
# %%
total_genes_per_cells = adata.obs['n_genes_by_counts']  # total genes detected in each cell
total_genes_per_cells
# %%
median_genes = np.median(total_genes_per_cells)

# Calculate MAD
mad_genes = np.median(np.abs(total_genes_per_cells - median_genes))

# Define lower and upper bounds
lower_bound = median_genes - 3 * mad_genes
upper_bound = median_genes + 3 * mad_genes

# Identify outlier cells
outlier_cells = np.sum((total_genes_per_cells < lower_bound) | (total_genes_per_cells > upper_bound))

print("The number of outlier cells is: ", outlier_cells)

# %%
# Before filtering cells 
sbn.distplot(adata.obs.n_genes_by_counts)
sbn.rugplot(adata.obs.n_genes_by_counts)

# After filtering cells 
adata_filtered = adata[adata.obs.n_genes_by_counts > lower_bound, :]
adata_filtered = adata_filtered[adata_filtered.obs.n_genes_by_counts < upper_bound, :]
print('Number of cells after filtering cells: ', adata_filtered.shape[0])

# Check the effect of data preprocessing, notice the change to the normal distribution 
sbn.distplot(adata_filtered.obs.n_genes_by_counts)
sbn.rugplot(adata_filtered.obs.n_genes_by_counts)
# %%
print(adata_filtered.shape[1])
sc.pp.filter_genes(adata_filtered, min_cells=3)
print(adata_filtered.shape[1])
# %%

sc.pp.normalize_total(adata_filtered, target_sum=None, inplace=True)

# %%

# Log transform the data
adata_filtered
sc.pp.log1p(adata_filtered)
# %%
# Filter for highly variable gene
sc.pp.highly_variable_genes(adata_filtered, min_mean=0.0125, max_mean=3, min_disp=0.5)
# %%
sc.pp.scale(adata_filtered)

# %%
sc.tl.pca(adata_filtered, svd_solver='arpack')
# %%
sc.pp.neighbors(adata_filtered, n_neighbors=15, n_pcs=40)
# Embedding the neighborhood graph
sc.tl.umap(adata_filtered)
# %%
