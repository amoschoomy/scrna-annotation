# %%
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as anndata
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from natsort import natsorted
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import keras
import tensorflow as tf

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
total_genes_per_cells = adata.obs['n_genes_by_counts']  # total genes detected in each cell

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
sc.pp.log1p(adata_filtered)

# %%
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
X = adata_filtered.obsm["X_pca"]  # features (PCA components)
y = adata_filtered.obs['cell-types']  # labels (cell types)

# %%
from sklearn.preprocessing import OneHotEncoder

# Assuming 'y' is your target labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.33, random_state=42)

# Reshape the data to be compatible with a 1D CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


def build_model(hp):
    # Define the model
    model = Sequential()
    hp_filters = hp.Int('filters', min_value=4, max_value=64, step=4)
    hp_kernel_size = hp.Int('kernel_size', min_value=2, max_value=5, step=1)
    hp_pool_size = hp.Int('pool_size', min_value=2, max_value=5, step=1)
    hp_neurons = hp.Int('neurons', min_value=4, max_value=64, step=4)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=hp_pool_size))
    model.add(Flatten())
    model.add(Dense(hp_neurons, activation='relu'))
    model.add(Dense(y.nunique(), activation='softmax'))  # Assuming 'y' is categorical

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    # # Train the model
    # model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=2)
#%%
tuner = kt.BayesianOptimization(build_model, objective='val_accuracy', max_trials=5, seed=42)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=50, validation_split=0.3, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# %%
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.3)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
# %%
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.3)
#%%
# Make predictions on the test set
y_pred = hypermodel.predict(X_test)


# %%
y_pred
# %%
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Convert one-hot encoded test label to label encoded
y_test_label = np.argmax(y_test, axis=1)

# Convert one-hot encoded predictions to label encoded
y_pred_label = np.argmax(y_pred, axis=1)

# Calculate Accuracy
accuracy = accuracy_score(y_test_label, y_pred_label)
print(f'Accuracy: {accuracy}')

# Calculate F1 Score
f1 = f1_score(y_test_label, y_pred_label, average='macro')
print(f'F1 Score: {f1}')

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test_label, y_pred_label)

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
# %%