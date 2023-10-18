# %%

# !pip install -r requirements.txt

# %%%
import random as rn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.layers import Conv1D, LSTM, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import F1Score
from keras.regularizers import l2, l1

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as anndata
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import keras
import tensorflow as tf


# %%

# set seed for reproducibility
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
# %%
# load data
adata = sc.read_h5ad('Group_7.h5ad')
adata

# %%

# map cell types to a generic cell type mapping
cell_type_mapping = {
    'CD8+/CD45RA+ Naive Cytotoxic': 'T cell',
    'CD14+ Monocyte': 'Monocyte',
    'CD56+ NK': 'NK cell',
    'CD19+ B': 'B cell',
}
adata.obs['cell-types'] = adata.obs['cell-types'].map(cell_type_mapping)
# %%

# calculate QC metrics
# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# Identify ribosomal genes (replace 'RPS' and 'RPL' with the actual prefixes used in your dataset)
adata.var['rb'] = adata.var_names.str.startswith(('RPS', 'RPL'))

# Calculate QC metrics for both mitochondrial and ribosomal genes
sc.pp.calculate_qc_metrics(
    adata, qc_vars=['mt', 'rb'], percent_top=None, log1p=False, inplace=True)

# %%

# preprocess, calculate and filter genes, normalize and scale data, reduce dimensionality
# total genes detected in each cell
total_genes_per_cells = adata.obs['n_genes_by_counts']

median_genes = np.median(total_genes_per_cells)

# Calculate MAD
mad_genes = np.median(np.abs(total_genes_per_cells - median_genes))

# Define lower and upper bounds
lower_bound = median_genes - 3 * mad_genes
upper_bound = median_genes + 3 * mad_genes

# Identify outlier cells
outlier_cells = np.sum((total_genes_per_cells < lower_bound)
                       | (total_genes_per_cells > upper_bound))

print("The number of outlier cells is: ", outlier_cells)
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
sc.pp.filter_genes(adata_filtered, min_cells=3)
sc.pp.normalize_total(adata_filtered, target_sum=None, inplace=True)
sc.pp.log1p(adata_filtered)
sc.pp.highly_variable_genes(
    adata_filtered, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.scale(adata_filtered)
sc.tl.pca(adata_filtered, svd_solver='arpack')
sc.pp.neighbors(adata_filtered, n_neighbors=15, n_pcs=40)
# Embedding the neighborhood graph
sc.tl.umap(adata_filtered)

# %%
X = adata_filtered.obsm["X_umap"]  # features (PCA components)
y = adata_filtered.obs['cell-types']  # labels (cell types)

# %%

# label encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, random_state=42)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# %%

# CNN model


def build_model(hp):
    # Define the model
    model = Sequential()
    hp_neurons = hp.Int('neurons', min_value=2, max_value=12, step=1)
    model.add(Conv1D(filters=5, kernel_size=2,
              activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.25), bias_regularizer=l2(0.25), input_shape=(X_train.shape[1], 1), padding="same"))
    model.add(Conv1D(filters=5, kernel_size=2, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(
        0.25), bias_regularizer=l2(0.25), input_shape=(X_train.shape[1], 1), padding="same"))
    model.add(MaxPooling1D(pool_size=1))  # Added MaxPooling layer
    model.add(BatchNormalization())  # Added batch normalization layer
    model.add(Flatten())
    model.add(Dropout(0.1))  # Added dropout layer

    # Assuming 'y' is categorical
    model.add(Dense(hp_neurons, activation='relu',
              kernel_regularizer=l2(0.25), bias_regularizer=l2(0.25)))
    # Assuming 'y' is categorical
    model.add(Dense(hp_neurons//2, activation='relu',
              kernel_regularizer=l2(0.25), bias_regularizer=l2(0.25)))
    model.add(BatchNormalization())  # Added batch normalization layer
    model.add(Dropout(0.1))  # Added dropout layer
    model.add(Dense(y.nunique(), activation='softmax',
              kernel_regularizer=l2(0.25), bias_regularizer=l2(0.25)))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=0.001), loss='categorical_crossentropy', metrics=F1Score(average='macro'))

    return model

# %%

# uncomment to perform hyperparameter tuning


# hyperparameter search and k fold cross validation with no trial 3
tuner = kt.BayesianOptimization(
    build_model, objective='loss', max_trials=3, seed=42)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
tuner.search(X_train, y_train, epochs=50, validation_split=0.2,
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# %%

# uncomment for hyperparameter search and k fold cross validation with no trial 3

# model = tuner.hypermodel.build(best_hps)
# history = model.fit(X_train, y_train, epochs=50, batch_size=16,
#                     validation_split=0.2, callbacks=stop_early)

# val_acc_per_epoch = history.history['val_loss']
# best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))
best_epoch = 12

# %%

# uncomment them if you run hyperparameter search and k fold cross validation with no trial 3

# Plot training & validation accuracy values
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['f1_score'])
# plt.plot(history.history['val_f1_score'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# plt.tight_layout()
# plt.show()

# %%
# Now retrain the model with the best epochs and params
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch, batch_size=16)

# %%
# Make predictions on the test set
y_pred = hypermodel.predict(X_test)
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

classes = [label.split('_')[1] for label in encoder.get_feature_names_out()]

# Generate confusion matrix
cm = confusion_matrix(y_test_label, y_pred_label)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sbn.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
# %%
# Now test on unseen data
test_adata = sc.read_h5ad('Test_dataset.h5ad')

# %%

# Preprocess unseen data and filter genes
test_adata.var['mt'] = test_adata.var_names.str.startswith('MT-')

# Identify ribosomal genes (replace 'RPS' and 'RPL' with the actual prefixes used in your dataset)
test_adata.var['rb'] = test_adata.var_names.str.startswith(('RPS', 'RPL'))

# Calculate QC metrics for both mitochondrial and ribosomal genes
sc.pp.calculate_qc_metrics(
    test_adata, qc_vars=['mt', 'rb'], percent_top=None, log1p=False, inplace=True)
total_genes_per_cells = test_adata.obs['n_genes_by_counts']

median_genes = np.median(total_genes_per_cells)

# Calculate MAD
mad_genes = np.median(np.abs(total_genes_per_cells - median_genes))

# Define lower and upper bounds
lower_bound = median_genes - 3 * mad_genes
upper_bound = median_genes + 3 * mad_genes

# Identify outlier cells
outlier_cells = np.sum((total_genes_per_cells < lower_bound)
                       | (total_genes_per_cells > upper_bound))

print("The number of outlier cells is: ", outlier_cells)

# Before filtering cells
sbn.distplot(test_adata.obs.n_genes_by_counts)
sbn.rugplot(test_adata.obs.n_genes_by_counts)
# After filtering cells
test_adata_filtered = test_adata[test_adata.obs.n_genes_by_counts > lower_bound, :]
test_adata_filtered = test_adata_filtered[test_adata_filtered.obs.n_genes_by_counts < upper_bound, :]
print('Number of cells after filtering cells: ', test_adata_filtered.shape[0])

# Check the effect of data preprocessing, notice the change to the normal distribution
sbn.distplot(test_adata_filtered.obs.n_genes_by_counts)
sbn.rugplot(test_adata_filtered.obs.n_genes_by_counts)

sc.pp.filter_genes(test_adata_filtered, min_cells=3)

# %%

# Map to a general cell type
cell_types_to_filter = ['NK cell', 'B cell', 'CD8+ T cell', 'CD14+ Monocyte',
                        'CD4+ T cell', 'CD16+ Monocyte', 'Plasmablast', 'Other T']
mask = test_adata_filtered.obs['cell-types'].isin(cell_types_to_filter)
test_adata_filtered = test_adata_filtered[mask]

cell_type_mapping = {
    'CD8+ T cell': 'T cell',
    'CD4+ T cell': 'T cell',
    'Other T': 'T cell',
    'CD14+ Monocyte': 'Monocyte',
    'CD16+ Monocyte': 'Monocyte',
    'NK cell': 'NK cell',
    'B cell': 'B cell',
    'Plasmablast': 'B cell'
}

test_adata_filtered.obs['cell-types'] = test_adata_filtered.obs['cell-types'].map(
    cell_type_mapping)
# %%
# Transform data
X_unseen = test_adata_filtered.obsm["X_umap"]  # features (PCA components)
y_unseen = test_adata_filtered.obs['cell-types']  # labels (cell types)
y_onehot_unseen = encoder.transform(y_unseen.values.reshape(-1, 1))
# Reshape the data to be compatible with a 1D CNN
X_unseen = X_unseen.reshape((X_unseen.shape[0], X_unseen.shape[1], 1))
# %%
# Predict unseen data
y_unseen_pred = hypermodel.predict(X_unseen)

# Convert one-hot encoded test label to label encoded
y_unseen_label = np.argmax(y_onehot_unseen, axis=1)
original_labels = encoder.inverse_transform(y_onehot_unseen)
original_labels_pred = encoder.inverse_transform(y_unseen_pred)
print(original_labels)

# Convert one-hot encoded predictions to label encoded
y_unseen_pred_label = np.argmax(y_unseen_pred, axis=1)

# Calculate Accuracy
accuracy = accuracy_score(original_labels, original_labels_pred)
print(f'Accuracy: {accuracy}')

# Calculate F1 Score
f1 = f1_score(y_unseen_label, y_unseen_pred_label, average='macro')
print(f'F1 Score: {f1}')
# %%
cm = confusion_matrix(original_labels, original_labels_pred,)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sbn.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()

# %%
hypermodel.summary()
# %%