# Author 
# Abdullah Bdaiwi
# Email: Work: abdullah.bdaiwi@cchmc.org  | Personnal: abdaiwi89@gmail.com
# Cincinnati Children's Hospital Medical Center 
# 
# 
from __future__ import print_function, unicode_literals, absolute_import, division

# Import necessary libraries
import tensorflow as tf
import n2v
import numpy as np
import random
import scipy.fft as fft
import scipy.ndimage as ndimage
import HelpFunctions as HF  # these are local functions I use to upload my data, please create your own functions to read in your data type
import matplotlib.pyplot as plt
from skimage.io import imshow
from csbdeep.utils import plot_history, axes_dict, plot_some
from csbdeep.models import Config, CARE
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import ssl

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Print TensorFlow and N2V versions
print(tf.__version__)
print(n2v.__version__)

# ----------------------------
# Load Training Data
# ----------------------------
DATA_PATH = 'D:/2D_VentData/train'
IM_SIZE = 128
Xeimages, Himages, masks = HF.read_2D_nifti_Xe_H_masks(DATA_PATH, IM_SIZE)
n_channels = 1
images = Xeimages
print(images.shape, images.dtype)

# Display a random image slice
image_x = random.randint(0, images.shape[0])
imshow(images[image_x, :, :, 0])
plt.show()

# ----------------------------
# Remove Zero Signal Slices
# ----------------------------
def remove_slice_without_signal(data, mask):
    S, X, Y, P = data.shape
    images, masks = [], []
    for s in range(S):
        for p in range(P):
            msk = mask[s, :, :, p]
            if np.sum(msk) > 5:
                images.append(data[s, :, :, p])
                masks.append(msk)
    return np.array(images), np.array(masks)

data, masks = remove_slice_without_signal(images, masks)

# Display a random image slice after filtering
image_x = random.randint(0, data.shape[0])
imshow(data[image_x, :, :])
plt.show()

# ----------------------------
# Calculate Signal-to-Noise Ratio (SNR)
# ----------------------------
def calculate_snr(image, mask):
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    dilated_mask = ndimage.binary_dilation(mask, structure=kernel).astype(np.uint8) > 0
    dilated_mask[:, 64 - 15:64 + 15] = 1
    signal_pixels = image[mask == 1]
    noise_pixels = image[dilated_mask == 0]
    return np.mean(signal_pixels) / np.std(noise_pixels), mask

snr_values, snr_masks = [], []
for i in range(data.shape[0]):
    snr, mask = calculate_snr(data[i, :, :], masks[i, :, :])
    snr_values.append(snr)
    snr_masks.append(mask)

snr_values, snr_masks = np.array(snr_values), np.array(snr_masks)

# ----------------------------
# Filter Images Based on SNR
# ----------------------------
filtered_data, filtered_masks = [], []
for i in range(data.shape[0]):
    if snr_values[i] > 20:
        I = (data[i, :, :] - np.min(data[i, :, :])) / (np.max(data[i, :, :]) - np.min(data[i, :, :]))
        filtered_data.append(I)
        filtered_masks.append(snr_masks[i, :, :])

data_SNRfiltered, masks_SNRfiltered = np.array(filtered_data), np.array(filtered_masks)

# Display a filtered image slice
image_x = random.randint(0, data_SNRfiltered.shape[0])
imshow(data_SNRfiltered[image_x, :, :])
plt.show()

# ----------------------------
# Add Noise to Images
# ----------------------------
n_noiselevels, noise_std1, noise_std2, adjust_std = 3, 2, 2.5, 0.5
noisy_patches = np.empty((*data_SNRfiltered.shape, n_noiselevels))
for iteration in range(n_noiselevels):
    print(iteration)
    reconstructed_patches = np.zeros(data_SNRfiltered.shape, dtype=complex)
    if iteration > 0:
        for i in range(data_SNRfiltered.shape[0]):
            k_space = fft.fftn(data_SNRfiltered[i])
            real_noise = np.random.normal(0, noise_std1 + random.uniform(-adjust_std, adjust_std), k_space.shape)
            imag_noise = np.random.normal(0, noise_std1 + random.uniform(-adjust_std, adjust_std), k_space.shape)
            noisy_k_space = k_space + (real_noise + 1j * imag_noise)
            reconstructed_patches[i] = fft.ifftn(noisy_k_space)
        reconstructed_patches = np.abs(reconstructed_patches)
    noise_std1 += noise_std2
    noisy_patches[..., iteration] = reconstructed_patches if iteration > 0 else data_SNRfiltered

print("Noisy Train Images Array Shape:", noisy_patches.shape)

# ----------------------------
# Train the Model
# ----------------------------
train_val_split = int(noisy_patches.shape[0] * 0.85)
x_train, x_val = noisy_patches[:train_val_split], noisy_patches[train_val_split:]
X, Y = x_train[..., 1], x_train[..., 0]
X_val, Y_val = x_val[..., 1], x_val[..., 0]
X, X_val = np.expand_dims(X, -1).astype(np.float32), np.expand_dims(X_val, -1).astype(np.float32)
Y, Y_val = np.expand_dims(Y, -1).astype(np.float32), np.expand_dims(Y_val, -1).astype(np.float32)

config = Config('SXYC', X.shape[-1], Y.shape[-1], unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=50, train_epochs=200)
model = CARE(config, 'VentTrad5NL_2D_e200', basedir='models')

history = model.train(X, Y, validation_data=(X_val, Y_val))
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
plt.show()