# Author 
# Abdullah Bdaiwi
# Email: Work: abdullah.bdaiwi@cchmc.org  | Personnal: abdaiwi89@gmail.com
# Cincinnati Children's Hospital Medical Center 
# 
import tensorflow as tf
import n2v
import numpy as np
import random
import scipy.fft as fft
import HelpFunctions as HF   # these are local functions I use to upload my data, please create your own functions to read in your data type
import matplotlib.pyplot as plt
from skimage.io import imshow
from n2v.models import N2VConfig, N2V
from csbdeep.utils import plot_history
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import ssl

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Print versions
print(tf.__version__)
print(n2v.__version__)

# Load training data
DATA_PATH = 'D:/3D_GasExData/train'
IM_SIZE = 112
Xeimages, _, _, _, _, HHRimages, masks = HF.read_3D_nifti_Xe_H_masks(DATA_PATH, IM_SIZE)
data = np.stack((HHRimages,), axis=-1).astype(np.float32)
print(data.shape, data.dtype)

# Display a random image slice
sub_x = random.randint(0, data.shape[0] - 1)
image_x = random.randint(40, data.shape[1] - 40)
imshow(data[sub_x, :, :, image_x, 0])
plt.show()

# Generate patches
datagen = N2V_DataGenerator()
patch_size = 64
patch_shape = (patch_size, patch_size, 8)
patches = datagen.generate_patches_from_list(data, shape=patch_shape)
print(patches.shape)

# Add noise
N_NOISELEVELS, NOISE_STD1, NOISE_STD2, NOISE_STD_CHANGE = 2, 5, 1, 1
patches = patches[:, :, :, :, 0]
noisy_patches = np.empty((*patches.shape, N_NOISELEVELS))

for iteration in range(N_NOISELEVELS):
    print(iteration)
    reconstructed_patches = np.zeros(patches.shape, dtype=complex)
    for i in range(patches.shape[0]):
        k_space = fft.fftn(patches[i])
        random_float = random.uniform(-NOISE_STD_CHANGE + NOISE_STD1, NOISE_STD1 - NOISE_STD_CHANGE)
        real_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        imag_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        noisy_k_space = k_space + (real_noise + 1j * imag_noise)
        reconstructed_patches[i] = fft.ifftn(noisy_k_space)
    
    reconstructed_patches = np.abs(reconstructed_patches)
    NOISE_STD1 += NOISE_STD2
    noisy_patches[..., iteration] = reconstructed_patches

print("Noisy Train Images Array Shape:", noisy_patches.shape)
noisy_patches = np.clip(noisy_patches, 0, 1)

# Display noisy images
image_x = random.randint(0, noisy_patches.shape[0])
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(noisy_patches[image_x, :, :, 0, 0])
plt.title('Level 1 noise')
plt.subplot(1, 2, 2)
plt.imshow(noisy_patches[image_x, :, :, 0, 1])
plt.title('Level 2 noise')
plt.show()

# Split into training and validation
n_channels = 1
train_val_split = int(noisy_patches.shape[0] * 0.80)
x_train, x_val = noisy_patches[:train_val_split], noisy_patches[train_val_split:]
X, X_val = x_train[:, :, :, :, 1], x_val[:, :, :, :, 1]
X, X_val = np.stack((X,), axis=-1).astype(np.float32), np.stack((X_val,), axis=-1).astype(np.float32)

print(X.shape, X.dtype, X_val.shape)

# Display patches
image_x = random.randint(0, X_val.shape[0])
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(X[image_x, :, :, 0, 0])
plt.title('Training Patch')
plt.subplot(1, 2, 2)
plt.imshow(X_val[image_x, :, :, 0, 0])
plt.title('Validation Patch')
plt.show()

# Train model
patch_size = 64
train_batch = 4
epochs = 200
config = N2VConfig(X, unet_kern_size=3, unet_n_first=64, unet_n_depth=3, train_steps_per_epoch=int(X.shape[0] / 128),
                   train_epochs=epochs, train_loss='mse', batch_norm=True, train_batch_size=train_batch,
                   n2v_perc_pix=0.198, n2v_patch_shape=(patch_size, patch_size, 8), n2v_manipulator='uniform_withCP',
                   n2v_neighborhood_radius=5, single_net_per_channel=False)

model_name = f'n2v_3D_GasExHHR_{epochs}e'
model = N2V(config, model_name, basedir='models')
history = model.train(X, X_val)

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ['loss', 'val_loss'])
plt.show()
