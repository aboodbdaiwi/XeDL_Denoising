
# **Denoising 2D multi-channel  images using Noise2Void deep learning approach**

# Install the tensorflow library suggested by N2V. 
# pip install tensorflow==2.4

# Install N2V
# pip install n2v

# Author 
# Abdullah Bdaiwi
# Email: Work: abdullah.bdaiwi@cchmc.org  | Personnal: abdaiwi89@gmail.com
# Cincinnati Children's Hospital Medical Center 

# import packages 
import tensorflow as tf
import n2v 
import numpy as np
import random
import scipy.fft as fft
import ssl
import HelpFunctions as HF # these are local functions I use to upload my data, please create your own functions to read in your data type
from csbdeep.utils import plot_history
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from skimage.io import imread, imshow
from matplotlib import pyplot as plt

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Print versions
print(tf.__version__)
print(n2v.__version__)

# Load training data
DATA_PATH = 'D:/2D_VentData/train'
IM_SIZE = 128
Xeimages, Himages, masks = HF.read_2D_nifti_Xe_H_masks(DATA_PATH, IM_SIZE)
n_channels = 1
images = np.stack((Xeimages,) * n_channels, axis=-1).astype(np.float32)

# Display a random image slice
image_x = random.randint(0, images.shape[0])
imshow(images[image_x, :, :, 10])
plt.show()

# Function to remove slices without signals
def remove_slice_without_signal(data, mask):
    S, X, Y, Z, C = data.shape
    images = []
    for s in range(S):
        for ss in range(Z):
            if np.sum(mask[s, :, :, ss]) > 5:
                images.append(data[s, :, :, ss, :])
    return np.array(images)

# Process data
data = remove_slice_without_signal(images, masks)
image_x = random.randint(0, data.shape[0])
imshow(data[image_x, :, :, 0])
plt.show()

# Generate patches
datagen = N2V_DataGenerator()
PATCH_SIZE = 128
patches = datagen.generate_patches_from_list(data, shape=(PATCH_SIZE, PATCH_SIZE))
print(patches.shape)

# Add noise and split into training/validation
N_NOISELEVELS = 3
NOISE_STD1 = 1
NOISE_STD2 = 6
ADJUST_STD = 1
patches = patches[:, :, :, 0]
noisy_patches = np.empty((patches.shape[0], patches.shape[1], patches.shape[2], N_NOISELEVELS))

for iteration in range(N_NOISELEVELS):
    reconstructed_patches = np.zeros(patches.shape, dtype=complex)
    for i in range(patches.shape[0]):
        k_space = fft.fftn(patches[i])
        random_float = random.uniform(-ADJUST_STD + NOISE_STD1, NOISE_STD1 - ADJUST_STD)
        real_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        imag_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        noisy_k_space = k_space + (real_noise + 1j * imag_noise)
        reconstructed_patches[i] = fft.ifftn(noisy_k_space)
    
    reconstructed_patches = np.abs(reconstructed_patches)
    NOISE_STD1 += NOISE_STD2
    noisy_patches[:, :, :, iteration] = reconstructed_patches

print("Noisy Train Images Array Shape:", noisy_patches.shape)
noisy_patches = np.clip(noisy_patches, 0, 1)

# Split data into training and validation
train_val_split = int(noisy_patches.shape[0] * 0.80)
x_train = noisy_patches[:train_val_split]
x_val = noisy_patches[train_val_split:]

X = np.stack((x_train[:, :, :, 1],) * n_channels, axis=-1).astype(np.float32)
X_val = np.stack((x_val[:, :, :, 1],) * n_channels, axis=-1).astype(np.float32)

# Display training and validation patches
image_x = random.randint(0, X_val.shape[0])
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(X[image_x, :, :, 0])
plt.title('Training Patch')
plt.subplot(1, 2, 2)
plt.imshow(X_val[image_x, :, :, 0])
plt.title('Validation Patch')
plt.show()

# Model configuration
TRAIN_BATCH = 128
NUM_EPOCHS = 200
config = N2VConfig(X, unet_kern_size=3, unet_n_first=64, unet_n_depth=3, 
                   train_steps_per_epoch=int(X.shape[0] / TRAIN_BATCH), train_epochs=NUM_EPOCHS, 
                   train_loss='mse', batch_norm=True, train_batch_size=TRAIN_BATCH, 
                   n2v_perc_pix=0.198, n2v_patch_shape=(PATCH_SIZE, PATCH_SIZE), 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, 
                   single_net_per_channel=False)

MODEL_NAME = f'n2v_2D_Vent_128p{NUM_EPOCHS}e'
BASEDIR = 'models'
model = N2V(config, MODEL_NAME, basedir=BASEDIR)

# Train model
history = model.train(X, X_val)
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ['loss', 'val_loss'])

# Load test data
TEST_MODE = False
test_data_path = 'D:/2D_VentData/test' if TEST_MODE else 'D:/2D_VentData/test_temp'

test_Xeimages, test_Himages, test_masks = HF.read_2D_nifti_Xe_H_masks(test_data_path, IM_SIZE)
test_images = test_Xeimages

print(test_images.shape)
print(test_images.dtype)

# Display test images
sub_x = random.randint(0, test_images.shape[0] - 1)
slice_x = random.randint(0, test_images.shape[3] - 10)
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.imshow(test_images[sub_x, :, :, slice_x], cmap='inferno')
plt.subplot(1, 3, 2)
plt.imshow(test_masks[sub_x, :, :, slice_x], cmap='inferno')

# Denoising process
N2Vmodel = N2V(config=None, name=MODEL_NAME, basedir=BASEDIR)
for n in range(test_images.shape[0]):
    N_images = test_images[n]
    de_N2V = np.zeros_like(N_images)
    for sli in range(test_images.shape[3]):
        if np.sum(test_images[n, :, :, sli]) > 0:
            sli_imag = (N_images[:, :, sli] - np.min(N_images[:, :, sli])) / (np.max(N_images[:, :, sli]) - np.min(N_images[:, :, sli]))
            sli_imag = np.stack((sli_imag,) * n_channels, axis=-1).astype(np.float32)
            de_N2V[:, :, sli] = N2Vmodel.predict(sli_imag, axes='YXC')[:, :, 0]

# Display results
SLICE_X = 7
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(test_images[0, :, :, SLICE_X], cmap='gray')
plt.title('Input')
plt.subplot(1, 2, 2)
plt.imshow(de_N2V[:, :, SLICE_X], cmap='gray')
plt.title('N2V')
plt.show()
