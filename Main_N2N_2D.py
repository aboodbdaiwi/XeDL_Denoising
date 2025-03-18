
# Author 
# Abdullah Bdaiwi
# Email: Work: abdullah.bdaiwi@cchmc.org  | Personnal: abdaiwi89@gmail.com
# Cincinnati Children's Hospital Medical Center 
# 
import numpy as np
import random
import scipy.fft as fft
import scipy.ndimage as ndimage
import ssl
import HelpFunctions as HF # these are local functions I use to upload my data, please create your own functions to read in your data type
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from csbdeep.models import CARE, Config
from csbdeep.utils import plot_some, plot_history

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Load training data
DATA_PATH = 'D:/2D_VentData/train'
IM_SIZE = 128
Xeimages, Himages, masks = HF.read_2D_nifti_Xe_H_masks(DATA_PATH, IM_SIZE)
n_channels = 1
images = Xeimages

# Display a random image slice
image_x = random.randint(0, images.shape[0])
imshow(images[image_x, :, :, 10])
plt.show()

# Function to remove slices without signals
def remove_slice_without_signal(data, mask):
    S, X, Y, P = data.shape
    images, masks_filtered = [], []
    for s in range(S):
        for p in range(P):
            if np.sum(mask[s, :, :, p]) > 5:
                images.append(data[s, :, :, p])
                masks_filtered.append(mask[s, :, :, p])
    return np.array(images), np.array(masks_filtered)

# Process data
data, masks = remove_slice_without_signal(images, masks)
image_x = random.randint(0, data.shape[0])
imshow(data[image_x, :, :])
plt.show()

# Function to calculate SNR
def calculate_snr(image, mask):
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    dilated_mask = ndimage.binary_dilation(mask, structure=kernel).astype(np.uint8) > 0
    distance = 15
    dilated_mask[:, 64 - distance:64 + distance] = 1
    signal_pixels = image[mask == 1]
    noise_pixels = image[dilated_mask == 0]
    snr = np.mean(signal_pixels) / np.std(noise_pixels)
    return snr, mask

# Calculate SNR for all slices
snr_values, snr_masks = [], []
for i in range(data.shape[0]):
    snr, mask = calculate_snr(data[i, :, :], masks[i, :, :])
    snr_values.append(snr)
    snr_masks.append(mask)

snr_values, snr_masks = np.array(snr_values), np.array(snr_masks)

# Threshold images based on SNR
data_SNRfiltered, masks_SNRfiltered = [], []
for i in range(data.shape[0]):
    if snr_values[i] > 20:
        data_SNRfiltered.append(data[i, :, :])
        masks_SNRfiltered.append(snr_masks[i, :, :])

data_SNRfiltered, masks_SNRfiltered = np.array(data_SNRfiltered), np.array(masks_SNRfiltered)
image_x = random.randint(0, data_SNRfiltered.shape[0])
imshow(data_SNRfiltered[image_x, :, :])
plt.show()

# Add noise
N_NOISELEVELS = 5
NOISE_STD1, NOISE_STD2, ADJUST_STD = 1, 3, 0.5
patches = data_SNRfiltered
noisy_patches = np.empty((patches.shape[0], patches.shape[1], patches.shape[2], N_NOISELEVELS))

for iteration in range(N_NOISELEVELS):
    print(iteration)
    reconstructed_patches = np.zeros(patches.shape, dtype=complex)
    for i in range(patches.shape[0]):
        k_space = fft.fftn(patches[i])
        random_float = random.uniform(-ADJUST_STD + NOISE_STD1, NOISE_STD1 - ADJUST_STD)
        real_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        imag_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        noisy_k_space = k_space + (real_noise + 1j * imag_noise)
        reconstructed_patches[i] = fft.ifftn(noisy_k_space)
    noisy_patches[:, :, :, iteration] = np.abs(reconstructed_patches)
    NOISE_STD1 += NOISE_STD2

print("Noisy Train Images Array Shape:", noisy_patches.shape)
image_x = random.randint(0, noisy_patches.shape[0])
imshow(noisy_patches[image_x, :, :, 0], cmap='gray')
plt.show()

# Split into training and validation
train_val_split = int(noisy_patches.shape[0] * 0.80)
x_train, x_val = noisy_patches[:train_val_split], noisy_patches[train_val_split:]
print(x_train.shape, x_train.dtype, x_val.shape)

# Prepare paired training data
X, Y = [], []
for patch in x_train:
    for i in range(N_NOISELEVELS):
        for j in range(N_NOISELEVELS):
            if i != j:
                X.append(patch[..., i])
                Y.append(patch[..., j])
X, Y = np.array(X)[..., np.newaxis], np.array(Y)[..., np.newaxis]

X_val, Y_val = [], []
for patch in x_val:
    for i in range(N_NOISELEVELS):
        for j in range(N_NOISELEVELS):
            if i != j:
                X_val.append(patch[..., i])
                Y_val.append(patch[..., j])
X_val, Y_val = np.array(X_val)[..., np.newaxis], np.array(Y_val)[..., np.newaxis]

# Normalize data
mean, std = np.mean(np.concatenate([X, Y], axis=-1)), np.std(np.concatenate([X, Y], axis=-1))
normalize = lambda data, mean, std: (data - mean) / std
X, Y, X_val, Y_val = normalize(X, mean, std), normalize(Y, mean, std), normalize(X_val, mean, std), normalize(Y_val, mean, std)

# Train model
N_EPOCHS = 200
config = Config(axes='YXC', n_channel_in=1, n_channel_out=1, train_loss='mse', unet_kern_size=3, train_steps_per_epoch=100, train_epochs=N_EPOCHS)
model = CARE(config, name=f'N2N_VENT2D_e{N_EPOCHS}', basedir='models')
history = model.train(X, Y, (X_val, Y_val))
plt.figure(figsize=(16, 5))
plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse'])

