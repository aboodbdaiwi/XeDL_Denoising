
# Author 
# Abdullah Bdaiwi
# Email: Work: abdullah.bdaiwi@cchmc.org  | Personnal: abdaiwi89@gmail.com
# Cincinnati Children's Hospital Medical Center 
# 
import numpy as np
import random
import scipy.fft as fft
import HelpFunctions as HF   # these are local functions I use to upload my data, please create your own functions to read in your data type
import bm3d
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage.io import imshow
from csbdeep.models import CARE, Config
from csbdeep.utils import plot_history
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

# Load training data
DATA_PATH = 'D:/3D_GasExData/train'
IM_SIZE = 112
data, _, _, _, _, _, masks = HF.read_3D_nifti_Xe_H_masks(DATA_PATH, IM_SIZE)
data = np.stack((data,), axis=-1).astype(np.float32)
print(data.shape, data.dtype)

# Display a random image slice
sub_x = random.randint(0, data.shape[0] - 1)
image_x = random.randint(40, data.shape[1] - 40)
imshow(data[sub_x, :, :, image_x])
plt.show()

# Calculate SNR
snr_values = [HF.calculate_snr_3D(data[i], masks[i])[0] for i in range(data.shape[0])]
snr_values = np.array(snr_values)

# Threshold images based on SNR
data_SNRfiltered, mask_SNRfiltered = [], []
for i in range(data.shape[0]):
    if snr_values[i] > 10:
        data_SNRfiltered.append(data[i])
        mask_SNRfiltered.append(masks[i])

data, masks = np.array(data_SNRfiltered), np.array(mask_SNRfiltered)

# Display an image slice
image_x = random.randint(0, data.shape[0] - 1)
imshow(data[image_x, :, :, 50])
plt.show()

# Generate patches
datagen = N2V_DataGenerator()
patch_size = 112
patch_shape = (patch_size, patch_size, 1)
patches = datagen.generate_patches_from_list(data, shape=patch_shape)
print(patches.shape)

# Add noise
N_NOISELEVELS, NOISE_STD1, NOISE_STD2, ADJUST_STD = 4, 5, 5, 4
noisy_patches = np.empty((*patches.shape, N_NOISELEVELS))

for iteration in range(N_NOISELEVELS):
    print(iteration)
    reconstructed_patches = np.zeros(patches.shape, dtype=complex)
    for i in range(patches.shape[0]):
        Npatches = (patches[i] - np.min(patches[i])) / (np.max(patches[i]) - np.min(patches[i]))
        k_space = fft.fftn(Npatches)
        random_float = random.uniform(-ADJUST_STD + NOISE_STD1, NOISE_STD1 - ADJUST_STD)
        real_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        imag_noise = np.random.normal(0, NOISE_STD1 + random_float, k_space.shape)
        noisy_k_space = k_space + (real_noise + 1j * imag_noise)
        reconstructed_patches[i] = fft.ifftn(noisy_k_space)
    noisy_patches[..., iteration] = np.abs(reconstructed_patches)
    NOISE_STD1 += NOISE_STD2

print("Noisy Train Images Array Shape:", noisy_patches.shape)

# Split into training and validation
train_val_split = int(noisy_patches.shape[0] * 0.80)
x_train, x_val = noisy_patches[:train_val_split], noisy_patches[train_val_split:]
del noisy_patches

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
config = Config(axes='SYXZC', n_channel_in=1, n_channel_out=1, train_loss='mse', unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=16, train_epochs=N_EPOCHS)
model = CARE(config, name=f'N2N_VentGas3D_e{N_EPOCHS}', basedir='models')
history = model.train(X, Y, (X_val, Y_val))
plt.figure(figsize=(16, 5))
plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse'])

