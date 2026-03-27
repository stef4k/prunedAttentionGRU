import random
import numpy as np
import torch

def add_gaussian_noise(X_train, y_train, num_noise_copies=3):

    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    _, sequence_length, hid_dim = X_train_np.shape
    # #fix the random seed
    seed = 1
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    X_train_tensor = torch.as_tensor(X_train_np, dtype=torch.float32)
    if num_noise_copies <= 0:
        return X_train_tensor, y_train_tensor

    # #Gaussian noise add
    noise_mu, noise_sigma = 0.0, 1.0
    X_train_flat = X_train_np.reshape(-1, hid_dim)
    noisy_tensors = [X_train_tensor]
    for _ in range(num_noise_copies):
        noise = np.random.normal(noise_mu, noise_sigma, X_train_flat.shape).astype(np.float32, copy=False)
        noisy_data = X_train_flat + X_train_flat * noise
        noisy_tensors.append(
            torch.as_tensor(noisy_data.reshape(-1, sequence_length, hid_dim), dtype=torch.float32)
        )

    X_train_gn = torch.cat(noisy_tensors, dim=0)
    y_train_gn = torch.cat([y_train_tensor for _ in range(num_noise_copies + 1)], dim=0)
    return X_train_gn, y_train_gn

def shifting(data, shift_steps, axis=1):
    # 3차원 데이터를 shift_steps만큼 이동시킵니다.
    shifted_data = np.roll(data, shift=shift_steps, axis=axis)
    return shifted_data

def augment_labels(labels, num_shifts):
    return np.tile(labels, (num_shifts, 1))

def shift(X_train, y_train, shifts=None):
    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    augmented_data_fill = []
    if shifts is None:
        shifts = range(-10, 10)
    shifts = list(shifts)

    if not shifts:
        return X_train_np, y_train_np

    for shift_step in shifts:
        shifted_data_fill = shifting(X_train_np, shift_step)
        augmented_data_fill.append(shifted_data_fill)

    # Combining all augmented data
    X_train_sh = np.concatenate(augmented_data_fill, axis=0).astype(np.float32, copy=False)
    num_shifts = len(shifts)

    # Augmenting the labels
    y_train_sh = augment_labels(y_train_np, num_shifts).astype(np.float32, copy=False)
    return X_train_sh, y_train_sh

def augmentation(X, y, num_noise_copies=3, shifts=None):
    #Data augmentation
    X_train_gn, y_train_gn = add_gaussian_noise(X, y, num_noise_copies=num_noise_copies)
    X_train_sh, y_train_sh = shift(X, y, shifts=shifts)

    X_train_sh = torch.as_tensor(X_train_sh, dtype=torch.float32)
    y_train_sh = torch.as_tensor(y_train_sh, dtype=torch.float32)

    X_train = torch.cat([X_train_sh, X_train_gn], 0)
    y_train = torch.cat([y_train_sh, y_train_gn], 0)
    print("augmentation is done and X_train and y_train's shapes are",X_train.shape, y_train.shape)
    return X_train_gn, X_train_sh, X_train, y_train_gn, y_train_sh, y_train


    
