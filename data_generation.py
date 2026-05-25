import numpy as np
import torch
from torch.utils.data import Dataset


def function_library(function="sine", p=[0.4, 1.0, 2.0, -3.0]):
    if function == "sine":
        return lambda x: np.sin(np.pi * x[:, 0]).reshape(-1, 1)

    elif function == "scaledSine":
        return lambda x: np.sin(np.pi * p[1] * x[:, 0]).reshape(-1, 1)

    elif function == "oneKink":
        return lambda x: (
            p[2] * (x[:, 0] < p[0]) * (x[:, 0] - p[0])
            + p[1]
            + p[3] * (x[:, 0] > p[0]) * (x[:, 0] - p[0])
        ).reshape(-1, 1)

    elif function == "abs":
        return lambda x: (np.abs(x[:, 0] - 0.4) + 0.5).reshape(-1, 1)

    elif function == "squared":
        return lambda x: (x[:, 0] ** 2 - 0.5).reshape(-1, 1)

    elif function == "negToPos":
        return lambda x: (2.0 * (x[:, 0] > 0) - 1).astype(float).reshape(-1, 1)

    elif function == "cubed":
        return lambda x: (x[:, 0] ** 3).reshape(-1, 1)

    elif function == "exp":
        return lambda x: (np.exp(x[:, 0]) * 0.7357 - 1).reshape(-1, 1)

    elif function == "compositeSine":
        def composite(x):
            scaled = function_library("scaledSine", p=p)(x)
            if scaled.ndim == 1:
                scaled = scaled.reshape(-1, 1)
            comps = [
                function_library("oneKink", p=p)(scaled),
                function_library("abs", p=p)(scaled),
                function_library("squared", p=p)(scaled),
                function_library("negToPos", p=p)(scaled),
                function_library("cubed", p=p)(scaled),
                function_library("sine", p=p)(scaled),
                function_library("exp", p=p)(scaled),
            ]
            return np.hstack(comps)

        return composite

    elif function == "compositeSinePlus":
        def composite(x):
            scaled = function_library("scaledSine", p=p)(x)
            if scaled.ndim == 1:
                scaled = scaled.reshape(-1, 1)
            comps = [
                function_library("oneKink", p=p)(scaled),
                function_library("abs", p=p)(scaled),
                function_library("squared", p=p)(scaled),
                function_library("negToPos", p=p)(scaled),
                function_library("cubed", p=p)(scaled),
                function_library("sine", p=p)(scaled),
                function_library("exp", p=p)(scaled),
                function_library("oneKink", p=p)(x),
            ]
            return np.hstack(comps)

        return composite


def generate_data(n_samples=1000, function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1)):
    x = np.random.uniform(interval[0], interval[1], (n_samples, 1))
    y = function_library(function, p)(x)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return x, y


def generate_noisy_data(n_samples=1000, function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), noise_std=0.1):
    x = np.random.uniform(interval[0], interval[1], (n_samples, 1))
    y = function_library(function, p)(x)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = y + np.random.normal(0, noise_std, y.shape)
    return x, y


class CompositeSineDataset(Dataset):
    def __init__(self, n_samples=1000, function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), noise_std=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if noise_std is None:
            x, y = generate_data(n_samples=n_samples, function=function, p=p, interval=interval)
        else:
            x, y = generate_noisy_data(n_samples=n_samples, function=function, p=p, interval=interval, noise_std=noise_std)
        self.x = torch.from_numpy(x.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def plot_data(x, y, function="sine"):
    import matplotlib.pyplot as plt
    plt.scatter(x, y, alpha=0.5)
    plt.title(f"{function} function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()
