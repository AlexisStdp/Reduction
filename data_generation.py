import numpy as np
import torch
from torch.utils.data import Dataset


FUNCTION_COMPONENTS = {
    "compositeSine": [
        "oneKink(scaledSine(x))",
        "abs(scaledSine(x))",
        "squared(scaledSine(x))",
        "negToPos(scaledSine(x))",
        "cubed(scaledSine(x))",
        "sine(scaledSine(x))",
        "exp(scaledSine(x))",
    ],
    "compositeSinePlus": [
        "oneKink(scaledSine(x))",
        "abs(scaledSine(x))",
        "squared(scaledSine(x))",
        "negToPos(scaledSine(x))",
        "cubed(scaledSine(x))",
        "sine(scaledSine(x))",
        "exp(scaledSine(x))",
        "oneKink(x)",
    ],
    "allFunctions": [
        "scaledSine(x)",
        "oneKink(x)",
        "abs(x)",
        "squared(x)",
        "negToPos(x)",
        "cubed(x)",
        "sine(x)",
        "exp(x)",
    ]
}


def function_library(function="sine", p=[0.4, 1.0, 2.0, -3.0]):
    if function == "sine":
        return lambda x: np.sin(np.pi * x[:, 0]).reshape(-1, 1)

    elif function == "scaledSine":
        return lambda x: (0.5 * np.cos(2.5 * np.pi * (x[:, 0] + 1))).reshape(-1, 1)

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
    
    elif function == "allFunctions":
        def composite(x):
            comps = [
                function_library("scaledSine", p=p)(x),
                function_library("oneKink", p=p)(x),
                function_library("abs", p=p)(x),
                function_library("squared", p=p)(x),
                function_library("negToPos", p=p)(x),
                function_library("cubed", p=p)(x),
                function_library("sine", p=p)(x),
                function_library("exp", p=p)(x),
            ]
            return np.hstack(comps)

        return composite

    valid_functions = [
        "sine",
        "scaledSine",
        "oneKink",
        "abs",
        "squared",
        "negToPos",
        "cubed",
        "exp",
        "compositeSine",
        "compositeSinePlus",
        "allFunctions",
    ]
    raise ValueError(f"Unknown function '{function}'. Expected one of: {', '.join(valid_functions)}")


def generate_data(n_samples=1000, function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.uniform(interval[0], interval[1], (n_samples, 1))
    y = function_library(function, p)(x)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return x, y


def generate_noisy_data(n_samples=1000, function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), noise_std=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.uniform(interval[0], interval[1], (n_samples, 1))
    y = function_library(function, p)(x)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = y + rng.normal(0, noise_std, y.shape)
    return x, y


class CompositeSineDataset(Dataset):
    def __init__(self, n_samples=1000, function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), noise_std=None, seed=None):
        rng = np.random.default_rng(seed)
        if noise_std is None:
            x, y = generate_data(n_samples=n_samples, function=function, p=p, interval=interval, rng=rng)
        else:
            x, y = generate_noisy_data(n_samples=n_samples, function=function, p=p, interval=interval, noise_std=noise_std, rng=rng)
        self.x = torch.from_numpy(x.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def sample_function_grid(function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), n_points=1000):
    x = np.linspace(interval[0], interval[1], n_points).reshape(-1, 1)
    y = function_library(function, p)(x)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return x, y


def plot_data(x, y, function="sine", component_names=None, show_points=False):
    import matplotlib.pyplot as plt

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    order = np.argsort(x[:, 0])
    x_sorted = x[order, 0]
    y_sorted = y[order]

    n_outputs = y_sorted.shape[1]
    if component_names is None:
        component_names = FUNCTION_COMPONENTS.get(function)
    if component_names is None:
        component_names = [f"output_{idx}" for idx in range(n_outputs)]

    n_cols = min(3, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.ravel()

    for idx in range(n_outputs):
        ax = axes[idx]
        ax.plot(x_sorted, y_sorted[:, idx], linewidth=2)
        if show_points:
            ax.scatter(x[:, 0], y[:, idx], alpha=0.35, s=12)
        ax.set_title(component_names[idx] if idx < len(component_names) else f"output_{idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    for idx in range(n_outputs, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"{function} visualization", fontsize=14)
    fig.tight_layout()
    plt.show()


def visualize_function(function="sine", p=[0.4, 1.0, 2.0, -3.0], interval=(-1, 1), n_points=256):
    x, y = sample_function_grid(function=function, p=p, interval=interval, n_points=n_points)
    plot_data(x, y, function=function)


if __name__ == "__main__":
    # visualize_function(function="compositeSinePlus")
    # visualize_function(function="allFunctions")
    visualize_function(function="scaledSine")
