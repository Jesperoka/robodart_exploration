import h5py
import numpy as np
from typeguard import typechecked


@typechecked
def add(filename: str, label: str, data: np.ndarray, max_shape=(None, ), chunks=True):
    with h5py.File(filename, 'a') as file:
        if label in file:
            dataset = file[label]
            dataset.resize((dataset.shape[0] + data.shape[0], ))  # type: ignore
            dataset[-data.shape[0]:] = data  # type: ignore
        else:
            file.create_dataset(label, data=data, maxshape=max_shape, chunks=chunks)


@typechecked
def save(filename: str, data: list[np.ndarray], labels: list[str]):
    for d, l in zip(data, labels):
        add(filename, l, d)


@typechecked
def load(filename: str, label: str) -> np.ndarray:
    with h5py.File(filename, 'r') as file:
        data = file[label][:] # type: ignore
        return data # type: ignore
