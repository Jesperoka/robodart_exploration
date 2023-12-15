import matplotlib.pyplot as plt
import numpy as np

from data import load


def plot_performance(aggregated_data, config_labels, xlabel="Episode", ylabel="Metric", title=None):
    plt.figure(figsize=(10, 6))
    for (mean_data, min_data, max_data), label in zip(aggregated_data, config_labels):
        x = np.arange(len(mean_data))
        plt.plot(x, mean_data, label=label)
        plt.fill_between(x, min_data, max_data, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(config_labels))
    plt.subplots_adjust(bottom=0.2)  # Adjust as needed for spacing
    plt.show()


def load_and_aggregate_data(filenames_per_config, label):
    aggregated_data = []
    for config_filenames in filenames_per_config:
        config_data = [load(filename, label) for filename in config_filenames]
        config_data = np.array(config_data)
        mean_data = np.mean(config_data, axis=0)
        min_data = np.min(config_data, axis=0)
        max_data = np.max(config_data, axis=0)
        aggregated_data.append((mean_data, min_data, max_data))
    return aggregated_data 


if __name__ == "__main__":
    # Example filenames grouped by configuration
    filenames_per_config = [
        ['logs/baseline_run0.h5py', 'logs/baseline_run1.h5py', "logs/baseline_run2.h5py"],
        ['logs/munch_r0_run0.h5py', 'logs/munch_r0_run1.h5py', "logs/munch_r0_run2.h5py"],
        ['logs/hybrid_her_r0_run0.h5py', 'logs/hybrid_her_r0_run1.h5py', "logs/hybrid_her_r0_run2.h5py"],
    ]
    config_labels = ['Config 1', 'Config 2', "Config 2"]
    data_label = 'final distance' 

    aggregated_data = load_and_aggregate_data(filenames_per_config, data_label)
    plot_performance(aggregated_data, config_labels, ylabel=data_label)

