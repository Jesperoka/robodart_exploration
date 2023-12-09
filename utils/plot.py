import matplotlib.pyplot as plt
import numpy as np

from data import load


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running Average with 100 Episode Backward-Only Window')
    plt.grid(True)
    plt.savefig(figure_file)

def plot_moving_average(x, scores, figure_file, title="Training Progress", xlabel="Episodes", ylabel="Score", window_size=100):
    running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(x[:len(running_avg)], running_avg)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(figure_file)
    plt.close()


if __name__ == "__main__":
    print("Not yet implemented.") 
