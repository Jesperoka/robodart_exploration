from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
from dtypes import NP_DTYPE


def unpack_dataclass(dataclass: type):
    return [f.default for f in fields(dataclass)]


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running Average with 100 Episode Backward-Only Window')
    plt.grid(True)
    plt.savefig(figure_file)


def all_finite_inputs(f0):
    def f1(*args):
        for arg in args: assert(np.isfinite(arg).all()), arg
        return f0(*args)
    return f1

def all_finite_outputs(f0):
    def f1(*args):
        res = f0(*args)
        for r in res: assert(np.isfinite(r).all()), r 
        return res 
    return f1

def all_finite_in_out(f0):
    def f1(*args):
        for arg in args: assert(np.isfinite(arg).all()), arg
        res = f0(*args)
        for r in res: assert(np.isfinite(r).all()), r 
        return res 
    return f1
        
def all_correct_dtype(f0)
    def f1(*args):
        
        return f0
    return f1
