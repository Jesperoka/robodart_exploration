from dataclasses import fields
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np

from utils.dtypes import NP_DTYPE, T_DTYPE

RNG = np.random.default_rng(seed=69420)


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

# Sanity assertions as decorators.
# Ironically I went a bit crazy making this, but its funny so whatever.
# All of this just to not have assertions clutter up my code, and be toggleable.
# Sometimes I actually kinda sorta want #define directives
# TODO: can maybe just conver args to numpy, then I dont have to used method kwarg,
# simplifies things a lot...

# ---------------------------------------------------------------------------- #
DO_ASSERTIONS = True  # Toggle assertion decorators


# Decorator that turns on or off the other decorators
def do_assertions(on=DO_ASSERTIONS):

    def decorator_wrapper(decorator):

        @wraps(decorator)
        def new_decorator(*args, **kwargs):
            if on:
                return decorator(*args, **kwargs)
            else:
                if not args: return lambda f: f
                return args[0]

        return new_decorator

    return decorator_wrapper


@do_assertions(on=DO_ASSERTIONS)
def all_finite_in(method=False):  # method kwarg is needed to not check 'self' argument for methods

    def decorator(f0):
        idx = 1 if method else 0

        def f1(*args, **kwargs):
            for arg in args[idx:]:
                assert (np.isfinite(arg).all()), arg
            return f0(*args, **kwargs)

        return f1

    return decorator


@do_assertions(on=DO_ASSERTIONS)
def all_finite_out(f0):

    def f1(*args, **kwargs):
        res = f0(*args, **kwargs)
        for r in res:
            assert (np.isfinite(r).all()), r
        return res

    return f1


@do_assertions(on=DO_ASSERTIONS)
def all_finite_in_out(f0):

    def f1(*args, **kwargs):
        for arg in args:
            assert (np.isfinite(arg).all()), arg
        res = f0(*args, **kwargs)
        for r in res:
            assert (np.isfinite(r).all()), r
        return res

    return f1


@do_assertions(on=DO_ASSERTIONS)
def all_np_dtype_in(method=False):

    def decorator(f0):
        idx = 1 if method else 0

        def f1(*args, **kwargs):
            for arg in args[idx:]:
                assert (arg.dtype == NP_DTYPE), arg
            return f0(*args, **kwargs)

        return f1

    return decorator


@do_assertions(on=DO_ASSERTIONS)
def all_np_dtype_out(f0):

    def f1(*args, **kwargs):
        res = f0(*args, **kwargs)
        for r in res:
            assert (r.dtype == NP_DTYPE), r
        return res

    return f1


@do_assertions(on=DO_ASSERTIONS)
def all_np_dtype_in_out(method=False):

    def decorator(f0):
        idx = 1 if method else 0

        def f1(*args, **kwargs):
            for arg in args[idx:]:
                assert (arg.dtype == NP_DTYPE), arg
            res = f0(*args, **kwargs)
            for r in res:
                assert (r.dtype == NP_DTYPE), r
            return res

        return f1

    return decorator


@do_assertions(on=DO_ASSERTIONS)
def all_t_dtype_in(method=False):

    def decorator(f0):
        idx = 1 if method else 0

        def f1(*args, **kwargs):
            for arg in args[idx:]:
                assert (arg.dtype == T_DTYPE), arg
            return f0(*args, **kwargs)

        return f1

    return decorator


@do_assertions(on=DO_ASSERTIONS)
def all_t_dtype_out(f0):

    def f1(*args, **kwargs):
        res = f0(*args, **kwargs)
        for r in res:
            assert (r.dtype == T_DTYPE), r
        return res

    return f1


@do_assertions(on=DO_ASSERTIONS)
def all_t_dtype_in_out(method=False):

    def decorator(f0):
        idx = 1 if method else 0

        def f1(*args, **kwargs):
            for arg in args[idx:]:
                assert (arg.dtype == T_DTYPE), arg
            res = f0(*args, **kwargs)
            for r in res:
                assert (r.dtype == T_DTYPE), r
            return res

        return f1

    return decorator


# ---------------------------------------------------------------------------- #
