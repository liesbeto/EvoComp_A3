import numpy as np


TARGET_POSITION = [5, 0, 0.5]


def given_fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (yt - yc) ** 2 + (xt - xc) ** 2 + (zt - zc) ** 2
    )
    return -cartesian_distance


def fitness_function_simple(history: list[float]) -> float:
    """Rewards positive x-movement."""
    
    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]
    fitness = (xc - xs)
    return fitness


def fitness_function_complex(history: list[float], a=0.5) -> float:
    """Rewards positive x-movement and punishes any y-movement (throughout
    whole history)."""

    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]

    if zc < 0:
        return -10

    average_ydeviation = np.mean(history[:][1])
    fitness = (xc - xs) - a*abs(average_ydeviation)
    return fitness