import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_bar(values, x_values, xlabel, ylabel, title):
    """
    Plot a histogram with several bars.

    Parameters:
    - values (dict): A dictionary containing the values to plot, where keys represent different attributes
                     (e.g., thresholds) and values are lists of scores corresponding to each attribute.
    - x_values (tuple): A tuple containing the x-axis values representing different values of 'k'.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title for the plot.

    Returns:
    - None

    Example:
    plot_cmc_or_map({'threshold': [0.418, 0.573, 0.625, 0.641, 0.656],
                     0.68: [0.242, 0.553, 0.625, 0.649, 0.66],
                     0.95: [0.431, 0.595, 0.632, 0.641, 0.656],
                     0.997: [0.455, 0.59, 0.623, 0.636, 0.651]},
                    (1, 2, 3, 4, 5),
                    'k',
                    'Score',
                    'CMC@k')
    """
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(len(x_values))
    width = 0.2
    multiplier = 0

    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=4)
        multiplier += 1

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x + (width * 1.5), x_values)
    ax.legend(loc='upper left', ncols=4)
    plt.show()
