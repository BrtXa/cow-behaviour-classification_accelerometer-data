import matplotlib.axes
import matplotlib.container
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def calculate_distributions(
    predictions: np.ndarray,
) -> tuple[np.ndarray]:
    """
    Outputs unique labels, total counts and percentages of predictions stored in
    a numpy array.

    Parameters
    ----------
    predictions : array-like
        An array-like object (preferably numpy array) that stores prediction results.

    Returns
    -------
    tuple[numpy.ndarray]
        A tuple that store the unique labels, their counts and percentages.
    """

    labels_and_counts: tuple = np.unique(
        ar=predictions,
        return_counts=True,
    )

    labels: np.ndarray = labels_and_counts[0]
    counts: np.ndarray = labels_and_counts[1]
    percentages: np.ndarray = np.array(
        object=(counts / np.sum(a=counts)) * 100,
        dtype=float,
    )
    percentages = np.round(
        a=percentages,
        decimals=2,
    )

    return (labels, counts, percentages)


def plot_distributions(
    labels: np.ndarray,
    counts: np.ndarray,
    percentages: np.ndarray,
    plot_title: str,
) -> None:
    """
    Plot a pie chart and a bar chart side-by-side that illustrate how classes
    are distributed among predictions.

    Parameters
    ----------
    labels : array-like
        An array-like object (preferably numpy array) that stores prediction labels.
    counts : array-like
        An array-like object (preferably numpy array) that stores count numbers of
        each label.
    percentages : array-like
        An array-like object (preferably numpy array) that stores percentages of labels.
    plot_title: str
        The name of the graph, should be set as the name of the sensor.

    Returns
    -------
    None
    """
    # Instantiate the graph for each sensor with label.
    subplot_tuple: tuple = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 9),
    )
    fig: matplotlib.figure.Figure = subplot_tuple[0]
    ax: tuple[matplotlib.axes.Axes] = subplot_tuple[1]
    plt.suptitle(t=plot_title)

    # Plot 0: The pie chart.
    ax[0].pie(
        x=counts,
        labels=["{}: {}%".format(l, p) for l, p in zip(labels, percentages)],
        colors=["olivedrab", "rosybrown", "red", "saddlebrown"],
    )

    # Plot 1: The bar chart.
    rects: matplotlib.container.BarContainer = ax[1].bar(
        x=labels,
        height=percentages,
    )
    ax[1].bar_label(
        container=rects,
        labels=["{}".format(n) for n in counts],
        # fmt="%.2f",
    )
