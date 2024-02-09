import pathlib
from pathlib import Path

import matplotlib.axes
import matplotlib.container
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy
import pandas


def calculate_distributions(
    predictions: numpy.ndarray,
) -> tuple[numpy.ndarray]:
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

    labels_and_counts: tuple = numpy.unique(
        ar=predictions,
        return_counts=True,
    )

    labels: numpy.ndarray = labels_and_counts[0]
    counts: numpy.ndarray = labels_and_counts[1]
    percentages: numpy.ndarray = numpy.array(
        object=(counts / numpy.sum(a=counts)) * 100,
        dtype=float,
    )
    percentages = numpy.round(
        a=percentages,
        decimals=2,
    )

    return (labels, counts, percentages)


def plot_distributions(
    labels: numpy.ndarray,
    counts: numpy.ndarray,
    percentages: numpy.ndarray,
    plot_title: str = "",
    xlabel_bar: str = "",
    ylabel_bar: str = "",
    output_path: str = ".",
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
    xlabel_bar: str
        The name of the x axis of the bar chart.
    ylabel_bar: str
        The name of the y axis of the bar chart.
    output_path: str
        The path where output images are placed.

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
        height=counts,
    )
    ax[1].bar_label(
        container=rects,
        labels=["{}".format(n) for n in counts],
        # fmt="%.2f",
    )
    ax[1].set_xlabel(xlabel=xlabel_bar)
    ax[1].set_ylabel(ylabel=ylabel_bar)
    # plt.show()
    plot_path: pathlib.PosixPath = Path(output_path, "{}.png".format(plot_title))
    fig.savefig(fname=plot_path)
    plt.close()


def multiday_plotter(
    daily_sensor_df: list[pandas.DataFrame] = [],
    sensor_name: str = "",
    output_path: str = ".",
) -> None:
    output_path: pathlib.PosixPath = pathlib.Path(output_path).resolve()
    # Output the distribution of all behaviours throughout the recorded
    # period.
    behaviours: numpy.ndarray = daily_sensor_df[0]["predicted_labels"]
    counts: numpy.ndarray = round(daily_sensor_df[0]["counts (mins)"], 2)
    percentages: numpy.ndarray = round(counts / counts.sum() * 100, 2)
    plot_distributions(
        labels=behaviours,
        counts=counts,
        percentages=percentages,
        plot_title="{}_Total_time".format(sensor_name),
        xlabel_bar="Behaviour",
        ylabel_bar="Total time (Min)",
        output_path=output_path,
    )
    # Output the distribution of behaviours in each respective day.
    for df in daily_sensor_df[1:]:
        behaviours: numpy.ndarray = df["predicted_labels"]
        counts: numpy.ndarray = round(df["counts (mins)"], 2)
        percentages: numpy.ndarray = round(counts / counts.sum() * 100, 2)
        plot_distributions(
            labels=behaviours,
            counts=counts,
            percentages=percentages,
            plot_title="{}_Total_time: {}".format(
                sensor_name,
                df["timestamps"].iloc[0],
            ),
            xlabel_bar="Behaviour",
            ylabel_bar="Total time (Min)",
            output_path=output_path,
        )
