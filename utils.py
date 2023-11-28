import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay


def get_frame(
    df: pd.DataFrame,
    window_size: int,
    window_per_epoch: int,
):
    """
    Get only one single frame that will be fed to the model.
    """
    random_range: int = df.shape[0] - window_size

    windows: list = []
    labels: list = []
    for _ in range(window_per_epoch):
        random_starting_point: int = random.randint(0, random_range)
        window: pd.DataFrame = df.iloc[
            random_starting_point : random_starting_point + window_size
        ]

        windows.append(window.iloc[:, :-1])
        labels.append(window["label"])

    # Bring the segments into a better shape
    windows = np.asarray(windows)  # .reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return windows, labels


def get_sequential_frame(
    df: pd.DataFrame,
    window_size: int,
):
    """
    An altenative of the `get_frame` function, which create a set of
    multiple windows in a sequential manner.
    # TODO: This function should be merged with the `get_frame` function
    # to create a cleaner codebase.
    """
    window_range: int = df.shape[0] - window_size

    windows: list = []
    labels: list = []
    for index in range(0, window_range, window_size):
        window: pd.DataFrame = df.iloc[index : index + window_size]

        windows.append(window.iloc[:, :-1])
        labels.append(window["label"])

    # Bring the segments into a better shape
    windows = np.asarray(windows)  # .reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return windows, labels


def get_epoch_frame(
    df: pd.DataFrame,
    window_size: int,
    window_per_epoch: int,
    epoch: int,
):
    """
    Get frames for the specified number of `epoch`.
    """
    random_range: int = df.shape[0] - window_size
    epoch_windows: list = []
    epoch_labels: list = []

    for _ in range(epoch):
        windows: list = []
        labels: list = []
        for _ in range(window_per_epoch):
            random_starting_point: int = random.randint(0, random_range)
            window: pd.DataFrame = df.iloc[
                random_starting_point : random_starting_point + window_size
            ]

            windows.append(window.iloc[:, :-1])
            labels.append(window["label"])

        # Bring the segments into a better shape
        windows = np.asarray(windows)  # .reshape(-1, frame_size, N_FEATURES)
        labels = np.asarray(labels)

        epoch_windows.append(windows)
        epoch_labels.append(labels)

    epoch_windows = np.asarray(epoch_windows)
    epoch_labels = np.asarray(epoch_labels)

    return epoch_windows, epoch_labels


def handle_minor_classes(
    data: pd.DataFrame,
    minor_classes: list[str],
    type: str = "remove",
) -> pd.DataFrame:
    """
    Handle
    """
    type = type.lower()
    if type == "remove":
        for value in minor_classes:
            data = data[data["Behaviour"] != value]
    else:
        data = data.replace(minor_classes, "o")

    return data


def standardize_dataframe(
    data: pd.DataFrame,
    std_cols: Union[
        str,
        list,
    ],
):
    """
    This function will take a dataframe, standardizes specified columns and return
    the standardized dataframe.
    """
    scaler = StandardScaler()

    to_be_std: pd.DataFrame = data[std_cols]
    std: np.ndarray = scaler.fit_transform(to_be_std)

    remaining_cols: list = list(set(data.columns) - set(std_cols))
    not_std: np.ndarray = data[remaining_cols].to_numpy()

    merged_arr: np.ndarray = np.concatenate(
        (std, not_std),
        axis=1,
    )

    std_df: pd.DataFrame = pd.DataFrame(
        data=merged_arr,
        columns=data.columns,
    )
    # std_df.iloc[:, -1] = std_df.iloc[:, -1].astype("int")

    return std_df


def plot_learning_curve(
    metrics: dict[str, list],
    epochs: int,
):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, metrics["train_accuracy"])
    plt.plot(epoch_range, metrics["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, metrics["train_loss"])
    plt.plot(epoch_range, metrics["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()


def plot_confusion_matrix(
    actual: Union[
        list,
        np.ndarray,
    ],
    pred: Union[
        list,
        np.ndarray,
    ],
    unique_numerical_labels: Union[
        list,
        np.ndarray,
    ],
    display_labels: Union[
        list,
        np.ndarray,
    ],
    normalize: str = "true",
    title: str = "Confusion matrix",
    export: str = "false",
) -> np.ndarray:
    """
    A comfort function to autoamtically calculate and plot a confusion matrix.

    Parameters
    ----------
    actual : array-like that stores the ground truth (correct) target values.

    pred : array-like that stores the estimated targets as returned by a classifier.

    unique_numerical_labels: array-like that stores the merical encoding of unique classes.

    display_labels: array-like that stores the class names.

    normalize (default="true"): string that specifies how the confusion matrix should be normalized. This will
    be passed to ``sklearn.metrics.confusion_matrix``

    title (default="Confusion matrix"): string that is used as the name of the confusion matrix.

    export (default=false): string value that decides whether or not the confusion matrix should
    be exported as a picture.
    """
    mat = confusion_matrix(
        y_true=actual,
        y_pred=pred,
        labels=unique_numerical_labels,
        normalize=normalize,
    )

    fig, ax = plt.subplots()
    ax.set_title(label=title)
    cmd = ConfusionMatrixDisplay(mat, display_labels=display_labels)
    cmd.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")

    if export != "false":
        plt.savefig(
            export,
            bbox_inches="tight",
        )

    return mat
