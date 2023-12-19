import pandas as pd
import numpy as np
import tensorflow as tf

from utils import get_sequential_input, standardize_dataframe


def clean_data(full_data: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicated rows.
    full_data.drop_duplicates(inplace=True)

    # # Sort values based on timestamps.
    full_data.sort_values(
        by=["timestamps"],
        ascending=True,
        inplace=True,
    )
    full_data = full_data.reset_index(drop=True)

    # Convert numerical values (input data) to float types.
    full_data_export: pd.DataFrame = full_data.copy()
    full_data.iloc[:, 5:] = full_data.iloc[:, 5:].astype("float")

    # Keep only the timestamp and the data columns.
    timestamp_col_index: int = full_data.columns.get_loc("timestamps")
    kept_cols: list[str] = full_data.columns[timestamp_col_index:]
    full_data = full_data[kept_cols]

    # Standardize data.
    full_data = standardize_dataframe(
        data=full_data,
        std_cols=full_data.columns[1:],
    )

    return full_data


def label_data(
    raw_data: pd.DataFrame,
    model: tf.keras.Model,
    window_size: int,
    batch_size: int,
) -> np.ndarray:
    x_test: np.ndarray = get_sequential_input(
        df=raw_data,
        window_size=window_size,
        labelled=False,
    )
    test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)
    # Process test results and output confusion matrices.
    y_pred: np.ndarray = model.predict(
        x=test_ds,
        verbose=0,
    )

    # Obtain the prediction by taking the argument max of the returned tensor, flatten the
    # prediction tensor and create a numpy array from that.
    y_pred_cm: tf.Tensor = (
        tf.math.argmax(
            y_pred,
            axis=2,
            output_type=tf.int64,
        ),
    )[0]
    y_pred_cm = tf.reshape(
        tensor=y_pred_cm,
        shape=(-1),
    )
    predicted_labels: np.ndarray = np.array(
        y_pred_cm,
        dtype=np.int32,
    )

    # There are some rows that are not included in the windows (some final rows). Get the
    # sequential input for the last `window_size` rows and take only the missing one.
    # no_missing_rows: int = len(raw_data) % window_size
    # last_window: pd.DataFrame = raw_data.iloc[-window_size:]
    # missing_inputs: np.ndarray = get_sequential_input(
    #     df=last_window,
    #     window_size=window_size,
    #     labelled=False,
    # )

    # labelled_classes: np.ndarray = np.concatenate(
    #     (predicted_labels, trimmed_predicted_labels),
    #     axis=0,
    #     dtype=np.int32,
    # )
    labelled_classes: np.ndarray = predicted_labels
    label_map: dict[int, str] = {
        0: "g",
        1: "i",
        2: "o",
        3: "r",
    }

    label_col: pd.Series = pd.Series(labelled_classes)
    label_col = label_col.map(label_map)
    labels: np.ndarray = label_col.to_numpy(dtype=str)

    return labels
