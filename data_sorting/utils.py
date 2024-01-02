import gc
import os
import random
from collections import deque
from itertools import islice
from pathlib import Path

import numpy as np


def merge_external_files(
    f1: str,
    f2: str,
    output_path: str,
) -> None:
    """
    Read and merge-sort two text file and output the merged file.

    Parameters
    ----------
        f1: str
            Path to the first data file.

        f2: str
            Path to the second data file.

    Returns
    -------
    None
    """

    # Need a header row to keep the format correct.
    header_row: str = ...
    file_uid: int = random.randint(0, 9999999)
    os.system(
        "rm -rf {}merged_{}.txt".format(
            output_path,
            file_uid,
        )
    )
    Path(
        "{}merged_{}.txt".format(
            output_path,
            file_uid,
        )
    ).touch()
    with open(
        file="{}merged_{}.txt".format(
            output_path,
            file_uid,
        ),
        mode="a",
    ) as output_file:
        # Implementing the "merge" part of the merge sort algorithm externally.
        # Read 2 files simutenously.
        with open(f1) as file1, open(f2) as file2:
            # Write the head row first.
            header_row = file1.readline()
            next(file2)
            output_file.write(header_row)

            # Comparing the timestamp value of each row from the two files
            # and add them to the output file accordingly.
            line1: str
            line2: str
            line1_written: bool = True
            line2_written: bool = True
            while True:
                if line1_written:
                    line1 = file1.readline()
                    line1_written = False
                if line2_written:
                    line2 = file2.readline()
                    line2_written = False
                if not line1 or not line2:
                    break

                # if np.datetime64(file1[i1][4]) <= np.datetime64(file2[i2][4]):
                time1: np.datetime64 = np.datetime64(line1.split(sep=",")[4])
                time2: np.datetime64 = np.datetime64(line2.split(sep=",")[4])
                if time1 <= time2:
                    # output_string: str = ",".join(file1[i1])
                    output_file.write(line1)
                    line1_written = True
                else:
                    # output_string: str = ",".join(file2[i2])
                    output_file.write(line2)
                    line2_written = True

                gc.collect()

            if line1:
                output_file.write(line1)
                while True:
                    line1 = file1.readline()
                    if not line1:
                        break
                    output_file.write(line1)
                    gc.collect()
            if line2:
                output_file.write(line2)
                while True:
                    line2 = file2.readline()
                    if not line2:
                        break
                    output_file.write(line2)
                    gc.collect()

        gc.collect()

    # Remove the two files after merged.
    os.system(command="rm -rf {} {}".format(f1, f2))


def merge_external_files_chunks(
    f1: str,
    f2: str,
    output_path: str,
    chunk_size: int = 1,
) -> None:
    """
    Read and merge-sort two text file and output the merged file.

    Parameters
    ----------
        f1: str
            Path to the first data file.

        f2: str
            Path to the second data file.

        output_path: str
            Path to the directory in which merged file will
            be stored.

        chunk_size: int, default=1
            The number of rows that will be read and store on memory
            \tfor each file. Used to avoid memory overflow.

    Returns
    -------
    None
    """

    # Need a header row to keep the format correct.
    header_row: str = ...
    file_uid: int = random.randint(0, 9999999)
    os.system(
        "rm -rf {}merged_{}.txt".format(
            output_path,
            file_uid,
        )
    )
    Path(
        "{}merged_{}.txt".format(
            output_path,
            file_uid,
        )
    ).touch()

    # Open 2 files that need to be merged and the output file to be written to.
    with open(
        file="{}merged_{}.txt".format(output_path, file_uid), mode="a"
    ) as output_file, open(f1) as file1, open(f2) as file2:
        # Write the head row first.
        header_row = file1.readline()
        next(file2)
        output_file.write(header_row)

        # Read a chunk of rows from both file. This is to ensure that the
        # function won't eat up the entire memory as well as complete its
        # task in an acceptable time.
        merged_rows: list[str] = []
        f1_remaining: bool = False
        f2_remaining: bool = False

        while True:
            if not f1_remaining:
                f1_rows: deque[str] = deque(islice(file1, chunk_size))
                if f1_rows:
                    f1_remaining = True
            if not f2_remaining:
                f2_rows: deque[str] = deque(islice(file2, chunk_size))
                if f2_rows:
                    f2_remaining = True

            # Write the remaining lines of either file if the other file
            # runs out of line.
            if f1_remaining and not f2_remaining:
                output_file.writelines(f1_rows)
                while line1 := list(islice(file1, chunk_size)):
                    output_file.writelines(line1)
                break
            if f2_remaining and not f1_remaining:
                output_file.writelines(f2_rows)
                while line2 := list(islice(file2, chunk_size)):
                    output_file.writelines(line2)
                break

            # If there is no rows left from any of the two files, write the
            # remaining rows to the output file and break out of the loop.

            # For each row from two files, compare the timestamp values and
            # add the "smaller" row into the list of merged rows.
            while f1_rows and f2_rows:
                # Compare, write and iterate.
                time1: np.datetime64 = np.datetime64(f1_rows[0].split(sep=",")[4])
                time2: np.datetime64 = np.datetime64(f2_rows[0].split(sep=",")[4])
                if time1 <= time2:
                    merged_rows.append(f1_rows.popleft())
                else:
                    merged_rows.append(f2_rows.popleft())

            if not f1_rows:
                f1_remaining = False
            if not f2_rows:
                f2_remaining = False

            # Write the merge-sorted rows into the output files.
            output_file.writelines(merged_rows)
            # Reset the merged list.
            merged_rows = []

            gc.collect()

        gc.collect()

    # Remove the two files after merged.
    os.system(command="rm -rf {} {}".format(f1, f2))
