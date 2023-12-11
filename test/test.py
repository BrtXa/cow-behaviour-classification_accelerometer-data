#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'minimumTime' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY data
#


import numpy as np

# def adjust(working_times: dict[int, int]) -> list:
#     # The working hours will first needs to be stored for further reference. The idea
#     # is to give a proficient task of the user with most hours to the users with
#     # least hours.
#     original_times: dict[int, int] = working_times.copy()

#     # Re-distributions of tasks will be meaningless when the differences
#     # between the number of hours each users need is less than 3.
#     most_minutes: int = max(working_times, key=working_times.get)
#     least_minutes: int = min(working_times, key=working_times.get)

#     # Ensure that only proficient tasks are moved (-1) instead of non-proficient tasks (-2)
#     if working_times[most_minutes] -

# def minimumTime(n, data):
#     # The dict that store the number of hours each user needs.
#     working_times: dict[int, int] = {}

#     # Assign users with all their proficient tasks.
#     for t in data:
#         if t in working_times:
#             working_times[t] += 1
#         else:
#             working_times[t] = 1

#     # With the current list of working times, start adjusting the tasks to
#     # reduce the working time.
#     adjusted_time: list = adjust(working_times)


import numpy as np


def adjust(working_times: np.ndarray) -> np.ndarray:
    while True:
        # The working hours will first needs to be sorted. The idea is to give
        # a proficient task of the user with most hours to the users with least hours.

        # Re-distributions of tasks will be meaningless when the differences
        # between the number of hours each users need is less than 3.
        # working_times.sort()
        np.sort(working_times)
        if working_times[-1] - working_times[0] >= 3:
            pass
        else:
            break
        working_times[-1] -= 1
        working_times[0] += 2

    return working_times


def minimumTime(n, data):
    # The array that store the number of hours each user needs.
    # working_times: list = [0 for _ in range(n)]
    working_times: np.ndarray = np.zeros(n, dtype=int)

    # Assign users with all their proficient tasks.
    for t in data:
        working_times[t - 1] += 1

    # With the current list of working times, start adjusting the tasks to
    # reduce the working time.
    print(working_times)
    adjusted_time: list = adjust(working_times)
    return np.max(adjusted_time)


if __name__ == "__main__":
    result = minimumTime(3, [1, 2, 3, 2, 2, 2])
    # print(result)
