import numpy as np, copy, os, re, argparse
from calibration_library import *
from dataParsing_library import *
from debug_test import *
from distortion_library import * 

def read_coordinates(file_path):
    """
    Reads a file, skips the header, and extracts coordinates, assuming they are comma-separated.

    Parameters:
    - file_path: Path to the file containing two sets of (x, y, z) points.
    Returns:
    - A list of tuples, where each tuple contains two sets of (x, y, z) coordinates.
    """

    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)

        coordinates = []
        for line in file:
            values = list(map(float, line.strip().split()))
            coordinates.append(((values[0], values[1], values[2]), (values[3], values[4], values[5])))
        return coordinates

def compute_error(computed_data, target_data):
    """
    Computes the Euclidean distance between computed data points and target data points.

    Parameters:
    - computed_data: A list of tuples/lists representing the computed (x, y, z) points.
    - target_data: A list of tuples/lists representing the target (x, y, z) points.

    Returns:
    - A list of Euclidean distances representing the error for each corresponding point.
    """

    if len(computed_data) != len(target_data):
        raise ValueError("The number of computed data points must match the number of target data points.")

    errors = [np.linalg.norm(np.array(computed) - np.array(target)) for computed, target in zip(computed_data, target_data)]
    return errors

def main():
    # User interface prompt that takes input from user
    parser = argparse.ArgumentParser(description='error analysis input')
    parser.add_argument('choose_set', help='The alphabetical index of the data set')
    parser.add_argument('input_type', help='The debug or unknown input data to process')
    args = parser.parse_args()

    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    base_path_output = os.path.join(dirname, f'OUTPUT\\PA4-{args.choose_set}-{args.input_type}-Output.txt')
    base_path_sample = os.path.join(dirname, f'PROGRAMS\\2023_pa345_student_data\\PA4-{args.choose_set}-{args.input_type}-Output.txt')
    computed_data = read_coordinates(base_path_output)
    target_data = read_coordinates(base_path_sample)
    error = compute_error(computed_data, target_data)

    print(error)

if __name__ == '__main__':
    main()

"""
def calculate_error_from_sample(self, file1, file2, use_reference=0):
    with open(file1, 'r') as f1:
        next(f1)
        data1 = [list(map(float, line.strip().split(','))) for line in f1]

    with open(file2, 'r') as f2:
        next(f2)
        data2 = [list(map(float, line.strip().split(','))) for line in f2]

    reference_data = data1 if use_reference == 0 else data2

    percentage_differences = []
    for row1, row2 in zip(data1, data2):
        percentage_diff_row = []
        for val1, val2 in zip(row1, row2):
            percentage_diff = ((val2 - val1) / val1)
            percentage_diff_row.append(percentage_diff)
        percentage_differences.append(percentage_diff_row)

    return percentage_differences
"""
