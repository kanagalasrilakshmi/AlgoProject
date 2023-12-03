import warnings
import time
import psutil
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(123)


# Function to measure time and memory usage
def measure_sorting_performance(algorithm, df):
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024  # in kilobytes

    algorithm(df)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024  # in kilobytes

    # Calculate time and memory usage
    execution_time = end_time - start_time
    memory_usage = abs(end_memory - start_memory)

    # Output results
    # print(f"Execution Time: {execution_time} seconds")
    # print(f"Memory Usage: {memory_usage} KB\n")
    return execution_time, memory_usage


# Merge Sort implementation for sorting DataFrame by rows
def merge_sort(df):
    if len(df) <= 1:
        return df

    mid = len(df) // 2
    left = df.iloc[:mid, :]
    right = df.iloc[mid:, :]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)


def merge(left, right):
    result = pd.DataFrame()

    i, j = 0, 0

    while i < len(left) and j < len(right):
        if left.iloc[i, 0] < right.iloc[j, 0]:
            result = pd.concat([result, left.iloc[[i]]], ignore_index=True)
            i += 1
        else:
            result = pd.concat([result, right.iloc[[j]]], ignore_index=True)
            j += 1

    while i < len(left):
        result = pd.concat([result, left.iloc[[i]]], ignore_index=True)
        i += 1

    while j < len(right):
        result = pd.concat([result, right.iloc[[j]]], ignore_index=True)
        j += 1

    return result


def quick_sort(df):
    if len(df) <= 1:
        return df

    pivot = df.iloc[len(df) // 2, 0]

    left = df[df.iloc[:, 0] < pivot]
    middle = df[df.iloc[:, 0] == pivot]
    right = df[df.iloc[:, 0] > pivot]

    return pd.concat([quick_sort(left), middle, quick_sort(right)], ignore_index=True)


def bubble_sort(df):
    n = len(df)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if df.iloc[j, 0] > df.iloc[j + 1, 0]:
                df.iloc[j, :], df.iloc[j + 1, :] = df.iloc[j + 1, :].copy(), df.iloc[j, :].copy()
                swapped = True

        if not swapped:
            break

    return df


def count_sort(df):
    counting_array = [0] * 11

    for value in df.iloc[:, 0]:
        counting_array[value] += 1

    sorted_df = pd.DataFrame(columns=df.columns)

    for i in range(1, 11):
        sorted_df = pd.concat([sorted_df, df[df.iloc[:, 0] == i]], ignore_index=True)

    return sorted_df


def main():
    df = pd.read_csv('../data/data.csv')

    print("Successfully read the data")
    algorithms = {'Merge Sort': merge_sort, 'Quick Sort': quick_sort, 'Bubble Sort': bubble_sort,
                  'Count Sort': count_sort}
    data_size = {100000: df, 10000: df.head(10000), 1000: df.head(1000), 100: df.head(100)}

    # List to store individual DataFrames
    dfs = []

    # Loop through algorithms
    for algorithm_name, algorithm_func in algorithms.items():
        # Loop through data sizes
        for size, fraction in data_size.items():
            result = measure_sorting_performance(algorithm_func, fraction)
            print(f"Algorithm: {algorithm_name}, dataset size: {size}, time: {result[0]}")
            df_result = pd.DataFrame(result).T
            df_result['Algorithm'] = algorithm_name
            df_result['Data Size'] = size
            dfs.append(df_result)

    # Concatenate individual DataFrames into a single DataFrame
    results_df = pd.concat(dfs, ignore_index=True)
    results_df = results_df.rename(columns={0: 'Execution Time', 1: 'Memory Usage'})
    results_df.to_csv('../results/sorting_results.csv', index=False)


if __name__ == "__main__":
    main()
