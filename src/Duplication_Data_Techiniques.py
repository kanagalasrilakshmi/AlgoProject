import pandas as pd
import numpy as np
import hashlib
from collections import deque
import time
import psutil


# Data creation for duplication
# Have 1000 rows and 5 columns such that 30% rows are duplicates
def creation_dataset():
    # Define the size of the dataset
    num_rows = 1000
    num_columns = 5
    # Create a random DataFrame
    data = np.random.randint(0, 10, size=(num_rows, num_columns))
    df = pd.DataFrame(data, columns=[f'Column_{i+1}' for i in range(num_columns)])
    # Introduce duplicates (let's assume 30% of the data will be duplicates)
    num_duplicates = int(0.3 * num_rows)
    duplicate_indices = np.random.choice(df.index, num_duplicates, replace=False)
    df_duplicates = df.iloc[duplicate_indices]
    # Concatenate the original DataFrame with duplicates
    df_with_duplicates = pd.concat([df, df_duplicates])
    # Save the DataFrame with duplicates to a CSV file
    df_with_duplicates.to_csv('dataset_with_duplicates.csv', index=False)
    print("Dataset with duplicates created and saved as 'dataset_with_duplicates.csv'")
    return df_with_duplicates

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
     #print(f"Execution Time: {execution_time} seconds")
     #print(f"Memory Usage: {memory_usage} KB\n")
    return execution_time, memory_usage

# methods for removing duplicates
# method-1 - hashing
def custom_hash(row):
    # Custom hash function for lists to create a unique hash
    row_str = str(row)
    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()
    # Update the hash object with the row string
    sha256.update(row_str.encode('utf-8'))
    # Return the hexadecimal digest of the hash
    return sha256.hexdigest()

def remove_duplicates(data):
    seen = {}  # Dictionary to store hashed rows
    unique_data = []
    for row in data:
        hashed = custom_hash(row)
        if hashed not in seen:
            seen[hashed] = True
            unique_data.append(row)
    return unique_data

# method-2 - sorting and hashing
# use merge sort, quick sort, bubble sort, count sort
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

def count_sort(df):
    counting_array = [0] * 11

    for value in df.iloc[:, 0]:
        counting_array[value] += 1

    sorted_df = pd.DataFrame(columns=df.columns)

    for i in range(1, 11):
        sorted_df = pd.concat([sorted_df, df[df.iloc[:, 0] == i]], ignore_index=True)

    return sorted_df

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

def quick_sort(df):
    if len(df) <= 1:
        return df

    pivot = df.iloc[len(df) // 2, 0]

    left = df[df.iloc[:, 0] < pivot]
    middle = df[df.iloc[:, 0] == pivot]
    right = df[df.iloc[:, 0] > pivot]

    return pd.concat([quick_sort(left), middle, quick_sort(right)], ignore_index=True)

# method -3 - stack
# Remove duplicates using stack
def remove_duplicates_with_stack(data):
    stack = []
    seen = set()
    for row in data:
      row_tuple = tuple(row)
      hashed = custom_hash(row)
      if row_tuple not in seen:
          stack.append(row)
          seen.add(row_tuple)
    return stack
def pop_elements(stack):
  list_stack = []
  while stack:
    list_stack.append(stack.pop())
  return list_stack


# method -4 - queue
def remove_duplicates_with_queue(data):
    queue = deque()
    seen = set()
    for row in data:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            queue.append(row)
            seen.add(row_tuple)
    return queue

def peek_pop_Queue(queue):
  queue_list = []
  while queue:
      top_element = queue.popleft()
      queue_list.append(top_element)
  return queue_list

def hashing_results(df):
    # Remove duplicates using custom hashing
    data = df.values
    unique_data_custom_hash = remove_duplicates(data)
    df = pd.DataFrame(unique_data_custom_hash)
    df.to_csv('data_from_hash.csv', index=False)

def merge_results(df):
    result_merge = merge_sort(df)
    # removing duplicates
    unique_data_custom_hash_merge = remove_duplicates(result_merge.values)
    df = pd.DataFrame(unique_data_custom_hash_merge)
    df.to_csv('data_from_hash_merge.csv', index=False)

def stack_results(df):
    data = df.values
    # Remove duplicates using stack
    unique_data_stack = remove_duplicates_with_stack(data)
    list_stack_popped = pop_elements(unique_data_stack)
    df = pd.DataFrame(list_stack_popped)
    df.to_csv('data_from_stack.csv', index=False)

def queue_results(df):
    data = df.values
    # Remove duplicates using queue
    unique_data_queue = remove_duplicates_with_queue(data)
    queue_list = peek_pop_Queue(unique_data_queue)
    df = pd.DataFrame(queue_list)
    df.to_csv('data_from_queue.csv', index=False)

def count_results(df):
    result = count_sort(df)
    # removing duplicates
    unique_data_custom_hash_count = remove_duplicates(result.values)
    df = pd.DataFrame(unique_data_custom_hash_count)
    df.to_csv('data_from_hash_count.csv', index=False)

def bubble_sort_results(df):
    result = bubble_sort(df)
    # removing duplicates
    unique_data_custom_hash_bubble = remove_duplicates(result.values)
    df = pd.DataFrame(unique_data_custom_hash_bubble)
    df.to_csv('data_from_hash_bubble.csv', index=False)

def quick_sort_results(df):
    result = quick_sort(df)
    # removing duplicates
    unique_data_custom_hash_quick = remove_duplicates(result.values)
    df = pd.DataFrame(unique_data_custom_hash_quick)
    df.to_csv('data_from_hash_quick.csv', index=False)


def main():
    # creation of dataset
    df_with_duplicates = creation_dataset()
    df = pd.read_csv('dataset_with_duplicates.csv')
    algorithms = {'Hashing':hashing_results,'Merge Sort': merge_results, 'Quick Sort': quick_sort_results, 
                  'Bubble Sort': bubble_sort_results, 'Count Sort': count_results,
                  'Stack':stack_results,'Queue':queue_results}
    # List to store individual DataFrames
    dfs = []
    columns = ['Algorithm', 'Execution Time','Memory Usage']
    dfsorting = pd.DataFrame(columns=columns)
    # Loop through algorithms
    for algorithm_name, algorithm_func in algorithms.items():
        execution_time, memory_usage = measure_sorting_performance(algorithm_func, df)

        new_row = pd.DataFrame([[algorithm_name, execution_time,memory_usage]], columns=columns)
        dfsorting = pd.concat([dfsorting, new_row], ignore_index=True)

    # # Concatenate individual DataFrames into a single DataFrame
    # results_df = pd.concat(dfs, ignore_index=True)
    # results_df = results_df.rename(columns={0: 'Execution Time', 1: 'Memory Usage'})
    dfsorting.to_csv('sorting_results.csv', index=False)
    

if __name__ == "__main__":
    main()



