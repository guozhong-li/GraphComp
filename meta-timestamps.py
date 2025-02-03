import numpy as np
import pandas as pd
from skimage.segmentation import felzenszwalb
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# Function to process a single pair of timestamps (vectorized optimization)
def process_timestamp_pair(ds_500, t1, t2):
    segments_t1 = felzenszwalb(ds_500[t1], scale=10, sigma=1, min_size=1)
    original_matrix = ds_500[t2]
    unique_segments, inverse_indices = np.unique(segments_t1, return_inverse=True)
    avg_values = np.bincount(inverse_indices, weights=original_matrix.ravel()) / np.bincount(inverse_indices)
    modified_matrix = avg_values[inverse_indices].reshape(original_matrix.shape)
    difference = np.abs(modified_matrix - original_matrix)
    return np.mean(difference)


# Function to process a single timestamp
def process_single_timestamp_global(ds_500, t1):
    segments_t1 = felzenszwalb(ds_500[t1], scale=10, sigma=1, min_size=1)
    print(np.unique(segments_t1).shape)  
    unique_segments, inverse_indices = np.unique(segments_t1, return_inverse=True)
    results = []

    for t2 in range(ds_500.shape[0]):
        original_matrix = ds_500[t2]
        avg_values = np.bincount(inverse_indices, weights=original_matrix.ravel()) / np.bincount(inverse_indices)
        modified_matrix = avg_values[inverse_indices].reshape(original_matrix.shape)
        difference = np.abs(modified_matrix - original_matrix)
        results.append(np.mean(difference))
    
    return results


# Parallel processing function
def process_timestamps_parallel(ds_500, output_csv):
    if ds_500.shape[0] < 500:
        raise ValueError("The dataset must contain at least 500 timestamps for processing.")

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    process_single_timestamp_global,
                    [ds_500] * ds_500.shape[0],  
                    range(ds_500.shape[0]),     
                ),
                total=ds_500.shape[0],
                desc="Processing timestamps",
            )
        )

    result_matrix = np.array(results)
    pd.DataFrame(result_matrix).to_csv(output_csv, index=False, header=False)


# Prefix sum for cost calculation
def compute_prefix_sum(matrix):
    return np.cumsum(matrix, axis=1)


# Cost function for a row segment
def cost(prefix_sum, row, left, right):
    return prefix_sum[row, right] - (prefix_sum[row, left - 1] if left > 0 else 0)


# Dynamic programming function
def dynamic_programming_partition(matrix, max_rows=10):
    n_rows, n_cols = matrix.shape
    prefix_sum = compute_prefix_sum(matrix)
    dp = np.full((max_rows + 1, n_cols + 1), float('inf'))
    path = np.zeros((max_rows + 1, n_cols + 1), dtype=int)

    for j in range(n_cols):
        dp[1][j] = cost(prefix_sum, 0, 0, j)
        path[1][j] = 0

    for k in range(2, max_rows + 1):
        for j in range(n_cols):
            for m in range(j):  
                cur_cost = dp[k - 1][m] + cost(prefix_sum, k - 1, m + 1, j)
                if cur_cost < dp[k][j]:
                    dp[k][j] = cur_cost
                    path[k][j] = m  

    best_cost = float('inf')
    best_row_count = -1
    for k in range(1, max_rows + 1):
        if dp[k][n_cols - 1] < best_cost:
            best_cost = dp[k][n_cols - 1]
            best_row_count = k

    result = []
    current_col = n_cols - 1
    for k in range(best_row_count, 0, -1):
        start_col = path[k][current_col]
        result.append((k - 1, start_col, current_col))
        current_col = start_col - 1
    result.reverse()

    return best_cost, best_row_count, result


# Main function
def main():
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/path/to/your/data', dtype=np.float32).reshape(-1, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)
    output_csv = "segmentation_results.csv"

    print("Starting parallel timestamp processing...")
    process_timestamps_parallel(temperature_data, output_csv)
    print(f"Results saved to {output_csv}")

    # Example matrix for DP partitioning
    matrix = np.random.randint(1, 100, (Max_meta, temperature_data.shape[0]))  # 10 rows, 20 columns
    best_cost, best_row_count, result = dynamic_programming_partition(matrix)

    print("Best cost:", best_cost)
    print("Best row count:", best_row_count)
    print("Optimal partitions:", result)


if __name__ == "__main__":
    wide = 721
    length  = 1440

    Max_meta = 10

    main()