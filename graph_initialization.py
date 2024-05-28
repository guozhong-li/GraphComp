import time
import joblib
import numpy as np
from skimage import graph
from skimage import img_as_float
from skimage.segmentation import felzenszwalb
from multiprocessing import Pool, cpu_count

def calculate_segments_fz(temperature_matrix, scale, sigma, min_size):
    temperature_matrix_float = img_as_float(temperature_matrix)
    segments_fz = felzenszwalb(temperature_matrix_float, scale=scale, sigma=sigma, min_size=min_size)
    return segments_fz

def graph_initialization(temperature_matrix, segments_fz):
    # Assuming temperature_matrix is your original data matrix
    temperature_matrix_float = img_as_float(temperature_matrix)

    # Create a Region Adjacency Graph using mean temperatures
    rag = graph.rag_mean_color(temperature_matrix_float, segments_fz)

    # Iterate over regions
    for region in np.unique(segments_fz):
        # Find the mean temperature of the region
        region_pixels = temperature_matrix_float[segments_fz == region]
        mean_temperature = np.mean(region_pixels)
        # std_temperature = np.std(region_pixels)
        
        # Assign the mean temperature to the corresponding node in the graph
        rag.nodes[region]['mean temperature'] = mean_temperature
        # rag.nodes[region]['std temperature'] = std_temperature

    return rag

def process_matrix(args):
    index, matrix, first_segments_fz = args
    return index, graph_initialization(matrix, first_segments_fz)

def main():
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/path/to/your/data', dtype=np.float32).reshape(num_timepoints, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    # Different sets of parameters
    parameters = [(10, 1, 1)]


    # Iterate over each set of parameters
    for scale, sigma, min_size in parameters:
        start_time = time.time()  # Start time
        first_segments_fz = calculate_segments_fz(temperature_data[timestamp-1,:,:], scale, sigma, min_size)

        with Pool(cpu_count()) as pool:
            results = pool.map(process_matrix, [(i, matrix, first_segments_fz) for i, matrix in enumerate(temperature_data)])

        # Sort results by index to maintain original order
        results.sort(key=lambda x: x[0])
        rag = [result[1] for result in results]

        end_time = time.time()  # End time
        duration = end_time - start_time  # Duration in seconds

        # Save the graph to a file
        file_name = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}_t{timestamp}seg.pkl"
        with open(file_name, "wb") as file:
            joblib.dump(rag, file)

        print(f"Graph created and saved to {file_name} in {duration:.2f} seconds")

if __name__ == "__main__":
    num_timepoints = 500
    wide = 855
    length  = 1215

    scale = 10
    sigma = 1
    min_size = 1

    timestamp = 1

    main()

