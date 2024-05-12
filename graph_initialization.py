from skimage import img_as_float
from skimage.segmentation import felzenszwalb
from skimage import graph
import numpy as np
import networkx as nx
import time
import pickle

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
        # rag.nodes[region]['std_temperature'] = std_temperature

    return rag

def main():
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/path/to/your/data/data', dtype=np.float32).reshape(num_timepoints, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    # Different sets of parameters
    parameters = [(100, 1, 1)]

    first_segments_fz = calculate_segments_fz(temperature_data[0,:,:], scale=100, sigma=1, min_size=1)

    # Iterate over each set of parameters
    for scale, sigma, min_size in parameters:
        start_time = time.time()  # Start time

        rag = [graph_initialization(matrix, first_segments_fz) for matrix in temperature_data]

        end_time = time.time()  # End time
        duration = end_time - start_time  # Duration in seconds

        # Save the graph to a file
        file_name = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}_t1seg.pkl"
        with open(file_name, "wb") as file:
            pickle.dump(rag, file)

        print(f"Graph created and saved to {file_name} in {duration:.2f} seconds")

if __name__ == "__main__":
    num_timepoints = 500
    wide = 855
    length  = 1215

    scale = 100
    sigma = 1
    min_size = 1

    main()

