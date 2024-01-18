from skimage import img_as_float
from skimage.segmentation import felzenszwalb
from skimage import graph
import numpy as np
import networkx as nx
import time
import pickle

def graph_initialization(temperature_matrix, scale,sigma,min_size):
    # Assuming temperature_matrix is your original data matrix
    temperature_matrix = img_as_float(temperature_matrix)

    # Use felzenszwalb method for segmentation
    segments_fz = felzenszwalb(temperature_matrix, scale=scale, sigma=sigma, min_size=min_size)

    # Create a Region Adjacency Graph using mean temperatures
    rag = graph.rag_mean_color(temperature_matrix, segments_fz)

    # Iterate over regions
    for region in np.unique(segments_fz):
        # Find the mean temperature of the region
        region_pixels = temperature_matrix[segments_fz == region]
        mean_temperature = np.mean(region_pixels)
        
        # Assign the mean temperature to the corresponding node in the graph
        rag.nodes[region]['mean temperature'] = mean_temperature

    # Define edge weights as absolute difference of mean temperatures
    for edge in rag.edges:
        source, target = edge
        diff = np.abs(rag.nodes[source]['mean temperature'] - rag.nodes[target]['mean temperature'])

        # Set edge weight to 0 if the temperature difference is below a certain threshold, otherwise set it to 1
        weight = 0 if diff <= 0 else 1
        
        rag.edges[source, target]['weight'] = weight

    return rag

def main():
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/home/lig0d/compression/sample_t2.dat', dtype=np.float32).reshape(num_timepoints, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    # Different sets of parameters
    # parameters = [(500, 1, 500), (500, 1, 100), (500, 1, 1), (500, 0.5, 1), (100, 0.2, 1)]
    parameters = [(500, 1, 500), (500, 1, 100)]

    # Iterate over each set of parameters
    for scale, sigma, min_size in parameters:
        print('min_size: ', min_size)
        start_time = time.time()  # Start time

        # rag = graph_initialization(temperature_data, scale, sigma, min_size)
        rag = [graph_initialization(data_matrix,scale,sigma,min_size) for data_matrix in temperature_data]

        end_time = time.time()  # End time
        duration = end_time - start_time  # Duration in seconds

        # Save the graph to a file
        file_name = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}-new.pkl"
        with open(file_name, "wb") as file:
            pickle.dump(rag, file)

        print(f"Graph created and saved to {file_name} in {duration:.2f} seconds")

if __name__ == "__main__":
    num_timepoints = 500
    wide = 855
    length  = 1215
    main()

