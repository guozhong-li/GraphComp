from skimage import img_as_float
from skimage.segmentation import felzenszwalb
import numpy as np
import networkx as nx
import time
import joblib
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count

def calculate_segments_fz(temperature_matrix, scale, sigma, min_size):
    temperature_matrix_float = img_as_float(temperature_matrix)
    segments_fz = felzenszwalb(temperature_matrix_float, scale=scale, sigma=sigma, min_size=min_size)
    return segments_fz

def graph_to_matrix_wostd(rag, segments_fz, shape):
    """
    Converts a RAG back to a temperature matrix.

    :param rag: The Region Adjacency Graph.
    :param segments_fz: The segmentation array from Felzenszwalb's algorithm.
    :param shape: Shape of the original temperature matrix.
    :return: Reconstructed temperature matrix.
    """
    # Initialize an empty matrix
    reconstructed_matrix = np.zeros(shape)

    # Iterate over each region in the segmentation
    for region in np.unique(segments_fz):
        # Get the mean temperature for this region from the RAG
        mean_temperature = rag.nodes[region]['mean temperature']

        # Assign the mean temperature to all pixels in this region
        reconstructed_matrix[segments_fz == region] = mean_temperature

    return reconstructed_matrix

def graph_to_matrix_wstd(rag, segments_fz, shape):
    
    reconstructed_matrix = np.zeros(shape)
    
    # Assume we have a method to assess the importance or specific attributes of regions, here simplified as region size
    # For example, larger regions might represent open spaces where temperature is more uniform
    for region in np.unique(segments_fz):
        region_mask = segments_fz == region
        region_size = np.sum(region_mask)
        mean_temperature = rag.nodes[region]['mean temperature']
        std_temperature = rag.nodes[region].get('std_temperature', 0)
        
        # Adjust the standard deviation based on region size, assuming smaller temperature variation in larger areas
        adjusted_std = std_temperature * (1 - min(region_size / np.max(segments_fz), 1))
        
        region_temperatures = np.random.normal(mean_temperature, adjusted_std, region_size)
        
        reconstructed_matrix[region_mask] = region_temperatures
    
    return reconstructed_matrix

def graph_to_matrix_gaussian(rag, segments_fz, shape):
    """
    An improved function for reconstructing temperature matrices that captures the influence of temperatures in adjacent regions using a Gaussian method.

    :param rag: Region Adjacency Graph (RAG).
    :param segments_fz: Segmentation array from the Felzenszwalb algorithm.
    :param shape: The shape of the original temperature matrix.
    :return: The reconstructed temperature matrix.
    """
    # Initialize the temperature matrix and weight matrix
    temperature_matrix = np.zeros(shape)
    weight_matrix = np.zeros(shape)
    
    # Fill temperature values and weight values for each region
    for region in np.unique(segments_fz):
        mean_temperature = rag.nodes[region]['mean temperature']
        region_mask = (segments_fz == region)
        
        temperature_matrix[region_mask] = mean_temperature
        weight_matrix[region_mask] = 1  # Assume an initial weight of 1 for each region

    # Apply Gaussian filter to simulate the spatial diffusion effect of temperature
    gaussian_sigma = 5  # Standard deviation of the Gaussian kernel, adjust as needed
    blurred_temperature = gaussian_filter(temperature_matrix, sigma=gaussian_sigma)
    blurred_weights = gaussian_filter(weight_matrix, sigma=gaussian_sigma)
    
     # Use Gaussian-weighted temperatures, avoiding division by zero
    reconstructed_matrix = np.divide(blurred_temperature, blurred_weights, 
                                     out=np.zeros_like(blurred_temperature), 
                                     where=blurred_weights!=0)

    return reconstructed_matrix

def process_graph(data):
    graph_data, first_segments_fz, common_shape = data
    # Reconstruct the temperature matrix from the RAG
    reconstructed_matrix = graph_to_matrix_wostd(graph_data, first_segments_fz, common_shape).astype(np.float32).flatten()
    return reconstructed_matrix

def main():
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/path/to/your/data', dtype=np.float32).reshape(-1, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    graphs_file_path = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}_t{timestamp}seg.pkl"
    save_path = './'
    
    common_shape = (855, 1215)  # Common shape
    parameters = [(1, 1, 1)]

    for scale, sigma, min_size in parameters:
        print("scale: ", scale, "sigma: ", sigma, "min_size: ", min_size)
        graphs_file_path = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}_t{timestamp}seg.pkl"
        
        # segments_path = '/path/to/segments'  # Change to your segments directory path
        first_segments_fz = calculate_segments_fz(img_as_float(temperature_data[timestamp-1,:,:]), scale=scale, sigma=sigma, min_size=min_size)

        with open(graphs_file_path, 'rb') as file:
                graphs = joblib.load(file)
        print("load graphs done.")
        reconstructed_matrices = []  
        start_time = time.time()  # Start time

        with Pool(cpu_count()) as pool:
            reconstructed_matrices = pool.map(process_graph, [(graph_data, first_segments_fz, common_shape) for graph_data in graphs])

        end_time = time.time()  # End time
        duration = end_time - start_time  # Duration in seconds
        # Now, reconstructed_matrix is your approximated temperature matrix
        np.save(os.path.join(save_path, f"re_era5_s{scale}s{sigma}m{min_size}_{seg_method}{gaussian_sigma}_t{timestamp}seg.npy"), reconstructed_matrices)
        
        print(f"reconstructed and saved  in {duration:.2f} seconds")
    

if __name__ == "__main__":

    wide = 721
    length  = 1440

    timestamp = 0

    seg_method = 'wostd' #gaussian    wo-gaussian
    gaussian_sigma = 5

    print('seg_method: ', seg_method, 'gaussian_sigma: ', gaussian_sigma)
    
    main()

