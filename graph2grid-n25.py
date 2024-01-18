from skimage import img_as_float
from skimage.segmentation import felzenszwalb
from skimage import graph
import numpy as np
import networkx as nx
import time
import pickle
import matplotlib.pyplot as plt
import os

def graph_to_matrix(rag, segments_fz, shape):
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


def calculate_pixelwise_mae(original_matrix, reconstructed_matrix):
    """
    Calculate the pixel-wise Mean Absolute Error (MAE) between the original and reconstructed matrices.

    :param original_matrix: The original temperature matrix.
    :param reconstructed_matrix: The reconstructed temperature matrix.
    :return: Array of pixel-wise MAE values.
    """
    # Ensure both matrices have the same shape
    assert original_matrix.shape == reconstructed_matrix.shape, "Matrices must have the same shape."

    # Calculate the pixel-wise MAE
    pixelwise_mae = np.abs(original_matrix - reconstructed_matrix)
    return pixelwise_mae


def main():
    all_rags = []  # 这里假设你已经有了一个包含所有RAGs的列表
    all_segments = []  # 这里假设你有一个与RAGs相对应的segments_fz列表
    common_shape = (855, 1215)

    reconstructed_matrices = []

    for rag, segments, shape in zip(all_rags, all_segments, common_shape):
        reconstructed_matrix = graph_to_matrix(rag, segments, shape)
        reconstructed_matrices.append(reconstructed_matrix)

    return reconstructed_matrices

def main():
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/home/lig0d/compression/sample_t2.dat', dtype=np.float32).reshape(num_timepoints, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    graphs_file_path = './re_graph_scale500_sigma1_minsize100.pkl'  # Change to your RAGs directory path
    # segments_path = '/path/to/segments'  # Change to your segments directory path
    save_path = './'
    num_graphs = 500  # Total number of graphs
    common_shape = (855, 1215)  # Common s
    reconstructed_matrices = []

    with open(graphs_file_path, 'rb') as file:
            graphs = pickle.load(file)
    print("load graphs done.")

    for i, graph_data in enumerate(graphs):

        temperature_matrix = temperature_data[i,:,:]

        # Convert temperature matrix to float
        temperature_matrix_float = img_as_float(temperature_matrix)

        # Perform segmentation
        segments_fz = felzenszwalb(temperature_matrix_float, scale=500, sigma=1, min_size=min_size)

        # Reconstruct the temperature matrix from the RAG
        reconstructed_matrix = graph_to_matrix(graph_data, segments_fz, common_shape).astype(np.float32).flatten()
        reconstructed_matrices.append(reconstructed_matrix)

    # Now, reconstructed_matrix is your approximated temperature matrix
    np.save(os.path.join(save_path, f'reconstructed_matrix_s500s1m100_5decimal.npy'), reconstructed_matrices)
    # Example usage
    # original_matrix = Your original temperature matrix
    # reconstructed_matrix = The matrix you reconstructed from the RAG

    pixelwise_mae = calculate_pixelwise_mae(temperature_data, np.array(reconstructed_matrices).reshape(500,855,1215))

    # Calculate max, min, and distribution
    max_mae = np.max(pixelwise_mae)
    min_mae = np.min(pixelwise_mae)
    mean_mae = np.mean(pixelwise_mae)
    std_mae = np.std(pixelwise_mae)

    print("Number of nodes:", len(graph_data.nodes))
    print("Number of edges:", len(graph_data.edges))

    print("Maximum MAE:", max_mae)
    print("Minimum MAE:", min_mae)
    print("Mean MAE:", mean_mae)
    print("Standard Deviation of MAE:", std_mae)

    # plt.imshow(reconstructed_matrix-temperature_matrix, cmap='gray_r', origin='lower')
    plt.imshow(pixelwise_mae, cmap='gray_r', vmax=max_mae, origin='lower') 

    plt.colorbar()
    # plt.axis('off')
    plt.show()

    plt.hist(pixelwise_mae.ravel(), bins=1000, color='blue', alpha=0.7)
    plt.title("Distribution of Pixel-wise MAE")
    plt.xlabel("MAE Value")
    plt.ylabel("Frequency")
    plt.savefig('The Distribution of Pixel-wise MAE-'+str(min_size)+'.png')

if __name__ == "__main__":
    num_timepoints = 500
    wide = 855
    length  = 1215
    min_size = 100
    print('min_size:', min_size)
    main()

