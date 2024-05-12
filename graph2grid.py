from skimage import img_as_float
from skimage.segmentation import felzenszwalb
from skimage import graph
import numpy as np
import networkx as nx
import time
import pickle
import matplotlib.pyplot as plt
import os
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter

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
    
    # 假设我们有一种方法来评估区域的重要性或特定属性，这里简化为区域大小
    # 例如，较大的区域可能代表开放空间，其温度可能更加均匀
    for region in np.unique(segments_fz):
        region_mask = segments_fz == region
        region_size = np.sum(region_mask)
        mean_temperature = rag.nodes[region]['mean temperature']
        std_temperature = rag.nodes[region].get('std_temperature', 0)
        
        # 根据区域大小调整标准差，假设较大的区域温度变化较小
        adjusted_std = std_temperature * (1 - min(region_size / np.max(segments_fz), 1))
        
        # 生成温度值
        region_temperatures = np.random.normal(mean_temperature, adjusted_std, region_size)
        
        # 填充重建矩阵
        reconstructed_matrix[region_mask] = region_temperatures
    
    return reconstructed_matrix

def graph_to_matrix_rbf(rag, segments_fz, shape):
    # 初始化重建的温度矩阵
    reconstructed_matrix = np.zeros(shape)
    
    # 准备RBF插值的输入数据
    x, y, temp = [], [], []
    for region in np.unique(segments_fz):
        # 获取该区域的中心点坐标
        indices = np.where(segments_fz == region)
        center_x = np.mean(indices[1])  # x坐标
        center_y = np.mean(indices[0])  # y坐标
        x.append(center_x)
        y.append(center_y)
        
        # 获取该区域的平均温度
        mean_temperature = rag.nodes[region]['mean temperature']
        temp.append(mean_temperature)
    
    # 使用RBF插值
    rbf = Rbf(x, y, temp, function='linear')
    
    # 为整个温度矩阵生成坐标网格
    xi, yi = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # 在网格上进行插值
    zi = rbf(xi, yi)
    
    # 使用插值结果填充重建的温度矩阵
    reconstructed_matrix = zi
    
    return reconstructed_matrix

def graph_to_matrix_gaussian(rag, segments_fz, shape):
    """
    使用高斯方法捕获邻接区域温度影响的改进温度矩阵重建函数。

    :param rag: 区域邻接图（Region Adjacency Graph）。
    :param segments_fz: 来自Felzenszwalb算法的分割数组。
    :param shape: 原始温度矩阵的形状。
    :return: 重建的温度矩阵。
    """
    # 初始化温度矩阵和权重矩阵
    temperature_matrix = np.zeros(shape)
    weight_matrix = np.zeros(shape)
    
    # 为每个区域填充温度值和权重值
    for region in np.unique(segments_fz):
        mean_temperature = rag.nodes[region]['mean temperature']
        region_mask = (segments_fz == region)
        
        temperature_matrix[region_mask] = mean_temperature
        weight_matrix[region_mask] = 1  # 假定每个区域的初始权重为1

    # 应用高斯滤波模拟温度在空间上的扩散效应
    # gaussian_sigma = 5  # 高斯核的标准差，根据需要调整
    blurred_temperature = gaussian_filter(temperature_matrix, sigma=gaussian_sigma)
    blurred_weights = gaussian_filter(weight_matrix, sigma=gaussian_sigma)
    
    # 使用高斯权重调整的温度值，避免除以0
    reconstructed_matrix = np.divide(blurred_temperature, blurred_weights, 
                                     out=np.zeros_like(blurred_temperature), 
                                     where=blurred_weights!=0)

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
    # Example temperature matrix (replace with your data)
    temperature_data = np.fromfile('/home/lig0d/compression/sample_t2.dat', dtype=np.float32).reshape(num_timepoints, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    # graphs_file_path = 're_graph_scale500_sigma1_minsize'+str(min_size)+'.pkl' 
    graphs_file_path = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}_t{timestamp}seg.pkl"
    # segments_path = '/path/to/segments'  # Change to your segments directory path
    save_path = './'
    
    common_shape = (855, 1215)  # Common s
    reconstructed_matrices = []

    first_segments_fz = calculate_segments_fz(img_as_float(temperature_data[499,:,:]), scale=100, sigma=1, min_size=min_size)

    with open(graphs_file_path, 'rb') as file:
            graphs = pickle.load(file)
    print("load graphs done.")

    for i, graph_data in enumerate(graphs):

        temperature_matrix = temperature_data[i,:,:]

        # Convert temperature matrix to float
        # temperature_matrix_float = img_as_float(temperature_matrix)

        # Perform segmentation
        # segments_fz = felzenszwalb(temperature_matrix_float, scale=500, sigma=1, min_size=min_size)

        # Reconstruct the temperature matrix from the RAG
        # reconstructed_matrix = graph_to_matrix_gaussian(graph_data, segments_fz, common_shape).astype(np.float32).flatten()
        # reconstructed_matrix = graph_to_matrix_gaussian(graph_data, first_segments_fz, common_shape).astype(np.float32).flatten()
        reconstructed_matrix = graph_to_matrix_wostd(graph_data, first_segments_fz, common_shape).astype(np.float32).flatten()
        reconstructed_matrices.append(reconstructed_matrix)

    # Now, reconstructed_matrix is your approximated temperature matrix
    # np.save(os.path.join(save_path, f'reconstructed_matrix_s500s1m'+str(min_size)+'_gaussian.npy'), reconstructed_matrices)
    # np.save(os.path.join(save_path, f"reconstructed_matrix_s500s1m{min_size}_{seg_method}_{gaussian_sigma}.npy"), reconstructed_matrices)
    np.save(os.path.join(save_path, f"reconstructed_redsea_s{scale}s{sigma}m{min_size}_{seg_method}{gaussian_sigma}_t{timestamp}seg.npy"), reconstructed_matrices)
    
    # Example usage
    # original_matrix = Your original temperature matrix
    # reconstructed_matrix = The matrix you reconstructed from the RAG

    pixelwise_mae = calculate_pixelwise_mae(temperature_data, np.array(reconstructed_matrices).reshape(num_timepoints,855,1215))

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

    plt.hist(pixelwise_mae.ravel(), bins=1000, color='blue', alpha=0.7)
    plt.title("Distribution of Pixel-wise MAE")
    plt.xlabel("MAE Value")
    plt.ylabel("Frequency")
    plt.savefig(f'The Distribution of Pixel-wise MAE-{min_size}_{seg_method}_{gaussian_sigma}.png')

if __name__ == "__main__":
    num_timepoints = 500
    wide = 855
    length  = 1215

    scale = 100
    sigma = 1
    min_size = 1

    timestamp = 499

    seg_method = 'wostd' #gaussian    wo-gaussian
    gaussian_sigma = 5

    print('seg_method: ', seg_method, 'gaussian_sigma: ', gaussian_sigma)
    
    print("scale: ", scale, "sigma: ", sigma, "min_size: ", min_size)
    main()

