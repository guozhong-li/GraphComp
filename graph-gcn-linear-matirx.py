from skimage.segmentation import felzenszwalb
from skimage import graph
from skimage.util import img_as_float
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_networkx
from torch.utils.data import Dataset, DataLoader

class TemperatureGraphDataset(Dataset):
    def __init__(self, temperature_matrices, graph_data):
        """
        temperature_matrices: List of 855x1215 temperature matrices
        graph_data: List of graph data, each corresponding to a temperature matrix
        """
        self.temperature_matrices = temperature_matrices
        self.graph_data = graph_data

    def __len__(self):
        return len(self.temperature_matrices)

    def __getitem__(self, idx):
        temp_matrix = self.temperature_matrices[idx]
        graph = self.graph_data[idx]

        # 将温度矩阵转换为适合卷积网络处理的形式
        # 假设这里温度矩阵是单通道的，所以添加一个通道维度
        temp_matrix = torch.tensor(temp_matrix, dtype=torch.float).unsqueeze(0)

        # 对于图数据，您需要提取节点特征、边索引等
        x, edge_index = graph  # 假设 graph 是一个元组 (节点特征, 边索引)
        
        return temp_matrix, (x, edge_index)

def graph_initialization(temperature_matrix):
    # Assuming temperature_matrix is your original data matrix
    temperature_matrix = img_as_float(temperature_matrix)

    # Use felzenszwalb method for segmentation
    segments_fz = felzenszwalb(temperature_matrix, scale=500, sigma=1, min_size=100)

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

def calculate_mean_std(train_graphs):
    all_temperatures = [data['mean temperature'] for graph in train_graphs for _, data in graph.nodes(data=True)]
    mean_temp = np.mean(all_temperatures)
    std_temp = np.std(all_temperatures)
    return mean_temp, std_temp

def normalize_temperature(train_graphs, mean_temp, std_temp):
    normalized_graphs = []
    for graph in train_graphs:
        normalized_graph = graph.copy()
        for node, data in normalized_graph.nodes(data=True):
            data['mean temperature'] = (data['mean temperature'] - mean_temp) / std_temp
        normalized_graphs.append(normalized_graph)
    return normalized_graphs

def denormalize_temperature(train_graphs, mean_temp, std_temp):
    for graph in train_graphs:
        for node, data in graph.nodes(data=True):
            data['mean temperature'] = (data['mean temperature'] * std_temp) + mean_temp


class CNNEncoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(CNNEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(64 * 855 * 1215, hidden_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, input_size, output_channels):
        super(Decoder, self).__init__()
        self.fc = torch.nn.Linear(input_size, 64 * 855 * 1215)
        self.deconv1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 855, 1215)  # 重塑为适合卷积层的维度
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # 使用sigmoid激活函数以确保输出在[0,1]范围内
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self, cnn_input_size, gcn_input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.cnn_encoder = CNNEncoder(cnn_input_size, hidden_size)
        self.gcn_encoder = GCNEncoder(gcn_input_size, hidden_size)
        self.decoder = Decoder(hidden_size * 2, output_size)  # 假设简单的连接操作

    def forward(self, temp_matrix, graph_data):
        # CNN编码温度矩阵
        encoded_temp = self.cnn_encoder(temp_matrix)
        # GCN编码图数据
        encoded_graph = self.gcn_encoder(graph_data)
        # 融合特征
        combined_feature = torch.cat((encoded_temp, encoded_graph), dim=1)
        # 解码重建温度矩阵
        decoded = self.decoder(combined_feature)
        return decoded

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        temp_matrix, graph_data = batch
        temp_matrix = temp_matrix.to(device)
        graph_data = (graph_data[0].to(device), graph_data[1].to(device))  # 假设graph_data是(x, edge_index)的元组

        optimizer.zero_grad()
        reconstructed_matrix = model(temp_matrix, graph_data)
        loss = criterion(reconstructed_matrix, temp_matrix)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def main():

    graphs_file_path = 'graph_scale500_sigma1_minsize5000.pkl'
    model_path = 'auto_gcn_linear_matrix.pth'

    # 检查是否存在保存的图数据文件
    if os.path.exists(graphs_file_path):
        # 如果文件存在，直接加载图数据
        with open(graphs_file_path, 'rb') as file:
            graphs = pickle.load(file)
        print("load graphs done.")
    else:
        # 如果文件不存在，生成图数据
        temperature_data = np.fromfile('/home/lig0d/compression/sample_t2.dat', dtype=np.float32).reshape(num_timepoints, wide, length)
        print("temperature_data.shape: ", temperature_data.shape)

        # 对每个输入数据应用 graph_initialization 函数
        graphs = [graph_initialization(data_matrix) for data_matrix in temperature_data]
        print("graph_initialization done.")

        # 保存图数据到本地
        with open(graphs_file_path, 'wb') as file:
            pickle.dump(graphs, file)
    
    mean_temp, std_temp = calculate_mean_std(graphs)
    normalized_graphs = normalize_temperature(graphs, mean_temp, std_temp)

    # 初始化无监督的图卷积神经网络模型
    input_size = 1  # 假设每个节点的特征是一个实数
    hidden_size = 64
    output_size = 1

    if os.path.exists(model_path):
        # 如果已经存在模型文件，直接加载模型
        model = AutoEncoder(input_channels=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        # 否则，训练模型并保存
        model = AutoEncoder(input_channels=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # 准备数据加载器
         # not normalized
        dataset = TemperatureGraphDataset(temperature_data, graphs)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        start_time = time.time()
        print("training start...")
        for epoch in range(epochs):
            loss = train(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f'Training done. Total time: {total_time:.2f} seconds')

        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':

    total_size = 1038825 #1215
    num_timepoints = 500
    wide = 855
    length  = 1215

    epochs  = 100

    loss_type = "mse"
    model_version = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main()
