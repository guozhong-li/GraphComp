from skimage.segmentation import felzenszwalb
from skimage import graph
from skimage.util import img_as_float
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count

import os
import time
import random
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm

def calculate_segments_fz(temperature_matrix, scale, sigma, min_size):
    temperature_matrix_float = img_as_float(temperature_matrix)
    segments_fz = felzenszwalb(temperature_matrix_float, scale=scale, sigma=sigma, min_size=min_size)
    return segments_fz

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
        # std_temperature = np.std(region_pixels)
        
        # Assign the mean temperature to the corresponding node in the graph
        rag.nodes[region]['mean temperature'] = mean_temperature
        # rag.nodes[region]['std temperature'] = std_temperature

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

# Define an unsupervised graph convolutional neural network model
class UnsupervisedGNNModel(MessagePassing):
    def __init__(self, input_size, hidden_size, output_size):
        super(UnsupervisedGNNModel, self).__init__(aggr='mean')
        self.conv1 = self.build_conv(input_size, hidden_size)
        self.bn1 = BatchNorm(hidden_size)

    def build_conv(self, in_channels, out_channels):
        return GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
       
        return x

class InnerProductDecoder(torch.nn.Module):
    def forward(self, z):
        # Compute the inner product score
        adj = torch.matmul(z, z.t())
        return adj

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encoder = UnsupervisedGNNModel(input_size, hidden_size, output_size)
        self.decoder = InnerProductDecoder() 

    def forward(self, x, edge_index):
        encoded = self.encoder(x, edge_index)
        decoded = self.decoder(encoded) 
        return decoded

def normalize_latent_representations(latent_representations):
    min_val = torch.min(latent_representations)
    max_val = torch.max(latent_representations)
    return (latent_representations - min_val) / (max_val - min_val), min_val, max_val

def denormalize_latent_representations(data, min_val, max_val):
    min_val = min_val.cpu().numpy() if isinstance(min_val, torch.Tensor) else min_val
    max_val = max_val.cpu().numpy() if isinstance(max_val, torch.Tensor) else max_val

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    return data * (max_val - min_val) + min_val

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Get the adjacency matrix from NetworkX graph data
def get_adjacency_matrix(graph):
    adj_matrix = nx.adjacency_matrix(graph)
    return torch.tensor(adj_matrix.todense(), dtype=torch.float32)
        
def train(model, graphs, optimizer, device, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for graph_data in graphs:
            x = torch.tensor([graph_data.nodes[node]['mean temperature'] for node in graph_data.nodes], dtype=torch.float32).view(-1, 1)
            edge_index = torch.tensor([[edge[0] for edge in graph_data.edges], [edge[1] for edge in graph_data.edges]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)

            adjacency_matrix = get_adjacency_matrix(graph_data).to(device)

            # train the model
            optimizer.zero_grad()
            encoded = model.encoder(data.x, data.edge_index)
            decoded_adjacency = model.decoder(encoded) 

            loss = ((decoded_adjacency - adjacency_matrix) ** 2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}: Loss: {total_loss / len(graphs)}')

    return model

def train_autoencoder(autoencoder, data, device, epochs=500, batch_size=16, model_path='autoencoder.pth'):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        autoencoder.load_state_dict(torch.load(model_path))
        autoencoder.to(device)
        autoencoder.eval()
        return autoencoder

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    autoencoder.train()
    data_tensor = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1).to(device)

    for epoch in range(epochs):
        total_loss = 0
        permutation = torch.randperm(data_tensor.size(0))

        for i in range(0, data_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch = data_tensor[indices]

            optimizer.zero_grad()
            output = autoencoder(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}: Loss: {total_loss / len(data_tensor)}')

    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return autoencoder

def save_cnn_encoded_representations(autoencoder, data, device, save_path):
    autoencoder.eval()  
    with torch.no_grad():  
        data_tensor = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1).to(device)
        encoded_data_tensor = autoencoder.encoder(data_tensor) 

        encoded_data = encoded_data_tensor.cpu().numpy()
        print(encoded_data.shape)

        encoded_data.tofile(save_path)
        print(f"Encoded representations saved to {save_path} as binary .dat file")

def generate_and_save_graph_embedding(model, graphs, device, save_path):
    model.eval() 
    latent_representations = []

    with torch.no_grad(): 
        for graph_data in graphs:
            x = torch.tensor([graph_data.nodes[node]['mean temperature'] for node in graph_data.nodes], dtype=torch.float32).view(-1, 1)
            edge_index = torch.tensor([[edge[0] for edge in graph_data.edges], [edge[1] for edge in graph_data.edges]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)

            latent_representation = model.encoder(data.x, data.edge_index)
            latent_representations.append(latent_representation.unsqueeze(0))  

    latent_representations = torch.cat(latent_representations, dim=0)
    latent_representations = latent_representations

    print("latent_representations.shape: ", latent_representations.shape)
    with open(f"{save_path}/rere_graph_s{scale}s{sigma}m{min_size}_latent_representations_t1seg.pkl", 'wb') as file:
        joblib.dump(latent_representations, file)

    return latent_representations

def compare_original_and_decoded(autoencoder, data, device, min_val, max_val):
    autoencoder.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1).to(device)
        decoded_data_tensor = autoencoder(data_tensor)

        # 将解码数据反归一化
        decoded_data = decoded_data_tensor.permute(0, 2, 1).cpu().numpy()
        decoded_data = denormalize_latent_representations(decoded_data, min_val, max_val)

        original_data = denormalize_latent_representations(data, min_val, max_val)  # 对原始数据也应用反归一化，以便正确比较

        mse = ((original_data - decoded_data) ** 2).mean()
        mae = abs(original_data - decoded_data).mean()

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        # 打印第一个样本的原始数据和解码数据
        print("\nOriginal Data vs Decoded Data (Sample 0):")
        print("{:<15} {:<15}".format("Original", "Decoded"))
        for orig, dec in zip(original_data[0], decoded_data[0]):
            print("{:<15} {:<15}".format(orig[0], dec[0]))

def process_matrix(args):
    index, matrix, first_segments_fz = args
    return index, graph_initialization(matrix, first_segments_fz)

def main():
    graphs_file_path = f"re_graph_scale{scale}_sigma{sigma}_minsize{min_size}_t1seg.pkl"
    model_path = f"auto_ggl_s{scale}_s{sigma}_m{min_size}_h1o1_t1seg.pkl"
    cnn_model_path = f"auto_cnn_s{scale}_s{sigma}_m{min_size}_e{epochs}_t1seg.pkl"
    cnn_latent_representation_save_path = f"auto_cnn_s{scale}_s{sigma}_m{min_size}_e{epochs}_latent_t1seg.dat"
    reconstructed_graphs_file_path = f"re2_graph_scale{scale}_sigma{sigma}_minsize{min_size}_t1seg.pkl"

    if os.path.exists(graphs_file_path):
        with open(graphs_file_path, 'rb') as file:
            graphs = joblib.load(file)
        print("load graphs done.")
    else:
        temperature_data = np.fromfile('/path/to/your/data/', dtype=np.float32).reshape(-1, wide, length)
        print("temperature_data.shape: ", temperature_data.shape)

        first_segments_fz = calculate_segments_fz(temperature_data[timestamp-1,:,:], scale=10, sigma=1, min_size=1)
        with Pool(cpu_count()) as pool:
            results = pool.map(process_matrix, [(i, matrix, first_segments_fz) for i, matrix in enumerate(temperature_data)])

        # Sort results by index to maintain original order
        results.sort(key=lambda x: x[0])
        graphs = [result[1] for result in results]
        
        print("graph_initialization done.")

        with open(graphs_file_path, 'wb') as file:
            joblib.dump(graphs, file)
    
    mean_temp, std_temp = calculate_mean_std(graphs)
    normalized_graphs = normalize_temperature(graphs, mean_temp, std_temp)

    input_size = 1 
    hidden_size = 1
    output_size = 1

    if os.path.exists(model_path):
        unsupervised_gnn_model = AutoEncoder(input_size, hidden_size, output_size).to(device)
        unsupervised_gnn_model.load_state_dict(torch.load(model_path))
        unsupervised_gnn_model.eval()
    else:
        unsupervised_gnn_model = AutoEncoder(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(unsupervised_gnn_model.parameters(), lr=0.001)

        start_time = time.time()
        print("training start...")
        representations = train(unsupervised_gnn_model, normalized_graphs, optimizer, device, epochs)
        torch.save(unsupervised_gnn_model.state_dict(), model_path)

        end_time = time.time()
        total_time = end_time - start_time
        print(f'Training done. Total time: {total_time:.2f} seconds')

    latent_representations =  generate_and_save_graph_embedding(unsupervised_gnn_model, graphs, device, '/path/to/save/latent/')

    latent_tensor = torch.tensor(latent_representations, dtype=torch.float32).permute(1, 0, 2)
    print("latent_tensor.shape: ", latent_tensor.shape)

    normalized_latent_representations, min_val, max_val = normalize_latent_representations(latent_tensor)
    autoencoder = ConvAutoEncoder().to(device)
    autoencoder = train_autoencoder(autoencoder, normalized_latent_representations, device, epochs=epochs, model_path=cnn_model_path)
    compare_original_and_decoded(autoencoder, normalized_latent_representations, device, min_val, max_val)
    save_cnn_encoded_representations(autoencoder, normalized_latent_representations, device, save_path=cnn_latent_representation_save_path)
    

if __name__ == '__main__':
    num_timepoints = 500
    wide = 855
    length  = 1215

    epochs  = 100

    scale = 500
    sigma = 1
    min_size = 1

    timestamp = 1

    loss_type = "mse"
    model_version = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
