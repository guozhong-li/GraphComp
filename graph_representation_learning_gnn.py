import os
import time
import joblib
import torch
import numpy as np
import networkx as nx
from skimage import graph
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb
from multiprocessing import Pool, cpu_count

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.utils import to_networkx

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

# Get the adjacency matrix from NetworkX graph data
def get_adjacency_matrix(graph):
    adj_matrix = nx.adjacency_matrix(graph)
    return torch.tensor(adj_matrix.todense(), dtype=torch.float32)


def weighted_binary_cross_entropy(output, target, weight=None):
    if weight is not None:
        assert len(weight) == 2
        loss = weight[1] * (target * torch.log(output)) + \
               weight[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

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


def reconstruct_graphs(autoencoder, graphs, device):
    reconstructed_graphs = []
    autoencoder.eval()
    with torch.no_grad():
        for graph_data in graphs:
            x = torch.tensor([graph_data.nodes[node]['mean temperature'] for node in graph_data.nodes], dtype=torch.float32).view(-1, 1)
            edge_index = torch.tensor([[edge[0] for edge in graph_data.edges], [edge[1] for edge in graph_data.edges]], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index).to(device)

            encoded = autoencoder.encoder(data.x, data.edge_index)
            decoded_adjacency = autoencoder.decoder(encoded) 

            # Convert the decoded output to a NetworkX graph object
            reconstructed_graph = to_networkx(data, to_undirected=True)

            # Set 'mean temperature' for each node
            for node_id, mean_temp in enumerate(x):
                reconstructed_graph.nodes[node_id]['mean temperature'] = mean_temp.item()

            decoded_adjacency = decoded_adjacency.cpu().numpy()
            edges = np.argwhere(decoded_adjacency > 0.5)
            reconstructed_graph.add_edges_from(edges)

            reconstructed_graphs.append(reconstructed_graph)

    return reconstructed_graphs

def view_graphs(graphs, reconstructed_graphs):
    selected_indices = [0, 10] 

    for i in selected_indices:
        original_graph = graphs[i]
        reconstructed_graph = reconstructed_graphs[i]

        original_num_nodes = len(original_graph.nodes)
        original_num_edges = len(original_graph.edges)
        reconstructed_num_nodes = len(reconstructed_graph.nodes)
        reconstructed_num_edges = len(reconstructed_graph.edges)

        print(f"Graph {i + 1} Node and Edge Counts:")
        print(f"Original: Nodes = {original_num_nodes}, Edges = {original_num_edges}")
        print(f"Reconstructed: Nodes = {reconstructed_num_nodes}, Edges = {reconstructed_num_edges}")

        print(f"Graph {i + 1} Node Temperatures:")
        for node in original_graph.nodes():
            original_temperature = original_graph.nodes[node]['mean temperature']
            reconstructed_temperature = reconstructed_graph.nodes[node]['mean temperature']
            temperature_difference = abs(original_temperature - reconstructed_temperature)
            print(f"Node {node}: Original Temperature = {original_temperature}, Reconstructed Temperature = {reconstructed_temperature}, Temperature Difference = {temperature_difference}")

        print("\n")

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
            latent_representations.append(latent_representation.cpu().numpy())

    print("latent_representation.shape: ", latent_representation.shape)

    latent_representations = np.array(latent_representations)

    with open(f"{save_path}/latent_representations_s{scale}_s{sigma}_m{min_size}.dat", 'wb') as file:
        latent_representations.tofile(file)

    return latent_representations

def main():

    graphs_file_path = f"graph_scale{scale}_sigma{sigma}_minsize{min_size}.pkl"
    model_path = f"auto_ggl_s{scale}_s{sigma}_m{min_size}_h1o1.pkl"
    reconstructed_graphs_file_path = f"re_graph_scale{scale}_sigma{sigma}_minsize{min_size}.pkl"

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
        train(unsupervised_gnn_model, normalized_graphs, optimizer, device, epochs)
        torch.save(unsupervised_gnn_model.state_dict(), model_path)

        end_time = time.time()
        total_time = end_time - start_time
        print(f'Training done. Total time: {total_time:.2f} seconds')

    generate_and_save_graph_embedding(unsupervised_gnn_model, graphs, device, '/path/to/save/latent/')

    reconstructed_graphs = reconstruct_graphs(unsupervised_gnn_model, normalized_graphs, device)
    denormalize_temperature(reconstructed_graphs, mean_temp, std_temp)
    with open(reconstructed_graphs_file_path, 'wb') as file:
            joblib.dump(reconstructed_graphs, file)

    view_graphs(graphs, reconstructed_graphs)

if __name__ == '__main__':

    num_timepoints = 500
    wide = 855
    length  = 1215

    epochs  = 100

    scale = 100
    sigma = 1
    min_size = 1

    timestamp = 1

    loss_type = "mse"
    model_version = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
