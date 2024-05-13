import os
import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 100  
        self.l1 = nn.Sequential(nn.Linear(self.init_size, 256 * 27 * 38))  

        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, 27, 38)
        out = self.res_blocks(out)
        img = self.conv_blocks(out)
        img = img[:, :, :855, :1215] 
        return img


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Adjust the size of the fully connected layer to match the output of the convolutional layer
        self.fc = nn.Linear(512 * 26 * 37, 1)
        # Note: The Critic in WGAN does not use a Sigmoid activation function

    def forward(self, img):
        conv_out = self.conv_layers(img)
        flat_out = conv_out.view(img.shape[0], -1)
        validity = self.fc(flat_out)
        return validity

def generate_temperature_matrices(generator_path, n_samples=10, data_min=0, data_range=88):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    generated_matrices = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, 100).to(device)  # Generate a single sample
            gen_matrix = generator(z).cpu().numpy()
            # Manually apply the inverse transformation
            gen_matrix = (gen_matrix * data_range) + data_min
            generated_matrices.append(gen_matrix)

    generated_matrices_np = np.concatenate(generated_matrices, axis=0)
    np.save(f'./source_{num_timepoints}_sample_t2_{sample}_gan.npy', generated_matrices_np)  # Replace with the actual path

    return generated_matrices_np

def generate_temperature_matrices_batch(generator_path, total_samples, batch_size, data_min, data_range):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    generated_matrices = []
    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            batch_samples = min(batch_size, total_samples - start_idx)
            z = torch.randn(batch_samples, 100).to(device) 
            gen_matrix = generator(z).cpu().numpy()
            
            gen_matrix = (gen_matrix * data_range) + data_min
            generated_matrices.append(gen_matrix)
            
            np.save(f'./all_96407/generated_data_{start_idx}_{start_idx+batch_samples}.npy', gen_matrix)

def train_wgan(generator, critic, dataloader, epochs, device, save_model_path, clip_value=0.01, n_critic=1):
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr) #0.001 0.00005
    optimizer_C = torch.optim.RMSprop(critic.parameters(), lr=lr) #0.05  0.00005

    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.to(device)

            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()

            # Calculate the Critic score for real
            critic_real = critic(real_imgs)
            # Generate fake
            z = torch.randn(imgs.size(0), 100, device=device)
            gen_imgs = generator(z)
            # Calculate the Critic score for fake
            critic_fake = critic(gen_imgs.detach())

            # Compute the loss for the Critic
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic_loss.backward()
            optimizer_C.step()

            # Clip the weights of the Critic to meet the Lipschitz constraint
            for param in critic.parameters():
                param.data.clamp_(-clip_value, clip_value)

            # Train the Generator after training the Critic n_critic times
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate fake 
                gen_imgs = generator(z)
                # Calculate the Critic score for fake
                critic_gen = critic(gen_imgs)
                # Compute the loss for the Generator
                gen_loss = -torch.mean(critic_gen)
                gen_loss.backward()
                optimizer_G.step()

                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [Critic loss: {critic_loss.item()}] [G loss: {gen_loss.item()}]")

    torch.save(generator.state_dict(), f"{save_model_path}/generator_best.pth")
    # torch.save(discriminator.state_dict(), f"{save_model_path}/discriminator_best.pth")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化生成器和判别器
    generator = Generator().to(device)
    # discriminator = Discriminator().to(device)
    critic = Critic().to(device)

    save_model_path = "./saved_models_lr8e5_E10K/"
    model_filename = f'generator_best.pth'
    generator_path = os.path.join(save_model_path, model_filename)

    #LOAD data
    temperature_data = np.fromfile('/path/to/your/sample_t2.dat', dtype=np.float32).reshape(num_timepoints, 1, wide, length)
    print("temperature_data.shape: ", temperature_data.shape)

    # data scaler
    scaler = MinMaxScaler()
    temperature_data_scaler = scaler.fit_transform(temperature_data.reshape(-1, 1)).reshape(num_timepoints, 1, wide, length)
    print("temperature_data_scaler.shape: ", temperature_data_scaler.shape)

    data_min = scaler.data_min_[0]
    data_range = scaler.data_max_[0] - data_min
    print('data_min: ', data_min)
    print('data_range: ', data_range)

    if os.path.exists(generator_path):
        print("Model exists. Generating data...")
        generate_temperature_matrices(generator_path, n_samples=sample, data_min=data_min, data_range=data_range)
        # generate_temperature_matrices_batch(generator_path, sample, sample_batch, data_min, data_range)
    else:
        print("Model not found. Training...")
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        temperature_data_tensor = torch.tensor(temperature_data_scaler, dtype=torch.float32)

        data_loader = DataLoader(temperature_data_tensor, batch_size=batch_size, shuffle=False)

        train_wgan(generator, critic, data_loader, epochs, device, save_model_path, clip_value=0.01, n_critic=1)
        print("Training completed. Generating data...")
        generate_temperature_matrices(generator_path, n_samples=sample, data_min=data_min, data_range=data_range)
        # generate_temperature_matrices_batch(generator_path, sample, sample_batch, data_min, data_range)


if __name__ == "__main__":
    
    num_timepoints = 500 #500  4000
    wide = 855
    length  = 1215
    
    batch_size = 32
    epochs  = 10000
    lr = 0.00008

    sample = 10000
    sample_batch = 500

    main()
