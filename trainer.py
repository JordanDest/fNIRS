import numpy as np
import scipy.stats as stats
from scipy.fft import fft
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import time

# Feature extraction helper functions
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def rate_of_change(data):
    return np.diff(data, prepend=data[0])

def double_derivative(data):
    first_derivative = rate_of_change(data)
    return rate_of_change(first_derivative)

def variance(data, window_size):
    var_list = [np.var(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]
    return np.array(var_list)

def standard_deviation(data, window_size):
    std_list = [np.std(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]
    return np.array(std_list)

def compute_fft(data):
    return np.abs(fft(data))

def compute_energy(data):
    return np.sum(data ** 2)

def compute_entropy(data):
    return stats.entropy(data)

def preprocess_sensor_data(sensor_data, window_size=3, reset_interval=10):
    num_sensors = len(sensor_data)
    num_time_steps = len(next(iter(sensor_data.values())))

    # Normalize sensor data
    sensor_data = {sensor: (readings - np.mean(readings)) / np.std(readings) for sensor, readings in sensor_data.items()}

    # Initialize feature arrays
    features = {sensor: {'raw': [], 'moving_avg': [], 'rate_of_change': [], 'double_derivative': [],
                         'variance': [], 'std_dev': [], 'fft': [], 'energy': [], 'entropy': []}
                for sensor in sensor_data}

    # Process each sensor's data
    for sensor, readings in sensor_data.items():
        readings = np.array(readings)

        # Compute features
        features[sensor]['raw'] = readings
        features[sensor]['moving_avg'] = moving_average(readings, window_size)
        features[sensor]['rate_of_change'] = rate_of_change(readings)
        features[sensor]['double_derivative'] = double_derivative(readings)
        features[sensor]['variance'] = variance(readings, window_size)
        features[sensor]['std_dev'] = standard_deviation(readings, window_size)
        features[sensor]['fft'] = compute_fft(readings)
        features[sensor]['energy'] = compute_energy(readings)
        features[sensor]['entropy'] = compute_entropy(readings)

    # Combine features into a single feature matrix for each time step
    feature_matrix = []
    for t in range(num_time_steps):
        feature_vector = []
        for sensor in sensor_data:
            if t < window_size - 1:
                # Not enough data to compute moving average, variance, std_dev, entropy
                moving_avg = np.nan
                var = np.nan
                std_dev = np.nan
                entropy = np.nan
            else:
                moving_avg = features[sensor]['moving_avg'][t - window_size + 1]
                var = features[sensor]['variance'][t - window_size + 1]
                std_dev = features[sensor]['std_dev'][t - window_size + 1]
                entropy = features[sensor]['entropy'][t - window_size + 1]

            feature_vector.extend([
                features[sensor]['raw'][t],
                moving_avg,
                features[sensor]['rate_of_change'][t],
                features[sensor]['double_derivative'][t],
                var,
                std_dev,
                features[sensor]['fft'][t] if t < len(features[sensor]['fft']) else np.nan,
                features[sensor]['energy'],
                entropy
            ])
        feature_matrix.append(feature_vector)

    # Reset the features every reset_interval
    feature_matrix = np.array(feature_matrix)
    reset_indices = np.arange(0, num_time_steps, reset_interval)
    for reset_idx in reset_indices:
        if reset_idx < num_time_steps:
            feature_matrix[reset_idx] = np.nan  # Indicating a reset

    return feature_matrix

# Example usage
sensor_names = [chr(65 + i) for i in range(26)] + ['AA', 'AB']  # Sensors A to Z, AA, AB
num_time_steps = 100  # Number of time steps

# Generate random data for each sensor
sensor_data = {name: np.random.randint(100, 800, num_time_steps) for name in sensor_names}

# Preprocess the sensor data
processed_data = preprocess_sensor_data(sensor_data)

# Creating edge index for connecting each node to its two left and two right neighbors
edge_index = []
num_sensors = len(sensor_names)

for i in range(num_sensors):
    if i - 2 >= 0:
        edge_index.append([i, i - 2])
        edge_index.append([i - 2, i])
    if i - 1 >= 0:
        edge_index.append([i, i - 1])
        edge_index.append([i - 1, i])
    if i + 1 < num_sensors:
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    if i + 2 < num_sensors:
        edge_index.append([i, i + 2])
        edge_index.append([i + 2, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Creating PyG Data object
data_list = []
for t in range(len(processed_data)):
    if not np.isnan(processed_data[t]).any():
        data_list.append(Data(x=torch.tensor(processed_data[t], dtype=torch.float).view(num_sensors, -1), edge_index=edge_index))

# DataLoader
loader = DataLoader(data_list, batch_size=1, shuffle=True)

# GNN-LSTM Model
class GNNLSTMModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GNNLSTMModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data, hidden):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)  # Pooling to get graph-level representation

        # Reshape for LSTM
        x = x.view(batch.max().item() + 1, -1, x.size(1))
        
        # LSTM to capture temporal dependencies
        x, hidden = self.lstm(x, hidden)
        
        # Only take the output of the last LSTM cell
        x = x[:, -1, :]
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1), hidden

# Function to save the model and relevant information
def save_model_and_info(model, optimizer, metrics, iteration):
    model_filename = f"fNIRS_GNNLSTM_Model_{iteration}.pth"
    metrics_filename = f"fNIRS_ConfMatrix_Model_{iteration}.png"
    metrics_txt_filename = f"fNIRS_Metrics_Model_{iteration}.txt"
    model_info_filename = f"fNIRS_Model_Info_Model_{iteration}.pkl"
    
    # Save the model
    torch.save(model.state_dict(), model_filename)
    
    # Save the metrics plot silently
    fig, ax = plt.subplots()
    conf_matrix = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(metrics_filename)
    plt.close(fig)
    
    # Save the metrics data to a text file
    with open(metrics_txt_filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Save all other relevant information
    with open(model_info_filename, 'wb') as f:
        pickle.dump({'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'metrics': metrics}, f)

# Training loop with cross-validation and evaluation metrics
def train_and_evaluate(model, data_list, k_folds=5, num_iterations=3):
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}...")
        start_time = time.time()
        
        kf = KFold(n_splits=k_folds, shuffle=True)
        all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'y_true': [], 'y_pred': []}

        fold_index = 0
        for train_index, test_index in kf.split(data_list):
            fold_index += 1
            print(f"\n  Starting fold {fold_index} of iteration {iteration}...")
            fold_start_time = time.time()
            
            train_data = [data_list[i] for i in train_index]
            test_data = [data_list[i] for i in test_index]
            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(10):  # Number of epochs
                print(f"    Starting epoch {epoch + 1} of fold {fold_index}...")
                epoch_start_time = time.time()
                
                model.train()
                for data in train_loader:
                    optimizer.zero_grad()
                    out, hidden = model(data, hidden)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
                
                epoch_end_time = time.time()
                print(f"    Finished epoch {epoch + 1} of fold {fold_index} in {epoch_end_time - epoch_start_time:.2f} seconds.")

            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for data in test_loader:
                    out, hidden = model(data, hidden)
                    y_true.append(data.y.item())
                    y_pred.append(out.argmax(dim=1).item())

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            all_metrics['accuracy'].append(accuracy)
            all_metrics['precision'].append(precision)
            all_metrics['recall'].append(recall)
            all_metrics['f1_score'].append(f1)
            all_metrics['y_true'].extend(y_true)
            all_metrics['y_pred'].extend(y_pred)

            fold_end_time = time.time()
            print(f"  Finished fold {fold_index} of iteration {iteration} in {fold_end_time - fold_start_time:.2f} seconds.")

        avg_metrics = {key: np.mean(values) if key not in ['y_true', 'y_pred'] else values
                       for key, values in all_metrics.items()}
        avg_metrics_std = {key: np.std(values) if key not in ['y_true', 'y_pred'] else values
                           for key, values in all_metrics.items()}
        
        for metric, values in avg_metrics.items():
            if metric not in ['y_true', 'y_pred']:
                print(f"Iteration {iteration} - {metric}: {values:.4f} ± {avg_metrics_std[metric]:.4f}")
        
        save_model_and_info(model, optimizer, avg_metrics, iteration)
        
        end_time = time.time()
        print(f"Finished iteration {iteration} in {end_time - start_time:.2f} seconds.")

# Train and evaluate the model
train_and_evaluate(model, data_list)
