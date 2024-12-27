# # Imports
# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# from scipy.fft import fft
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.data import Data, DataLoader
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import time

# # Define sensor positions (in mm) and distance threshold for edge creation
# sensor_positions = {
#     "A1": (0, 0), "A2": (7.5, 0), "A3": (15.1, 0), "A4": (22.6, 0),
#     "A5": (22.6, -30.1), "A6": (30.1, -30.1), "A7": (50.9, -30.1), "A8": (58.4, -30.1),
#     "A9": (30.8, -6.5), "A10": (38.3, -6.5), "A11": (44.4, -6.5), "A12": (51.9, -6.5),
#     "A13": (31.2, -37.0), "A14": (38.7, -37.0), "A15": (44.0, -37.0),
#     "B14": (55.7, -37.0), "B13": (63.2, -37.0), "B12": (86.5, -5.4), "B11": (94.0, -5.4),
#     "B10": (105.0, -5.4), "B9": (112.5, -5.4), "B8": (78.8, -22.4), "B7": (86.3, -22.4),
#     "B6": (128.8, -22.4), "B5": (136.3, -22.4), "B4": (107.3, -18.9), "B3": (107.3, -26.4),
#     "B2": (107.3, -24.2), "B1": (107.3, -26.4)
# }
# distance_threshold = 150  # Adjust this as needed

# # Function to create edges based on the distance threshold
# def create_edge_index(sensor_positions, threshold):
#     sensors = list(sensor_positions.keys())
#     edge_index = []

#     for i, sensor_a in enumerate(sensors):
#         for j, sensor_b in enumerate(sensors):
#             if i < j:  # Avoid duplicate pairs and self-loops
#                 pos_a = sensor_positions[sensor_a]
#                 pos_b = sensor_positions[sensor_b]
#                 distance = math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)
#                 if distance <= threshold:
#                     edge_index.append([i, j])
#                     edge_index.append([j, i])  # Ensure undirected edges

#     return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# # Create the edge index for the GCN using the updated logic
# edge_index = create_edge_index(sensor_positions, distance_threshold)

# # Feature extraction helper functions
# def moving_average(data, window_size):
#     return np.convolve(data, np.ones(window_size), 'valid') / window_size

# def rate_of_change(data):
#     return np.diff(data, prepend=data[0])

# def double_derivative(data):
#     first_derivative = rate_of_change(data)
#     return rate_of_change(first_derivative)

# def variance(data, window_size):
#     return np.array([np.var(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))])

# def standard_deviation(data, window_size):
#     return np.array([np.std(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))])

# def compute_fft(data):
#     return np.abs(fft(data))

# def compute_energy(data):
#     return np.sum(data ** 2)

# def compute_entropy(data):
#     data_shifted = data - np.min(data) + 1e-10  # Shift data to be positive
#     data_shifted /= np.sum(data_shifted)  # Normalize to make it a probability distribution
#     return stats.entropy(data_shifted)

# # Clean data function to avoid NaN or Inf issues
# def clean_data(value, replace_value=0.0):
#     if np.isnan(value) or np.isinf(value):
#         return replace_value
#     return value

# # Load the sensor data
# def load_and_preprocess_data(filename):
#     df = pd.read_csv(filename)
#     sensor_columns = list(sensor_positions.keys())
#     labels = df['Label'].values
#     sensor_data = df[sensor_columns]
#     sensor_data = sensor_data.apply(pd.to_numeric, errors='coerce')

#     if sensor_data.isnull().values.any():
#         nan_rows = sensor_data[sensor_data.isnull().any(axis=1)]
#         for index, row in nan_rows.iterrows():
#             print(f"Non-numeric or missing data found in row {index + 1}: {row.values}")
#         raise ValueError("Non-numeric data detected in sensor data after conversion. Please check the CSV file.")

#     std_dev = np.std(sensor_data.values, axis=0)
#     #print("Standard deviation of sensors:", std_dev)
#     zero_std_sensors = np.where(std_dev == 0)[0]
#     #if zero_std_sensors.size > 0:
#         #print("Sensors with zero standard deviation:", zero_std_sensors)

#     epsilon = 1e-8
#     std_dev[std_dev < epsilon] = 1.0
#     mean = np.mean(sensor_data.values, axis=0)
#     sensor_data_normalized = (sensor_data.values - mean) / std_dev

#     return labels, sensor_data_normalized

# # Preprocess the sensor data
# def preprocess_sensor_data(sensor_data, window_size=3, reset_interval=10):
#     num_sensors = sensor_data.shape[1]
#     num_time_steps = sensor_data.shape[0]

#     features = {'raw': [], 'moving_avg': [], 'rate_of_change': [], 'double_derivative': [],
#                 'variance': [], 'std_dev': [], 'fft': [], 'energy': [], 'entropy': []}

#     for sensor_idx in range(num_sensors):
#         readings = sensor_data[:, sensor_idx]
#         features['raw'].append(readings)
#         features['moving_avg'].append(moving_average(readings, window_size))
#         features['rate_of_change'].append(rate_of_change(readings))
#         features['double_derivative'].append(double_derivative(readings))
#         features['variance'].append(variance(readings, window_size))
#         features['std_dev'].append(standard_deviation(readings, window_size))
#         features['fft'].append(compute_fft(readings))
#         features['energy'].append([compute_energy(readings)] * len(readings))
#         features['entropy'].append([compute_entropy(readings)] * len(readings))

#     feature_matrix = []
#     for t in range(num_time_steps):
#         feature_vector = []
#         for sensor_idx in range(num_sensors):
#             if t < window_size - 1:
#                 moving_avg = var = std_dev = entropy = 0.0
#             else:
#                 moving_avg = features['moving_avg'][sensor_idx][t - window_size + 1]
#                 var = features['variance'][sensor_idx][t - window_size + 1]
#                 std_dev = features['std_dev'][sensor_idx][t - window_size + 1]
#                 entropy = features['entropy'][sensor_idx][t - window_size + 1]

#             feature_vector.extend([
#                 clean_data(features['raw'][sensor_idx][t]),
#                 clean_data(moving_avg),
#                 clean_data(features['rate_of_change'][sensor_idx][t]),
#                 clean_data(features['double_derivative'][sensor_idx][t]),
#                 clean_data(var),
#                 clean_data(std_dev),
#                 clean_data(features['fft'][sensor_idx][t] if t < len(features['fft'][sensor_idx]) else 0.0),
#                 clean_data(features['energy'][sensor_idx][t]),
#                 clean_data(entropy)
#             ])
#         feature_matrix.append(feature_vector)

#     feature_matrix = np.array(feature_matrix)
#     reset_indices = np.arange(0, num_time_steps, reset_interval)
#     for reset_idx in reset_indices:
#         if reset_idx < num_time_steps:
#             feature_matrix[reset_idx] = 0.0

#     return feature_matrix

# # Load and preprocess the data
# labels, sensor_data_normalized = load_and_preprocess_data("BrainScan_SensorData 12252024.csv")
# processed_data = preprocess_sensor_data(sensor_data_normalized)

# # Map labels to integers
# label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
# label_indices = np.array([label_map[label] for label in labels])

# # Create PyG Data objects
# data_list = []
# for t in range(len(processed_data)):
#     if not np.isnan(processed_data[t]).any() and not np.isinf(processed_data[t]).any():
#         data_list.append(Data(
#             x=torch.tensor(processed_data[t], dtype=torch.float).view(len(sensor_positions), -1),
#             edge_index=edge_index,
#             y=torch.tensor(label_indices[t], dtype=torch.long)
#         ))
#     else:
#         print(f"Skipping time step {t} due to NaN or Inf values.")

# print(f"Total data points: {len(processed_data)}")
# print(f"Data points without NaNs: {len(data_list)}")

# # Verify the DataLoader
# if len(data_list) == 0:
#     raise ValueError("No valid data points available after preprocessing. Please check your data and preprocessing steps.")

# loader = DataLoader(data_list, batch_size=1, shuffle=True)

# # Model Definition
# class GNNLSTMModel(nn.Module):
#     def __init__(self, num_node_features, hidden_dim, num_classes):
#         super(GNNLSTMModel, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(0.5)
#         self.hidden_dim = hidden_dim
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, data, hidden=None):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = global_mean_pool(x, batch)

#         if hidden is None:
#             batch_size = batch.max().item() + 1
#             hidden = (torch.zeros(1, batch_size, self.hidden_dim),
#                       torch.zeros(1, batch_size, self.hidden_dim))

#         x = x.view(batch.max().item() + 1, -1, x.size(1))
#         x, hidden = self.lstm(x, hidden)
#         x = x[:, -1, :]
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1), hidden

# # Function to save model, metrics, and plot confusion matrix
# def save_model_and_metrics(model, optimizer, metrics, iteration):
#     model_filename = f"fNIRS_GNNLSTM_Model_{iteration}.pth"
#     metrics_filename = f"fNIRS_ConfMatrix_Model_{iteration}.png"
#     metrics_txt_filename = f"fNIRS_Metrics_Model_{iteration}.txt"

#     torch.save(model.state_dict(), model_filename)
#     print(f"Model saved as {model_filename}")

#     fig, ax = plt.subplots()
#     conf_matrix = confusion_matrix(metrics['y_true'], metrics['y_pred'])
#     cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
#     fig.colorbar(cax)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig(metrics_filename)
#     plt.close(fig)
#     print(f"Confusion matrix saved as {metrics_filename}")

#     with open(metrics_txt_filename, 'w') as f:
#         for key, value in metrics.items():
#             if key not in ['y_true', 'y_pred']:
#                 f.write(f"{key}: {value:.4f}\n")
#     print(f"Metrics saved as {metrics_txt_filename}")

# # Training and Evaluation Function
# def train_and_evaluate(model, data_list, k_folds=5, num_iterations=3):
#     for iteration in range(1, num_iterations + 1):
#         print(f"\nStarting iteration {iteration}...")
#         start_time = time.time()

#         kf = KFold(n_splits=k_folds, shuffle=True)
#         all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'y_true': [], 'y_pred': []}

#         for fold_index, (train_index, test_index) in enumerate(kf.split(data_list), 1):
#             print(f"\n  Starting fold {fold_index} of iteration {iteration}...")
#             fold_start_time = time.time()

#             train_data = [data_list[i] for i in train_index]
#             test_data = [data_list[i] for i in test_index]
#             train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#             test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

#             optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#             criterion = nn.CrossEntropyLoss()

#             # Training
#             model.train()
#             for epoch in range(10):
#                 epoch_start_time = time.time()
#                 epoch_losses = []

#                 for data in train_loader:
#                     optimizer.zero_grad()
#                     hidden = (torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim),
#                               torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim))
#                     out, hidden = model(data, hidden)
#                     loss = criterion(out, data.y)
#                     if torch.isnan(loss) or torch.isinf(loss):
#                         print(f"NaN or Inf in loss at batch {data.batch}")
#                         continue
#                     loss.backward()
#                     optimizer.step()
#                     epoch_losses.append(loss.item())

#                 avg_epoch_loss = np.mean(epoch_losses)
#                 epoch_end_time = time.time()
#                 print(f"    Epoch {epoch + 1} of fold {fold_index} completed in {epoch_end_time - epoch_start_time:.2f} seconds. Loss: {avg_epoch_loss:.4f}")

#             # Evaluation
#             model.eval()
#             y_true, y_pred = [], []
#             with torch.no_grad():
#                 for data in test_loader:
#                     hidden = (torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim),
#                               torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim))
#                     out, hidden = model(data, hidden)
#                     y_true.append(data.y.item())
#                     y_pred.append(out.argmax(dim=1).item())

#             accuracy = accuracy_score(y_true, y_pred)
#             precision = precision_score(y_true, y_pred, average='weighted')
#             recall = recall_score(y_true, y_pred, average='weighted')
#             f1 = f1_score(y_true, y_pred, average='weighted')

#             all_metrics['accuracy'].append(accuracy)
#             all_metrics['precision'].append(precision)
#             all_metrics['recall'].append(recall)
#             all_metrics['f1_score'].append(f1)
#             all_metrics['y_true'].extend(y_true)
#             all_metrics['y_pred'].extend(y_pred)

#             fold_end_time = time.time()
#             print(f"  Fold {fold_index} of iteration {iteration} completed in {fold_end_time - fold_start_time:.2f} seconds. Accuracy: {accuracy:.4f}")

#         avg_metrics = {key: np.mean(values) if key not in ['y_true', 'y_pred'] else values
#                        for key, values in all_metrics.items()}
#         avg_metrics_std = {key: np.std(values) if key not in ['y_true', 'y_pred'] else values
#                            for key, values in all_metrics.items()}

#         for metric, values in avg_metrics.items():
#             if metric not in ['y_true', 'y_pred']:
#                 print(f"Iteration {iteration} - {metric}: {values:.4f} ± {avg_metrics_std[metric]:.4f}")

#         save_model_and_metrics(model, optimizer, avg_metrics, iteration)

#         iteration_end_time = time.time()
#         print(f"\nIteration {iteration} completed in {iteration_end_time - start_time:.2f} seconds.\n")

# # Create and train the model
# num_classes = len(label_map)
# num_node_features = processed_data.shape[1] // len(sensor_positions)
# model = GNNLSTMModel(num_node_features=num_node_features, hidden_dim=64, num_classes=num_classes)
# train_and_evaluate(model, data_list)



#####################################
#            IMPORTS
#####################################
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import fft
import math
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Traditional ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import joblib

#####################################
#        SENSOR & EDGE CREATION
#####################################
sensor_positions = {
    "A1": (0, 0), "A2": (7.5, 0), "A3": (15.1, 0), "A4": (22.6, 0),
    "A5": (22.6, -30.1), "A6": (30.1, -30.1), "A7": (50.9, -30.1), "A8": (58.4, -30.1),
    "A9": (30.8, -6.5), "A10": (38.3, -6.5), "A11": (44.4, -6.5), "A12": (51.9, -6.5),
    "A13": (31.2, -37.0), "A14": (38.7, -37.0), "A15": (44.0, -37.0),
    "B14": (55.7, -37.0), "B13": (63.2, -37.0), "B12": (86.5, -5.4), "B11": (94.0, -5.4),
    "B10": (105.0, -5.4), "B9": (112.5, -5.4), "B8": (78.8, -22.4), "B7": (86.3, -22.4),
    "B6": (128.8, -22.4), "B5": (136.3, -22.4), "B4": (107.3, -18.9), "B3": (107.3, -26.4),
    "B2": (107.3, -24.2), "B1": (107.3, -26.4)
}
distance_threshold = 150

def create_edge_index(sensor_positions, threshold):
    sensors = list(sensor_positions.keys())
    edge_index = []
    for i, sensor_a in enumerate(sensors):
        for j, sensor_b in enumerate(sensors):
            if i < j:  # Avoid duplicate pairs and self-loops
                pos_a = sensor_positions[sensor_a]
                pos_b = sensor_positions[sensor_b]
                distance = math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)
                if distance <= threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Undirected
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

edge_index = create_edge_index(sensor_positions, distance_threshold)

#####################################
#      FEATURE-EXTRACTION UTILS
#####################################
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def rate_of_change(data):
    return np.diff(data, prepend=data[0])

def double_derivative(data):
    first_derivative = rate_of_change(data)
    return rate_of_change(first_derivative)

def variance(data, window_size):
    return np.array([np.var(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))])

def standard_deviation(data, window_size):
    return np.array([np.std(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))])

def compute_fft(data):
    return np.abs(fft(data))

def compute_energy(data):
    return np.sum(data ** 2)

def compute_entropy(data):
    data_shifted = data - np.min(data) + 1e-10  # Shift data to be positive
    data_shifted /= np.sum(data_shifted)        # Probability distribution
    return stats.entropy(data_shifted)

def clean_data(value, replace_value=0.0):
    if np.isnan(value) or np.isinf(value):
        return replace_value
    return value

#####################################
#       LOAD + PREPROCESS DATA
#####################################
def load_and_preprocess_data_for_gnn(filename):
    """
    Loads CSV with 'Label' and sensor columns that match `sensor_positions`.
    Excludes time from these GNN-based features but uses the same pipeline as original code.
    """
    df = pd.read_csv(filename)

    # Extract sensor columns:
    sensor_columns = list(sensor_positions.keys())
    labels = df['Label'].values

    # Convert sensor data to numeric
    sensor_data = df[sensor_columns].apply(pd.to_numeric, errors='coerce')
    if sensor_data.isnull().values.any():
        nan_rows = sensor_data[sensor_data.isnull().any(axis=1)]
        for index, row in nan_rows.iterrows():
            print(f"Non-numeric or missing data found in row {index + 1}: {row.values}")
        raise ValueError("Non-numeric data detected in sensor data after conversion.")

    # Normalize
    std_dev = np.std(sensor_data.values, axis=0)
    epsilon = 1e-8
    std_dev[std_dev < epsilon] = 1.0
    mean = np.mean(sensor_data.values, axis=0)
    sensor_data_normalized = (sensor_data.values - mean) / std_dev

    return labels, sensor_data_normalized

def preprocess_sensor_data(sensor_data, window_size=3, reset_interval=10):
    """
    Applies the original feature extraction for the GNN pipeline:
    raw, moving_avg, rate_of_change, double_derivative, variance, std, fft, energy, entropy
    """
    num_sensors = sensor_data.shape[1]
    num_time_steps = sensor_data.shape[0]

    features = {
        'raw': [], 'moving_avg': [], 'rate_of_change': [],
        'double_derivative': [], 'variance': [], 'std_dev': [],
        'fft': [], 'energy': [], 'entropy': []
    }

    # Collect per-sensor features
    for sensor_idx in range(num_sensors):
        readings = sensor_data[:, sensor_idx]
        features['raw'].append(readings)
        features['moving_avg'].append(moving_average(readings, window_size))
        features['rate_of_change'].append(rate_of_change(readings))
        features['double_derivative'].append(double_derivative(readings))
        features['variance'].append(variance(readings, window_size))
        features['std_dev'].append(standard_deviation(readings, window_size))
        features['fft'].append(compute_fft(readings))
        features['energy'].append([compute_energy(readings)] * len(readings))
        features['entropy'].append([compute_entropy(readings)] * len(readings))

    # Build per-time-step feature vectors
    feature_matrix = []
    for t in range(num_time_steps):
        feature_vector = []
        for sensor_idx in range(num_sensors):
            if t < window_size - 1:
                # Not enough samples to compute sliding-window features
                moving_avg = var = stdv = ent = 0.0
            else:
                moving_avg = features['moving_avg'][sensor_idx][t - window_size + 1]
                var = features['variance'][sensor_idx][t - window_size + 1]
                stdv = features['std_dev'][sensor_idx][t - window_size + 1]
                ent = features['entropy'][sensor_idx][t - window_size + 1]

            # Some features must handle array bounds carefully
            fft_val = features['fft'][sensor_idx][t] if t < len(features['fft'][sensor_idx]) else 0.0

            feature_vector.extend([
                clean_data(features['raw'][sensor_idx][t]),
                clean_data(moving_avg),
                clean_data(features['rate_of_change'][sensor_idx][t]),
                clean_data(features['double_derivative'][sensor_idx][t]),
                clean_data(var),
                clean_data(stdv),
                clean_data(fft_val),
                clean_data(features['energy'][sensor_idx][t]),
                clean_data(ent)
            ])
        feature_matrix.append(feature_vector)

    feature_matrix = np.array(feature_matrix)

    # Reset some intervals to zero if needed
    reset_indices = np.arange(0, num_time_steps, reset_interval)
    for reset_idx in reset_indices:
        if reset_idx < num_time_steps:
            feature_matrix[reset_idx] = 0.0

    return feature_matrix

#####################################
#    GNN + LSTM MODEL DEFINITION
#####################################
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
        self.hidden_dim = hidden_dim
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data, hidden=None):
        x, edge_idx, batch = data.x, data.edge_index, data.batch
        # 1st GCN
        x = self.conv1(x, edge_idx)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # 2nd GCN
        x = self.conv2(x, edge_idx)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Pooling
        x = global_mean_pool(x, batch)

        # LSTM
        if hidden is None:
            batch_size = batch.max().item() + 1
            hidden = (torch.zeros(1, batch_size, self.hidden_dim),
                      torch.zeros(1, batch_size, self.hidden_dim))

        x = x.view(batch.max().item() + 1, -1, x.size(1))
        x, hidden = self.lstm(x, hidden)
        x = x[:, -1, :]
        x = self.fc(x)
        return F.log_softmax(x, dim=1), hidden

#####################################
# SAVE MODEL, METRICS, & CONFUSION
#####################################
def save_model_and_metrics(model, optimizer, metrics, iteration):
    model_filename = f"fNIRS_GNNLSTM_Model_{iteration}.pth"
    metrics_filename = f"fNIRS_ConfMatrix_Model_{iteration}.png"
    metrics_txt_filename = f"fNIRS_Metrics_Model_{iteration}.txt"

    # Save the model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

    # Confusion matrix
    fig, ax = plt.subplots()
    conf_matrix = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(metrics_filename)
    plt.close(fig)
    print(f"Confusion matrix saved as {metrics_filename}")

    # Metrics to txt
    with open(metrics_txt_filename, 'w') as f:
        for key, value in metrics.items():
            if key not in ['y_true', 'y_pred']:
                f.write(f"{key}: {value:.4f}\n")
    print(f"Metrics saved as {metrics_txt_filename}")

#####################################
#       TRAIN & EVALUATE GNN
#####################################
def train_and_evaluate(model, data_list, k_folds=5, num_iterations=3):
    """
    Original training loop for the GNN+LSTM using K-fold cross-validation.
    """
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}...")
        start_time = time.time()

        kf = KFold(n_splits=k_folds, shuffle=True)
        all_metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1_score': [], 'y_true': [], 'y_pred': []
        }

        for fold_index, (train_index, test_index) in enumerate(kf.split(data_list), 1):
            print(f"\n  Starting fold {fold_index} of iteration {iteration}...")
            fold_start_time = time.time()

            train_data = [data_list[i] for i in train_index]
            test_data = [data_list[i] for i in test_index]
            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()

            # ---- TRAIN ----
            model.train()
            for epoch in range(10):
                epoch_start_time = time.time()
                epoch_losses = []

                for data in train_loader:
                    optimizer.zero_grad()
                    hidden = (torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim),
                              torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim))
                    out, hidden = model(data, hidden)
                    loss = criterion(out, data.y)
                    # Skip if loss is NaN or Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN or Inf in loss at batch {data.batch}")
                        continue

                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

                avg_epoch_loss = np.mean(epoch_losses)
                epoch_end_time = time.time()
                print(f"    Epoch {epoch + 1} of fold {fold_index} | Loss: {avg_epoch_loss:.4f} | "
                      f"Duration: {epoch_end_time - epoch_start_time:.2f}s")

            # ---- EVALUATE ----
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for data in test_loader:
                    hidden = (torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim),
                              torch.zeros(1, data.batch.max().item() + 1, model.hidden_dim))
                    out, hidden = model(data, hidden)
                    y_true.append(data.y.item())
                    y_pred.append(out.argmax(dim=1).item())

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
            print(f"  Fold {fold_index} completed | Accuracy: {accuracy:.4f} | "
                  f"Duration: {fold_end_time - fold_start_time:.2f}s")

        # Summarize results over the folds
        avg_metrics = {
            key: np.mean(values) if key not in ['y_true', 'y_pred'] else values
            for key, values in all_metrics.items()
        }
        avg_metrics_std = {
            key: np.std(values) if key not in ['y_true', 'y_pred'] else values
            for key, values in all_metrics.items()
        }

        for metric, values in avg_metrics.items():
            if metric not in ['y_true', 'y_pred']:
                print(f"Iteration {iteration} - {metric}: {values:.4f} ± {avg_metrics_std[metric]:.4f}")

        # Save model & metrics
        save_model_and_metrics(model, optimizer, avg_metrics, iteration)

        iteration_end_time = time.time()
        print(f"\nIteration {iteration} completed in {iteration_end_time - start_time:.2f} seconds.\n")

#####################################
#       ADDITIONAL ML MODELS
#####################################
def train_ml_models(X_train, X_test, y_train, y_test, feature_set_name):
    """
    Trains several ML algorithms and saves each under
    BrainScan_{MODELNAME}_{feature_set_name}_12262024.pkl
    """
    # 7 models: RF, GB, SVM, KNN, LR, XGB, MLP
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True)),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200))
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} Classifier ({feature_set_name}) Report:")
        print(classification_report(y_test, y_pred))
        # Save each model
        joblib.dump(model, f"BrainScan_{name}_{feature_set_name}_12262024.pkl")

#####################################
#               MAIN
#####################################
if __name__ == "__main__":
    #############################
    # 1) GNN + LSTM Pipeline
    #############################
    # Example CSV: "BrainScan_SensorData_12262024.csv"
    # Load data for the GNN
    gnn_labels, gnn_sensor_data_normalized = load_and_preprocess_data_for_gnn("BrainScan_SensorData 12252024.csv")
    gnn_processed_data = preprocess_sensor_data(gnn_sensor_data_normalized)

    # Map labels to integers
    gnn_label_map = {label: idx for idx, label in enumerate(np.unique(gnn_labels))}
    gnn_label_indices = np.array([gnn_label_map[label] for label in gnn_labels])

    # Create PyG Data objects
    data_list = []
    for t in range(len(gnn_processed_data)):
        row = gnn_processed_data[t]
        if not np.isnan(row).any() and not np.isinf(row).any():
            data_list.append(
                Data(
                    x=torch.tensor(row, dtype=torch.float).view(len(sensor_positions), -1),
                    edge_index=edge_index,
                    y=torch.tensor(gnn_label_indices[t], dtype=torch.long)
                )
            )
        else:
            print(f"Skipping time step {t} due to NaN or Inf values.")

    if len(data_list) == 0:
        raise ValueError("No valid data points after GNN preprocessing.")

    # Build and train the GNN+LSTM model
    num_classes_gnn = len(gnn_label_map)
    num_node_features_gnn = gnn_processed_data.shape[1] // len(sensor_positions)
    gnn_model = GNNLSTMModel(num_node_features=num_node_features_gnn, hidden_dim=64, num_classes=num_classes_gnn)
    train_and_evaluate(gnn_model, data_list, k_folds=5, num_iterations=3)


    #############################
    # 2) Additional ML Models
    #############################
    # For these, we treat each time step as one sample,
    # with optional "Elapsed Time (s)" as an extra feature.
    # We'll assume your CSV also has "Elapsed Time (s)".
    df_ml = pd.read_csv("BrainScan_SensorData 12252024.csv")
    # Must have 'Label' and 'Elapsed Time (s)' columns
    ml_labels = df_ml['Label'].values
    ml_elapsed_time = df_ml['Elapsed Time (s)'].values

    # We'll reuse the same gnn_processed_data as our base sensor features
    # but we can incorporate time as an additional column in X_with_time.

    # Map labels
    ml_label_map = {label: idx for idx, label in enumerate(np.unique(ml_labels))}
    ml_y = np.array([ml_label_map[label] for label in ml_labels])

    # If gnn_processed_data has shape (num_samples, num_features), let's do:
    X_without_time = gnn_processed_data

    # Construct X_with_time by combining the "elapsed_time" column
    # with the gnn_processed_data
    # Make sure shapes align (the dataset must have the same # of rows as gnn_processed_data).
    if len(ml_elapsed_time) == len(gnn_processed_data):
        X_with_time = np.hstack((ml_elapsed_time.reshape(-1, 1), gnn_processed_data))
    else:
        # Fallback in case of mismatch, you can handle it as needed
        print("Warning: Elapsed Time count doesn't match sensor data. Falling back to no-time approach.")
        X_with_time = X_without_time

    # Standard train/test splits
    X_train_with_time, X_test_with_time, y_train, y_test = train_test_split(
        X_with_time, ml_y, test_size=0.2, random_state=42
    )
    X_train_without_time, X_test_without_time, y_train2, y_test2 = train_test_split(
        X_without_time, ml_y, test_size=0.2, random_state=42
    )

    # Train ML models on both sets
    print("\n===============================")
    print("ML Models (WithTime) Dataset:")
    print("===============================")
    train_ml_models(X_train_with_time, X_test_with_time, y_train, y_test, "WithTime")

    print("\n===============================")
    print("ML Models (WithoutTime) Dataset:")
    print("===============================")
    train_ml_models(X_train_without_time, X_test_without_time, y_train2, y_test2, "WithoutTime")
