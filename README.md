# Brain Scanner from a Bicycle Helmet
## This project uses Functional Near-Infrared Spectroscopy (fNIRS) and a machine learning model to classify neural responses during different physical activities. It's all about strapping a bunch of sensors to a bike helmet and letting machine learning do its thing in brain scanner land.
### Machine Learning Model

### Feature Extraction

Here’s the magic:
- Moving Average
- Rate of Change
- Double Derivative
- Variance
- Standard Deviation
- FFT
- Energy
- Entropy

### Model Architecture

Fancy words for cool stuff:
- Two GCNConv layers for graph convolution
- An LSTM layer to capture temporal dependencies
- A fully connected layer for classification

### Training and Evaluation

I used cross-validation and some metrics to make sure this thing works.

```python
# Example Feature Extraction Function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Example Model Definition
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

    def forward(self, data, hidden=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)

        if hidden is None:
            batch_size = batch.max().item() + 1
            hidden = (torch.zeros(1, batch_size, self.hidden_dim),
                      torch.zeros(1, batch_size, self.hidden_dim))

        x = x.view(batch.max().item() + 1, -1, x.size(1))
        x, hidden = self.lstm(x, hidden)
        x = x[:, -1, :]
        x = self.fc(x)
        return F.log_softmax(x, dim=1), hidden
