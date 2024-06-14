import numpy as np
np.bool = np.bool_
import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import backend as K
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

dict = arff.load(open('../Datasets/Valuation/gamestop_historical_financials.arff'))

title = []
for list in dict.get('attributes'):
    title.append(list[0])


df = pd.DataFrame(dict.get("data"))
df.columns = title
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handle missing values -> drop columns with more than 50% missing values and replace other missing values with the average for the column

threshold = 0.9* len(df)
cols_to_drop = df.columns[df.isnull().sum() > threshold]
print("Dropped columns: "+cols_to_drop)
df = df.drop(cols_to_drop, axis=1)

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()))

for col in cat_cols:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)

# Drop cat cols
df = df.drop(columns=cat_cols)

# Assume df is your dataframe
X = df.drop(columns=['adjclose_price']).values.astype(np.float32)  # Features
y = df['adjclose_price'].values.reshape(-1, 1).astype(np.float32)  # y

# Normalize the X (optional but often recommended)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
input_size = X_train.shape[1]
hidden_units = 10
output_size = 1

# Initialize weights and biases with float32
weights = {
    'h1': tf.Variable(tf.random.normal([input_size, hidden_units], dtype=tf.float32)),
    'out': tf.Variable(tf.random.normal([hidden_units, output_size], dtype=tf.float32))
}
biases = {
    'b1': tf.Variable(tf.random.normal([hidden_units], dtype=tf.float32)),
    'out': tf.Variable(tf.random.normal([output_size], dtype=tf.float32))
}

# Define the FNN model
@tf.function
def neural_net(x):
    # Hidden fully connected layer with ReLU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Output layer (without activation, for regression)
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    return out_layer

# Define the loss function (Mean Squared Error)
@tf.function
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the optimizer
learning_rate = 0.01
optimizer = tf.optimizers.Adam(learning_rate)

# Number of epochs and batch size
epochs = 5
batch_size = 32

# Training function
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = neural_net(x_batch)
        loss = mse_loss(y_batch, predictions)
    gradients = tape.gradient(loss, [weights['h1'], weights['out'], biases['b1'], biases['out']])
    optimizer.apply_gradients(zip(gradients, [weights['h1'], weights['out'], biases['b1'], biases['out']]))
    return loss

# Training loop
for epoch in range(epochs):
    # Shuffle and batch data
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]
    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        x_batch, y_batch = X_train[start:end], y_train[start:end]
        
        # Perform a training step
        loss = train_step(x_batch, y_batch)

    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Evaluate the model on the validation set
y_val_pred = neural_net(X_val)
validation_mse = mse_loss(y_val, y_val_pred)
print(f'Validation MSE (TensorFlow): {validation_mse.numpy()}')


# PyTorch FNN
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Feedforward Neural Network
class SimpleFNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the model
input_size = X_train.shape[1]
hidden_units = 10
output_size = 1
model = SimpleFNN(input_size, hidden_units, output_size)


# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Number of epochs
epochs = 5

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x_batch.size(0)  # Sum up batch loss
    
    # Average the loss over the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# Evaluate the model on the validation set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    y_val_pred = model(X_val_tensor)
    validation_mse = criterion(y_val_pred, y_val_tensor).item()

print(f'Validation MSE (PyTorch): {validation_mse:.4f}')

