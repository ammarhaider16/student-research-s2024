import numpy as np
np.bool = np.bool_
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


df = pd.read_csv("../Datasets/Valuation/automobile.csv")
df = df.drop(columns = ['symboling', 'normalized-losses','num-of-doors'])

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


# Normalize the input data
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['price']).values)
y = df['price'].values  # Assuming target does not need scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for RNN input: [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# RNN MODEL SET UP USING TENSOR FLOW
class SimpleRNNModelTensorFlow(tf.keras.Model):
    def __init__(self):
        super(SimpleRNNModelTensorFlow, self).__init__()
        self.rnn_layer = tf.keras.layers.SimpleRNN(50, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        rnn_output = self.rnn_layer(inputs)
        return self.output_layer(rnn_output)
    
# Create an instance of the model
model = SimpleRNNModelTensorFlow()

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, validation_split=0.2)

# Make predictions
y_pred = model(X_test)

# Calculate Mean Squared Error or any other metric
mse_tf = np.mean(np.square(y_test - y_pred))


# RNN MODEL SETUP USING PyTorch
class SimpleRNNModelPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNModelPyTorch, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Set up the model
input_size = X_train.shape[2]  # Number of features
hidden_size = 50  # Number of units in the RNN layer
output_size = 1  # Output size (common stock value)
model = SimpleRNNModelPyTorch(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the model
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
y_pred = y_pred_tensor.numpy()
mse_pytorch = mean_squared_error(y_test, y_pred)



print("Mean Squared Error (TensorFlow):", mse_tf)
print("Mean Squared Error (PyTorch):", mse_pytorch)