import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define a fully connected neural network
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Extract model parameters
config = {"layers": []}

# Input layer
config["layers"].append({
    "type": "input",
    "nodes": X.shape[1],
    "values": [0] * X.shape[1]  # Placeholder values
})

# Hidden and output layers
for layer in model.layers:
    if isinstance(layer, keras.layers.Dense):
        weights, biases = layer.get_weights()
        num_nodes = weights.shape[1]
        activation = layer.activation.__name__
        
        neurons = []
        for i in range(num_nodes):
            neuron_weights = weights[:, i].tolist()
            neuron_bias = biases[i].item()
            neurons.append({"weights": neuron_weights, "bias": neuron_bias, "activation": activation})
        
        layer_type = "output" if num_nodes == 1 else "hidden"
        config["layers"].append({
            "type": layer_type,
            "nodes": num_nodes,
            "neurons": neurons
        })

# Save to config.json
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("config.json file generated successfully!")

# Save example inputs to example_inputs.json
example_inputs = {"examples": X_test[:5].tolist()}
with open("example_inputs.json", "w") as f:
    json.dump(example_inputs, f, indent=2)

print("example_inputs.json file generated successfully!")
