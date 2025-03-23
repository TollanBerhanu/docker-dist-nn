import json
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data into train and test sets FIRST to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using training data statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use scaler from training data

# Define an improved neural network for regression
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    # keras.layers.BatchNormalization(),
    keras.layers.Dense(16, activation='relu'),
    # keras.layers.Dropout(0.3),
    # keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)  # Linear activation for regression
])

# Compile with appropriate metrics (MAE/MSE)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Add early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

# Train for more epochs with validation
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate properly
print("\nFinal evaluation:")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
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

# # Save to config.json
# with open("config_cali.json", "w") as f:
#     json.dump(config, f, indent=2)

# print("config.json file generated successfully!")

# # Save example inputs to example_inputs.json
# example_inputs = {"examples": X_test[:5].tolist()}
# with open("example_inputs_cali.json", "w") as f:
#     json.dump(example_inputs, f, indent=2)

# print("example_inputs.json file generated successfully!")


import time

# Take one example from the test set and compare prediction vs true label
sample_index = 0
sample_input = X_test[sample_index]
true_label = y_test[sample_index]

# Reshape the input for prediction (model expects batch dimension)
sample_input = sample_input.reshape(1, -1)

# Start the timer
start_time = time.time()

# Predict
predicted_label = model.predict(sample_input, verbose=0)[0][0]

# End the timer
end_time = time.time()
inference_time = end_time - start_time

# Print comparison
print(f"\nTest Sample {sample_index}:")
# print(f"  - Input Features: {sample_input.flatten()}")
print(f"  - True Label: {true_label:.4f}")
print(f"  - Predicted Label: {predicted_label:.4f}")
print(f"  - Inference Time: {inference_time:.4f}")
# print(f"  - Absolute Error: {abs(true_label - predicted_label):.4f}")

