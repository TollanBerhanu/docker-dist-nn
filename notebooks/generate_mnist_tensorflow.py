import tensorflow as tf
import numpy as np
import json, time

# Load and normalize the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
X_test = X_test / 255.0

# Extract 5 example inputs and flatten them to 784-element lists
examples = [X_train[i].flatten().tolist() for i in range(5)]

# Define the FCNN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784
    tf.keras.layers.Dense(128, activation='relu'),  # First hidden layer
    tf.keras.layers.Dense(64, activation='relu'),   # Second hidden layer
    tf.keras.layers.Dense(10, activation='softmax') # Output layer (10 digits)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, epochs=5, verbose=1)

start_time = time.time()

# Run inference on the first example input
first_example = X_train[0:1]  # Shape: (1, 28, 28)
predictions = model.predict(first_example, verbose=0)
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}, True class: {y_train[0]}")

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# Extract model weights and format them according to the specified structure
# layers_list = []
# for layer in model.layers:
#     if len(layer.get_weights()) > 0:  # Skip layers without weights (e.g., Flatten)
#         weights, biases = layer.get_weights()
#         num_nodes = biases.shape[0]
#         neurons = []
#         for j in range(num_nodes):
#             neuron_weights = weights[:, j].tolist()
#             neuron_bias = float(biases[j])
#             if layer.activation == tf.keras.activations.relu:
#                 activation = "relu"
#             elif layer.activation == tf.keras.activations.softmax:
#                 activation = "softmax"
#             else:
#                 raise ValueError("Unknown activation function")
#             neurons.append({
#                 "weights": neuron_weights,
#                 "bias": neuron_bias,
#                 "activation": activation
#             })
#         # Determine layer type: last layer with weights is "output", others are "hidden"
#         layer_type = "output" if layer == model.layers[-1] else "hidden"
#         layers_list.append({
#             "type": layer_type,
#             "nodes": num_nodes,
#             "neurons": neurons
#         })

# # Create the final dictionary with examples and model weights
# final_dict = {
#     "examples": examples,
#     "model": {"layers": layers_list}
# }

# # Save to a JSON file
# # with open('mnist_model_and_examples.json', 'w') as f:
# #     json.dump(final_dict, f, indent=4)

# # print("Example inputs and model weights have been saved to 'mnist_model_and_examples.json'")