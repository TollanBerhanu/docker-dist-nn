import numpy as np
import json, time

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Compute softmax over a 1-D numpy array
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def linear(x):
    return x

# Map activation names to functions
activation_map = {
    "relu": relu,
    "softmax": softmax,
    "linear": linear
}

def forward_pass(network_config, input_vector):
    """
    Performs a forward pass through the network.
    
    Args:
        network_config (dict): JSON dictionary containing the network configuration.
        input_vector (list or np.array): The input features.
        
    Returns:
        np.array: The output of the network.
    """
    # Start with the input vector
    a = np.array(input_vector)
    
    # Iterate through each layer in the configuration
    for idx, layer in enumerate(network_config['layers']):
        # Prepare a list to collect the pre-activation outputs (z) of all neurons in the layer
        z_values = []
        # Extract the activations of neurons in the layer
        activations_in_layer = [neuron.get('activation', 'linear') for neuron in layer['neurons']]
        # Check if the entire layer uses softmax (typically used for output layers)
        use_softmax_layer = all(act.lower() == 'softmax' for act in activations_in_layer)
        
        # Compute the linear combination for each neuron: dot(inputs, weights) + bias
        for neuron in layer['neurons']:
            weights = np.array(neuron['weights'])
            bias = neuron['bias']
            # Ensure the dimensions match: len(weights) should equal the size of a
            if len(weights) != a.shape[0]:
                raise ValueError(f"Dimension mismatch in layer {idx}: input dimension {a.shape[0]} does not match "
                                 f"number of weights {len(weights)}")
            z = np.dot(a, weights) + bias
            z_values.append(z)
        
        z_values = np.array(z_values)
        
        # Apply activation: if the entire layer uses softmax, then apply it on the whole vector
        if use_softmax_layer:
            a = softmax(z_values)
        else:
            # Otherwise, apply each neuron's activation individually.
            # This assumes that all neurons in the layer may use the same or different activations.
            a = np.array([
                activation_map.get(neuron.get('activation', 'linear').lower(), linear)(z)
                for neuron, z in zip(layer['neurons'], z_values)
            ])
    
    return a


CONFIG_FILE = "config/config_mnist.json"
INPUTS_FILE = "config/example_inputs/example_inputs_mnist.json"

# Load the JSON configurations from files
with open(CONFIG_FILE, 'r') as config_file:
    config_json = config_file.read()

with open(INPUTS_FILE, 'r') as inputs_file:
    examples_json = inputs_file.read()
    
# Load the JSON configurations
network_config = json.loads(config_json)
examples = json.loads(examples_json)['examples']

# Run inference on each example input
sum_inference_time = 0
for example in examples:
    start_time = time.time()
    output = forward_pass(network_config, example)
    end_time = time.time()
    inference_time = end_time - start_time
    sum_inference_time += inference_time
    print(f"Inference time: {inference_time:.4f} seconds")
    # print(f"Input: {example}")
    # print(f"Network output: {output}\n")
print(f"Total inference time: {sum_inference_time:.4f} seconds")
print(f"Average inference time: {sum_inference_time / len(examples):.4f} seconds")

