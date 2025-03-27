import docker
import socket
import json
import time
from tqdm import tqdm

# Load Inputs
with open("config/example_inputs/example_inputs_sample.json", "r") as f:
    inputs = json.load(f)["examples"]
input_values = inputs[1]
num_input = len(input_values)

# Load configuration to compute neuron mappings
with open("config/config_sample.json", "r") as f:
    config = json.load(f)
layers = config["layers"]

BASE_PORT_FIRST_HIDDEN = 4100
BASE_PORT_INCREMENT = 500
# The first inference is sent to the first hidden layer (layer index 1)
layer_index = 1
layer = layers[layer_index]
base_port = BASE_PORT_FIRST_HIDDEN + (layer_index * BASE_PORT_INCREMENT)
first_hidden_mapping = []
for j in range(layer["nodes"]):
    port = base_port + j + 1
    first_hidden_mapping.append(port)

start_inference = time.time()
# Send each input value to every neuron in the first hidden layer
for port in first_hidden_mapping:
    for idx, val in enumerate(input_values):
        msg = json.dumps({"value": val, "index": idx}).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", port))
            s.sendall(msg)
        time.sleep(0.5)

# Wait to allow inference processing (adjust delay as needed)
# time.sleep(5)

inference_time = time.time() - start_inference
print("Inference processing time: {:.3f} seconds".format(inference_time))

# Cleanup resources: stop & remove neural network containers and network.
# client = docker.from_env()
# network_name = "fcnn_network"
# for container in tqdm(client.containers.list(filters={"name": "nn_hidden_"}), desc="Stopping hidden containers"):
#     container.stop()
#     container.remove()
# for container in tqdm(client.containers.list(filters={"name": "nn_output_"}), desc="Stopping output containers"):
#     container.stop()
#     container.remove()

# try:
#     network = client.networks.get(network_name)
#     network.remove()
#     print("Network {} removed.".format(network_name))
# except Exception as e:
#     print("Error removing network {}: {}".format(network_name, e))