import docker
import socket
import json
import time
import threading
import platform
from tqdm import tqdm

CONFIG_FILE = "config/config_mnist.json"
CALLBACK_PORT = 9000
BASE_PORT_FIRST_HIDDEN = 4100    # Base port for the first hidden layer
BASE_PORT_INCREMENT = 500        # Increment per layer

# Load configuration
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

layers = config["layers"]
# Input layer
input_layer = layers[0]
input_values = input_layer["values"]
num_input = input_layer["nodes"]

# Pre-calculate container mapping for each layer (all layers after input)
# Mapping: for each layer index i (i>=1), generate a list of dicts with container name and listen port.
neuron_mappings = {}
for layer_index in range(1, len(layers)):
    layer = layers[layer_index]
    num_neurons = layer["nodes"]
    # Use a different port range per layer. For layer 1, start at BASE_PORT_FIRST_HIDDEN;
    # for later layers, add BASE_PORT_INCREMENT per layer.
    base_port = BASE_PORT_FIRST_HIDDEN + (layer_index - 1) * BASE_PORT_INCREMENT
    mapping = []
    for j in range(num_neurons):
        if layer["type"] == "output":
            container_name = f"nn_output_{j}"
        else:
            # For hidden layers, include the layer index in the name
            container_name = f"nn_hidden_{layer_index-1}_{j}"
        port = base_port + j + 1
        mapping.append({"name": container_name, "port": port})
    neuron_mappings[layer_index] = mapping

# Create Docker client and network
client = docker.from_env()
network_name = "fcnn_network"
try:
    network = client.networks.create(network_name, driver="bridge")
except Exception:
    network = client.networks.get(network_name)

# For Linux, get the Docker network gateway IP; for macOS use host.docker.internal.
network_details = network.attrs
if platform.system() == "Darwin":
    gateway_ip = "host.docker.internal"
else:
    gateway_ip = network_details['IPAM']['Config'][0]['Gateway']
print("Gateway IP for callback:", gateway_ip)

containers = {}

# Spawn containers for each neuron in layers 1 to end
for layer_index in range(1, len(layers)):
    layer = layers[layer_index]
    neurons = layer.get("neurons", [])
    num_neurons = layer["nodes"]
    # Expected inputs equals the number of neurons (or input nodes) in the previous layer
    if layer_index == 1:
        expected_inputs = num_input
    else:
        expected_inputs = layers[layer_index - 1]["nodes"]
    for j in range(num_neurons):
        mapping = neuron_mappings[layer_index][j]
        container_name = mapping["name"]
        listen_port = mapping["port"]

        # Determine next nodes: if not the final layer, forward to all neurons in the next layer.
        # Otherwise (for final layer) forward to the callback server.
        if layer_index < len(layers) - 1:
            next_mapping = neuron_mappings[layer_index + 1]
            next_nodes = [{"host": m["name"], "port": str(m["port"])} for m in next_mapping]
        else:
            next_nodes = [{"host": gateway_ip, "port": str(CALLBACK_PORT)}]

        neuron_config = neurons[j]
        weights = neuron_config["weights"]
        bias = neuron_config["bias"]
        activation = neuron_config["activation"]
        env = {
            "CONTAINER_NAME": str(container_name),
            "LISTEN_PORT": str(listen_port),
            "EXPECTED_INPUTS": str(expected_inputs),
            "WEIGHTS": json.dumps(weights),
            "BIAS": str(bias),
            "ACTIVATION": activation,
            "NEXT_NODES": json.dumps(next_nodes)
        }

        # For the first hidden layer, publish the container port so that the controller can send inputs
        ports = None
        if layer_index == 1:
            ports = {f"{listen_port}/tcp": listen_port}
            print(f"Spawned hidden neuron container '{container_name}' on port {listen_port}")
        else:
            if layer["type"] == "hidden":
                print(f"Spawned hidden neuron container '{container_name}' on port {listen_port}")
            else:
                print(f"Spawned output neuron container '{container_name}' on port {listen_port}")

        container = client.containers.run(
            "fcnn_image",  # Make sure to build your image with: docker build -t fcnn_image .
            detach=True,
            name=container_name,
            network=network_name,
            environment=env,
            ports=ports
        )
        containers[container_name] = container

# Callback server to receive final outputs from output neurons
final_results = []  # Changed from final_result to a list
expected_outputs = layers[-1]["nodes"]  # Number of neurons in the output layer

def callback_server():
    global final_results
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", CALLBACK_PORT))
        s.listen()
        print(f"Callback server listening on port {CALLBACK_PORT}...")
        while len(final_results) < expected_outputs:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024).decode()
                if data:
                    msg = json.loads(data)
                    final_results.append(msg.get("value"))
                    print(f"Received from {addr}: {msg.get('value')}")

callback_thread = threading.Thread(target=callback_server, daemon=True)
callback_thread.start()

# Allow time for containers to start up
time.sleep(10)
start_time = time.time()

# For the first hidden layer, send each input value to every neuron
for mapping in neuron_mappings[1]:
    for idx, val in enumerate(input_values):
        msg = json.dumps({"value": val, "index": idx}).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", mapping["port"]))
            s.sendall(msg)
        # Short delay to help preserve order if needed
        time.sleep(0.5)

# Wait until the expected number of outputs have been received or the timeout is reached
callback_thread.join(timeout=10)
elapsed_time = time.time() - start_time

if len(final_results) == expected_outputs:
    print("Final outputs from neural network:", final_results)
    print("Total processing time: {:.3f} seconds".format(elapsed_time))
else:
    print("Not all outputs received within the timeout period.")

# Cleanup: stop and remove all containers and the network.
for container in tqdm(containers.values(), desc="Stopping and removing containers"):
    # print(f"Stopping and removing container {container.name}")
    container.stop()
    container.remove()
print("Removing network", network_name)
network.remove()