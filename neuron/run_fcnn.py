import docker
import socket
import json
import time
import threading
import platform
import uuid
from tqdm import tqdm

CONFIG_FILE = "config/config_cali.json"
INPUTS_FILE = "config/example_inputs/example_inputs_cali.json"
CALLBACK_PORT = 9000
BASE_PORT_FIRST_HIDDEN = 1100
BASE_PORT_INCREMENT = 800

def main():
    start_time = time.time()
    
    # Load configuration
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    layers = config["layers"]
    
    # Load Inputs
    with open(INPUTS_FILE, "r") as f:
        inputs = json.load(f)["examples"]
    input_values = inputs[0]
    num_input = len(input_values)
    
    # Pre-calculate container mapping for each layer (all layers after input)
    # Mapping: for each layer index i (i>=0), generate a list of dicts with container name and listen port.
    neuron_mappings = {}
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        num_neurons = layer["nodes"]
        # Use a different port range per layer. For layer 0, start at BASE_PORT_FIRST_HIDDEN;
        # for later layers, add BASE_PORT_INCREMENT per layer.
        base_port = BASE_PORT_FIRST_HIDDEN + (layer_index * BASE_PORT_INCREMENT)
        mapping = []
        for j in range(num_neurons):
            if layer["type"] == "output":
                container_name = f"nn_output_{j}"
            else:
                container_name = f"nn_hidden_{layer_index}_{j}"  # Remove the -1
            port = base_port + j + 1
            mapping.append({"name": container_name, "port": port})
        neuron_mappings[layer_index] = mapping
    
    # Create Docker client and network
    client = docker.from_env()
    network_name = "fcnn_network"
    try:
        network = client.networks.create(network_name, driver="bridge")
        print(f"Created new network: {network_name}")
    except Exception:
        # Remove existing network and recreate to ensure clean state
        try:
            existing_network = client.networks.get(network_name)
            existing_network.remove()
            print(f"Removed existing network: {network_name}")
            network = client.networks.create(network_name, driver="bridge")
            print(f"Created new network: {network_name}")
        except Exception as e:
            print(f"Error handling network: {e}")
            network = client.networks.get(network_name)
    
    # For Linux, get the Docker network gateway IP; for macOS use host.docker.internal.
    network_details = network.attrs
    if platform.system() == "Darwin":
        gateway_ip = "host.docker.internal"
    else:
        gateway_ip = network_details['IPAM']['Config'][0]['Gateway']
    print("Gateway IP for callback:", gateway_ip)
    
    # Remove existing containers
    print("Checking for existing containers...")
    for layer_index in range(len(layers)):  # Changed to include all layers
        for j in range(layers[layer_index]["nodes"]):
            mapping = neuron_mappings[layer_index][j]
            container_name = mapping["name"]
            try:
                container = client.containers.get(container_name)
                print(f"Removing existing container: {container_name}")
                container.stop()
                container.remove()
            except docker.errors.NotFound:
                pass
    
    containers = {}
    
    # Spawn containers for each neuron
    print("Starting neural network containers...")
    for layer_index in range(len(layers)):  # Changed to include all layers
        layer = layers[layer_index]
        neurons = layer.get("neurons", [])
        num_neurons = layer["nodes"]
        # Expected inputs equals the number of neurons (or input nodes) in the previous layer
        if layer_index == 0:  # First hidden layer
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
            if layer_index == 0:  # Changed from layer_index == 1
                ports = {f"{listen_port}/tcp": listen_port}
                
            print(f"Spawning container '{container_name}' on port {listen_port}")
    
            container = client.containers.run(
                "fcnn_image",
                detach=True,
                name=container_name,
                network=network_name,
                environment=env,
                ports=ports
            )
            containers[container_name] = container
    
    # Wait for containers to start
    print(f"Waiting for all containers to initialize...")

    expected_outputs = layers[-1]["nodes"]  # Number of neurons in the output layer
    
    # Callback server to receive final outputs from output neurons
    def callback_server():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", CALLBACK_PORT))
            s.listen()
            print(f"Callback server listening on port {CALLBACK_PORT}...")
            
            while True:
                round_results = []  # Initialize results for current inference round
                while len(round_results) < expected_outputs:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(10240).decode()
                        if data:
                            try:
                                msg = json.loads(data)
                                result_value = msg.get("value")
                                print(f"Received from {addr}: {result_value}")
                                round_results.append(result_value)
                            except Exception as e:
                                print(f"Error decoding message from {addr}: {e}")
                print("Inference round complete with results:", round_results)
                # Forward results to run_inference
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as relay_sock:
                        relay_sock.connect(("127.0.0.1", 9500))
                        relay_msg = json.dumps({"results": round_results}).encode()
                        relay_sock.sendall(relay_msg)
                except Exception as e:
                    print(f"Error forwarding results to inference script: {e}")

    callback_thread = threading.Thread(target=callback_server, daemon=True)
    callback_thread.start()
    
    elapsed_time = time.time() - start_time
    print(f"Neural network started in {elapsed_time:.3f} seconds.")
    
    # Start progress monitor for neuron connection establishment
    total_neurons = sum(layer["nodes"] for layer in layers)
    LOGFILE = 'logs/neuron_logs.txt'
    def progress_monitor():
        import time
        last_count = -1
        while True:
            try:
                with open(LOGFILE, 'r') as f:
                    lines = f.readlines()
                count = sum(1 for line in lines if "Connections established" in line)
                if count != last_count:
                    print(f"Progress: {count}/{total_neurons} neurons have established connections.")
                    last_count = count
                if count >= total_neurons:
                    break
            except Exception as e:
                print("Progress monitor error:", e)
            time.sleep(2)
        print("All neurons connections established.")
    
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    print("Network is ready for inference. Run 'python run_inference.py' to send inputs.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the neural network...")
        for container_name, container in containers.items():
            print(f"Stopping container {container_name}...")
            container.stop()
            container.remove()
        print(f"Removing network {network_name}...")
        network.remove()
        print("Neural network shutdown complete.")

if __name__ == "__main__":
    main()