import docker
import socket
import json
import time
import threading
import platform

CONFIG_FILE = "config/mnist_model(10,10).json"
INPUTS_FILE = "config/example_inputs/mnist_examples_labels.json"
CALLBACK_PORT = 9100
BASE_PORT_FIRST_LAYER = 5100    # Base port for the first layer
BASE_PORT_INCREMENT = 100        # Increment per layer
LAYER_DISTRIBUTION = [5,5]  # Use this to define the distribution of layers across containers

def forward_results(msg):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as relay_sock:
            relay_sock.connect(("127.0.0.1", 9500))
            relay_msg = json.dumps({"results": msg}).encode()
            relay_sock.sendall(relay_msg)
    except Exception as e:
        print(f"Error forwarding results to inference script: {e}")

def callback_server():
    """Callback server to receive final output matrix."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", CALLBACK_PORT))
        s.listen()
        print(f"Callback server listening on port {CALLBACK_PORT}...")
        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    data = conn.recv(10240).decode()
                    msg = json.loads(data)
                    print("Received final output:", msg.get("matrix"))
                    forward_results(msg)
                except Exception as e:
                    print("Error in callback:", e)

def create_network(client, network_name):
    try:
        network = client.networks.create(network_name, driver="bridge")
        print(f"Created network: {network_name}")
        return network
    except Exception:
        try:
            net = client.networks.get(network_name)
            net.remove()
            network = client.networks.create(network_name, driver="bridge")
            print(f"Recreated network: {network_name}")
            return network
        except Exception as e:
            print(f"Error handling network: {e}")
            return client.networks.get(network_name)

def remove_existing_containers(client, layer_mappings):
    print("Checking for existing layer containers...")
    for mapping in layer_mappings.values():
        try:
            container = client.containers.get(mapping["container_name"])
            container.stop()
            container.remove()
            print(f"Removed container: {mapping['container_name']}")
        except Exception:
            pass

def spawn_layer_containers(client, layer_mappings, network_name):
    import os
    import json
    containers = {}
    THRESHOLD = 1000
    print("Spawning layer containers...")
    for container_index, mapping in layer_mappings.items():
        env = {
            "CONTAINER_NAME": mapping["container_name"],
            "LISTEN_PORT": str(mapping["listen_port"]),
            "EXPECTED_INPUT_DIM": str(mapping["expected_input"]),
            "NEXT_NODES": json.dumps(mapping["next_nodes"])
        }
        # Set NUM_INPUT_NODES: for output container, use num_containers, else "1"
        if mapping["container_name"] == "output_layer_container":
            env["NUM_INPUT_NODES"] = str(len(LAYER_DISTRIBUTION))
        else:
            env["NUM_INPUT_NODES"] = "1"
        # Prepare NEURONS_CONFIG string from neurons_config
        neurons_config_str = json.dumps(mapping["neurons_config"])
        volumes = None
        if len(neurons_config_str) > THRESHOLD:
            config_dir = os.path.abspath("cache/neuron_configs")
            os.makedirs(config_dir, exist_ok=True)
            config_file_host = os.path.join(config_dir, f"container_{container_index}_neurons_config.json")
            with open(config_file_host, "w") as f:
                f.write(neurons_config_str)
            container_config_path = f"/app/neuron_configs/container_{container_index}_neurons_config.json"
            env["NEURONS_FILE_CONFIG"] = container_config_path
            # Remove direct config
            # volumes mapping for container to read the file.
            volumes = {config_dir: {'bind': '/app/neuron_configs', 'mode': 'ro'}}
        else:
            env["NEURONS_CONFIG"] = neurons_config_str
        
        ports = {f"{mapping['listen_port']}/tcp": mapping["listen_port"]} if container_index == 0 else None
        if container_index == 0:
            print(f"First layer container '{mapping['container_name']}' published on port {mapping['listen_port']}")
        else:
            print(f"Spawned container '{mapping['container_name']}' on port {mapping['listen_port']}")
        # container = client.containers.run(
        #     "vertical_fcnn_image",  # ensure the image is built with layer.py as entrypoint
        #     detach=True,
        #     name=mapping["container_name"],
        #     network=network_name,
        #     environment=env,
        #     ports=ports,
        #     volumes=volumes
        # )
        container_kwargs = dict(
            image="horizontal_fcnn_image",
            detach=True,
            name=mapping["container_name"],
            network=network_name,
            environment=env,
            ports=ports
        )
        if volumes:
            container_kwargs["volumes"] = volumes
        container = client.containers.run(**container_kwargs)

        containers[mapping["container_name"]] = container
    return containers

def main():
    start_time = time.time()
    # Load configuration and inputs
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    layers = config["layers"]  # list of layer configurations
    distribution = LAYER_DISTRIBUTION
    if sum(distribution) != layers[0]["nodes"]:
        print("Error: Sum of layer_distribution does not match number of nodes in the first hidden layer.")
        return

    with open(INPUTS_FILE, "r") as f:
        input_matrix = json.load(f)["examples"]
    input_size = len(input_matrix[0]["input"])

    # Distribute neurons across all hidden layers into containers by node index fractions
    layer_mappings = {}
    hidden_layers = layers[:-1]  # Exclude output layer
    output_layer = layers[-1]
    num_hidden_layers = len(hidden_layers)
    num_containers = len(distribution)
    total_nodes = hidden_layers[0]["nodes"]  # Number of neurons in first hidden layer

    # Compute start/end indices for each container
    indices = []
    start = 0
    for count in distribution:
        end = start + int(round(count * total_nodes / sum(distribution)))
        indices.append((start, end))
        start = end
    # Adjust last container to include any rounding errors
    indices[-1] = (indices[-1][0], total_nodes)

    # Prepare hidden layer containers
    for container_index, (start_idx, end_idx) in enumerate(indices):
        neurons_config = {}
        for layer_idx, layer in enumerate(hidden_layers):
            # Each layer's neurons are split by the same indices
            neurons = layer.get("neurons", [])[start_idx:end_idx]
            neurons_config[f"layer_{layer_idx+1}"] = neurons
        container_name = f"layer_container_{container_index}"

        base_port = BASE_PORT_FIRST_LAYER + (container_index * BASE_PORT_INCREMENT)
        listen_port = base_port + 1
        expected_input = input_size if container_index == 0 else end_idx - start_idx

        # Next nodes: all other containers (except self)
        next_nodes = []
        for other_idx, (o_start, o_end) in enumerate(indices):
            if other_idx != container_index:
                other_base_port = BASE_PORT_FIRST_LAYER + (other_idx * BASE_PORT_INCREMENT)
                other_port = other_base_port + 1
                next_nodes.append({"host": f"layer_container_{other_idx}", "port": str(other_port)})

        # If this container contains any neurons from the last hidden layer, add output layer container as next node
        if end_idx > 0 and (num_hidden_layers - 1) in range(num_hidden_layers):
            # Check if this container has any neurons from the last hidden layer
            if len(hidden_layers[-1].get("neurons", [])[start_idx:end_idx]) > 0:
                output_base_port = BASE_PORT_FIRST_LAYER + (num_containers * BASE_PORT_INCREMENT)
                output_port = output_base_port + 1
                next_nodes.append({"host": "output_layer_container", "port": str(output_port)})

        layer_mappings[container_index] = {
            "container_name": container_name,
            "listen_port": listen_port,
            "expected_input": expected_input,
            "neurons_config": neurons_config,
            "next_nodes": next_nodes
        }

    # Add output layer container
    output_base_port = BASE_PORT_FIRST_LAYER + (num_containers * BASE_PORT_INCREMENT)
    output_port = output_base_port + 1
    output_neurons_config = {"layer_output": output_layer.get("neurons", [])}
    gateway_ip = "host.docker.internal" if platform.system() == "Darwin" else "127.0.0.1"
    output_next_nodes = [{"host": gateway_ip, "port": str(CALLBACK_PORT)}]
    layer_mappings[num_containers] = {
        "container_name": "output_layer_container",
        "listen_port": output_port,
        "expected_input": len(output_layer.get("neurons", [])),
        "neurons_config": output_neurons_config,
        "next_nodes": output_next_nodes
    }

    client = docker.from_env()
    network_name = "fcnn_layer_network"
    network = create_network(client, network_name)
    remove_existing_containers(client, layer_mappings)
    containers = spawn_layer_containers(client, layer_mappings, network_name)

    print("Waiting for containers to initialize...")

    callback_thread = threading.Thread(target=callback_server, daemon=True)
    callback_thread.start()

    # # Start progress monitor for neuron connection establishment
    # total_layers = len(layers)
    # LOGFILE = 'logs/neuron_logs.txt'
    # def progress_monitor():
    #     import time
    #     last_count = -1
    #     while True:
    #         try:
    #             with open(LOGFILE, 'r') as f:
    #                 lines = f.readlines()
    #             count = sum(1 for line in lines if "Connections established" in line)
    #             if count != last_count:
    #                 print(f"Progress: {count}/{total_layers} layers have established connections.")
    #                 last_count = count
    #             if count >= total_layers:
    #                 break
    #         except Exception as e:
    #             print("Progress monitor error:", e)
    #         time.sleep(2)
    #     print("All layers connections established.")
    
    # monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    # monitor_thread.start()
    
    elapsed_time = time.time() - start_time
    print(f"FCNN with layers started in {elapsed_time:.3f} seconds. Waiting for final output...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down layer-based FCNN...")
        for cont in containers.values():
            cont.stop()
            cont.remove()
        network.remove()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
