import docker
import socket
import json
import time
import threading
import platform

CONFIG_FILE = "config/config_mnist.json"
INPUTS_FILE = "config/example_inputs/example_inputs_mnist.json"
CALLBACK_PORT = 9100
BASE_PORT_FIRST_LAYER = 5100    # Base port for the first layer
BASE_PORT_INCREMENT = 100        # Increment per layer

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
        s.bind(("", CALLBACK_PORT))
        s.listen()
        print(f"Callback server listening on port {CALLBACK_PORT}...")
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
    import os  # ensure os is imported
    import json
    containers = {}
    print("Spawning layer containers...")
    for layer_index, mapping in layer_mappings.items():
        env = {
            "CONTAINER_NAME": mapping["container_name"],
            "LISTEN_PORT": str(mapping["listen_port"]),
            "EXPECTED_INPUT_DIM": str(mapping["expected_input"]),
            "NEXT_NODES": json.dumps(mapping["next_nodes"])
        }
        # Instead of passing large NEURONS inline, check its length.
        neurons_json = json.dumps(mapping["neurons"])
        volumes = {}
        THRESHOLD = 1000
        if len(neurons_json) > THRESHOLD or True:
            # Write neuron config to a file and mount it inside the container.
            config_dir = os.path.abspath("cache/neuron_configs")
            os.makedirs(config_dir, exist_ok=True)
            neuron_file_host = os.path.join(config_dir, f"layer_{layer_index}_neurons.json")
            with open(neuron_file_host, "w") as f:
                json.dump(mapping["neurons"], f)
            neuron_file_container = f"/app/neuron_configs/layer_{layer_index}_neurons.json"
            env["NEURONS_FILE"] = neuron_file_container
            print(f"Layer {layer_index}: using NEURONS_FILE with {neuron_file_host}")
            volumes[config_dir] = {'bind': '/app/neuron_configs', 'mode': 'ro'}
        else:
            env["NEURONS"] = neurons_json

        ports = {f"{mapping['listen_port']}/tcp": mapping["listen_port"]} if layer_index == 0 else None
        if layer_index == 0:
            print(f"First layer container '{mapping['container_name']}' published on port {mapping['listen_port']}")
        else:
            print(f"Spawned container '{mapping['container_name']}' on port {mapping['listen_port']}")
        container = client.containers.run(
            "fcnn_layer_image",  # ensure this image is built with layer.py as entrypoint
            detach=True,
            name=mapping["container_name"],
            network=network_name,
            environment=env,
            ports=ports,
            volumes=volumes if volumes else None
        )
        containers[mapping["container_name"]] = container
    return containers

def main():
    start_time = time.time()
    # Load configuration and inputs
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    layers = config["layers"]

    with open(INPUTS_FILE, "r") as f:
        inputs = json.load(f)["examples"]
    # Use all examples as a matrix (each example is a row)
    input_matrix = inputs

    # Pre-calculate container mapping for each layer
    layer_mappings = {}
    for layer_index, layer in enumerate(layers):
        container_name = f"layer_{layer_index}"
        base_port = BASE_PORT_FIRST_LAYER + (layer_index * BASE_PORT_INCREMENT)
        listen_port = base_port + 1  # assign one port per layer container
        # For each layer, expected input dimension is:
        expected_input = len(input_matrix[0]) if layer_index == 0 else layers[layer_index - 1]["nodes"]
        # Determine NEXT_NODES: if not final layer, forward to next layer container;
        # if final, forward to callback server.
        if layer_index < len(layers) - 1:
            next_base_port = BASE_PORT_FIRST_LAYER + ((layer_index+1) * BASE_PORT_INCREMENT)
            next_port = next_base_port + 1
            next_nodes = [{"host": f"layer_{layer_index+1}", "port": str(next_port)}]
        else:
            if platform.system() == "Darwin":
                gateway_ip = "host.docker.internal"
            else:
                # For Linux, assume a proper gateway IP is available; using localhost as placeholder.
                gateway_ip = "127.0.0.1"
            next_nodes = [{"host": gateway_ip, "port": str(CALLBACK_PORT)}]
        layer_mappings[layer_index] = {
            "container_name": container_name,
            "listen_port": listen_port,
            "expected_input": expected_input,
            "neurons": layer.get("neurons", []),
            "next_nodes": next_nodes
        }

    client = docker.from_env()
    network_name = "fcnn_layer_network"
    network = create_network(client, network_name)
    remove_existing_containers(client, layer_mappings)
    containers = spawn_layer_containers(client, layer_mappings, network_name)

    print("Waiting for containers to initialize...")

    callback_thread = threading.Thread(target=callback_server, daemon=True)
    callback_thread.start()

    # Start progress monitor for neuron connection establishment
    total_layers = len(layers)
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
                    print(f"Progress: {count}/{total_layers} layers have established connections.")
                    last_count = count
                if count >= total_layers:
                    break
            except Exception as e:
                print("Progress monitor error:", e)
            time.sleep(2)
        print("All layers connections established.")
    
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    

    # Send input matrix to first layer container
    # first_mapping = layer_mappings[0]
    # try:
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.connect(("127.0.0.1", first_mapping["listen_port"]))
    #         msg = json.dumps({"matrix": input_matrix}).encode()
    #         s.sendall(msg)
    #         print(f"Sent input matrix to layer '{first_mapping['container_name']}' on port {first_mapping['listen_port']}")
    # except Exception as e:
    #     print(f"Error sending input to first layer: {e}")

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
