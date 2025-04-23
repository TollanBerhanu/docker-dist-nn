import docker
import socket
import json
import time
import threading
import platform

CONFIG_FILE = "config/mnist_model(10,10).json"
INPUTS_FILE = "config/example_inputs/mnist_examples_5.json"
CALLBACK_PORT = 9100
BASE_PORT_FIRST_LAYER = 5100    # Base port for the first layer
BASE_PORT_INCREMENT = 100        # Increment per layer
LAYER_DISTRIBUTION = [1,1]  # Use this to define the distribution of layers across containers

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
            image="vertical_fcnn_image",
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
    distribution = config.get("layer_distribution", LAYER_DISTRIBUTION)
    if sum(distribution) != len(layers):
        print("Error: Sum of layer_distribution does not match number of layers in config.")
        return

    with open(INPUTS_FILE, "r") as f:
        inputs = json.load(f)["examples"]
    input_matrix = inputs

    # Compute container mappings based on distribution
    layer_mappings = {}
    global_layer_index = 0
    num_containers = len(distribution)
    for container_index in range(num_containers):
        group_count = distribution[container_index]
        neurons_config = {}
        for i in range(group_count):
            key = f"layer_{i+1}"
            neurons_config[key] = layers[global_layer_index].get("neurons", [])
            global_layer_index += 1
        container_name = f"layer_container_{container_index}"
        base_port = BASE_PORT_FIRST_LAYER + (container_index * BASE_PORT_INCREMENT)
        listen_port = base_port + 1
        expected_input = len(input_matrix[0]["input"]) if container_index == 0 else layers[global_layer_index - group_count - 1]["nodes"]
        if container_index < num_containers - 1:
            next_base_port = BASE_PORT_FIRST_LAYER + ((container_index+1) * BASE_PORT_INCREMENT)
            next_port = next_base_port + 1
            next_nodes = [{"host": f"layer_container_{container_index+1}", "port": str(next_port)}]
        else:
            gateway_ip = "host.docker.internal" if platform.system() == "Darwin" else "127.0.0.1"
            next_nodes = [{"host": gateway_ip, "port": str(CALLBACK_PORT)}]
        layer_mappings[container_index] = {
            "container_name": container_name,
            "listen_port": listen_port,
            "expected_input": expected_input,
            "neurons_config": neurons_config,
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
