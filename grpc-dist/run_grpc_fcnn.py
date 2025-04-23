import docker
import socket
import json
import time
import platform
import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration ---
# Use relative paths from the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(SCRIPT_DIR, "../config/mnist_model(10,10).json")
DEFAULT_INPUTS_FILE = os.path.join(SCRIPT_DIR, "../config/example_inputs/mnist_examples_5.json")

# Network and Container Config
DOCKER_NETWORK_NAME = "fcnn_layer_network"
DOCKER_IMAGE_NAME = "grpc_layer_image"
CONTAINER_NAME_PREFIX = "layer_container_"
BASE_PORT_FIRST_LAYER = 5100
BASE_PORT_INCREMENT = 100
DEFAULT_LAYER_DISTRIBUTION = [1, 1] # Default if not in config

# Readiness Check
READINESS_CHECK_TIMEOUT = 10 # seconds
READINESS_CHECK_INTERVAL = 0.5 # seconds

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Docker Helper Functions ---

def get_docker_client() -> docker.DockerClient:
    """Initializes and returns a Docker client."""
    try:
        client = docker.from_env()
        client.ping() # Check connection
        logging.info("Docker client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Docker client: {e}")
        raise

def manage_docker_network(client: docker.DockerClient, network_name: str) -> docker.models.networks.Network:
    """Creates or recreates a Docker bridge network."""
    try:
        network = client.networks.get(network_name)
        logging.info(f"Network '{network_name}' already exists. Removing...")
        network.remove()
    except docker.errors.NotFound:
        logging.info(f"Network '{network_name}' not found.")
    except Exception as e:
        logging.warning(f"Error removing existing network '{network_name}': {e}")

    try:
        network = client.networks.create(network_name, driver="bridge")
        logging.info(f"Created Docker network: {network_name} (ID: {network.id})")
        return network
    except Exception as e:
        logging.error(f"Failed to create Docker network '{network_name}': {e}")
        raise

def remove_existing_containers(client: docker.DockerClient, container_prefix: str):
    """Stops and removes containers matching the prefix."""
    logging.info(f"Checking for existing containers with prefix '{container_prefix}'...")
    containers_to_remove = client.containers.list(all=True, filters={"name": f"{container_prefix}*"})
    if not containers_to_remove:
        logging.info("No existing containers found to remove.")
        return

    for container in containers_to_remove:
        try:
            logging.info(f"Stopping container: {container.name} (ID: {container.id})")
            container.stop()
            logging.info(f"Removing container: {container.name} (ID: {container.id})")
            container.remove()
        except docker.errors.NotFound:
             logging.warning(f"Container {container.name} already removed.")
        except Exception as e:
            logging.error(f"Failed to stop/remove container {container.name}: {e}")

def spawn_layer_containers(
    client: docker.DockerClient,
    layer_mappings: Dict[int, Dict[str, Any]],
    network_name: str,
    image_name: str
) -> Dict[str, docker.models.containers.Container]:
    """Spawns Docker containers based on layer mappings."""
    containers = {}
    NEURON_CONFIG_SIZE_THRESHOLD = 1000 # Threshold to write config to file vs env var
    NEURON_CONFIG_CACHE_DIR = os.path.join(SCRIPT_DIR, "cache/neuron_configs")
    os.makedirs(NEURON_CONFIG_CACHE_DIR, exist_ok=True)

    logging.info(f"Spawning {len(layer_mappings)} layer containers using image '{image_name}'...")

    for container_index, mapping in layer_mappings.items():
        container_name = mapping["container_name"]
        listen_port = mapping["listen_port"]

        env = {
            "CONTAINER_NAME": container_name,
            "LISTEN_PORT": str(listen_port),
            "EXPECTED_INPUT_DIM": str(mapping["expected_input"]),
            "NEXT_NODES": json.dumps(mapping["next_nodes"])
        }

        neurons_config_str = json.dumps(mapping["neurons_config"])
        volumes = None

        # Decide whether to pass config via env var or file volume
        if len(neurons_config_str) > NEURON_CONFIG_SIZE_THRESHOLD:
            config_file_host = os.path.join(NEURON_CONFIG_CACHE_DIR, f"{container_name}_neurons_config.json")
            try:
                with open(config_file_host, "w") as f:
                    f.write(neurons_config_str)
                container_config_path = f"/app/neuron_configs/{container_name}_neurons_config.json"
                env["NEURONS_FILE_CONFIG"] = container_config_path
                volumes = {NEURON_CONFIG_CACHE_DIR: {'bind': '/app/neuron_configs', 'mode': 'ro'}}
                logging.info(f"Using file volume for neuron config for {container_name}.")
            except IOError as e:
                 logging.error(f"Failed to write neuron config file for {container_name}: {e}")
                 # Decide how to handle this - skip container? exit?
                 continue # Skip this container for now
        else:
            env["NEURONS_CONFIG"] = neurons_config_str
            logging.info(f"Using environment variable for neuron config for {container_name}.")

        # Port mapping: Only publish the first container's port to the host
        ports = {f"{listen_port}/tcp": listen_port} if container_index == 0 else None

        try:
            container_kwargs = dict(
                image=image_name,
                detach=True,
                name=container_name,
                network=network_name,
                environment=env,
                ports=ports,
                volumes=volumes # Will be None if not using file config
            )
            container = client.containers.run(**container_kwargs)
            containers[container_name] = container
            if container_index == 0:
                logging.info(f"Spawned first layer container '{container_name}' (ID: {container.id}), published port {listen_port}.")
            else:
                logging.info(f"Spawned container '{container_name}' (ID: {container.id}) on internal port {listen_port}.")

        except docker.errors.APIError as e:
            logging.error(f"Failed to start container {container_name}: {e}")
            # Consider cleanup or retry logic here
        except Exception as e:
             logging.error(f"An unexpected error occurred starting container {container_name}: {e}")

    return containers

def wait_for_first_layer(port: int, timeout: float, interval: float):
    """Waits for the first layer's gRPC server to become available on the host."""
    logging.info(f"Waiting up to {timeout}s for first layer to listen on 127.0.0.1:{port}...")
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=interval):
                logging.info(f"Layer 0 is now listening on port {port}.")
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(interval)
        except Exception as e:
            logging.warning(f"Error checking port {port}: {e}")
            time.sleep(interval)
    logging.warning(f"Timeout: First layer did not start listening on port {port} within {timeout}s.")
    return False

# --- Layer Mapping Logic ---

def calculate_layer_mappings(
    layers_config: List[Dict[str, Any]],
    distribution: List[int],
    input_matrix_example: List[Dict[str, Any]] # Used to get initial input dim
) -> Dict[int, Dict[str, Any]]:
    """Calculates container configurations based on layer distribution."""
    if sum(distribution) != len(layers_config):
        raise ValueError("Sum of layer_distribution does not match the number of layers in config.")

    layer_mappings = {}
    global_layer_index = 0
    num_containers = len(distribution)

    # Determine initial input dimension from example data
    initial_input_dim = 0
    if input_matrix_example and isinstance(input_matrix_example[0].get("input"), list):
         initial_input_dim = len(input_matrix_example[0]["input"])
    else:
         logging.warning("Could not determine initial input dimension from example inputs.")
         # Consider raising an error or using a default/config value

    current_expected_input_dim = initial_input_dim

    for container_index in range(num_containers):
        num_layers_in_container = distribution[container_index]
        container_neurons_config = {}
        output_dim_of_last_layer_in_container = current_expected_input_dim

        if num_layers_in_container == 0:
             logging.warning(f"Container {container_index} has 0 layers assigned, skipping.")
             continue

        for i in range(num_layers_in_container):
            if global_layer_index >= len(layers_config):
                 raise IndexError("Layer distribution requests more layers than available in config.")

            layer_config = layers_config[global_layer_index]
            layer_key = f"layer_{i+1}" # Key within the container's config
            container_neurons_config[layer_key] = layer_config.get("neurons", [])

            # Update the expected output dimension for the *next* layer/container
            output_dim_of_last_layer_in_container = layer_config.get("nodes", 0) # Get output nodes of this layer
            global_layer_index += 1

        container_name = f"{CONTAINER_NAME_PREFIX}{container_index}"
        listen_port = BASE_PORT_FIRST_LAYER + (container_index * BASE_PORT_INCREMENT) + 1

        # Determine next nodes
        if container_index < num_containers - 1:
            next_container_index = container_index + 1
            # Find the next container that actually has layers
            while next_container_index < num_containers and distribution[next_container_index] == 0:
                 next_container_index += 1
            
            if next_container_index < num_containers:
                 next_container_name = f"{CONTAINER_NAME_PREFIX}{next_container_index}"
                 next_listen_port = BASE_PORT_FIRST_LAYER + (next_container_index * BASE_PORT_INCREMENT) + 1
                 next_nodes = [{"host": next_container_name, "port": str(next_listen_port)}]
            else: # This container is the last one with layers
                 next_nodes = []
        else: # This is the last container index
            next_nodes = []

        layer_mappings[container_index] = {
            "container_name": container_name,
            "listen_port": listen_port,
            "expected_input": current_expected_input_dim, # Input dim for the *first* layer in this container
            "neurons_config": container_neurons_config,
            "next_nodes": next_nodes
        }
        
        # The input dim for the next container is the output dim of the last layer in *this* container
        current_expected_input_dim = output_dim_of_last_layer_in_container


    logging.info(f"Calculated mappings for {len(layer_mappings)} containers.")
    return layer_mappings


# --- Main Execution ---

def main(config_file: str, inputs_file: str):
    """Main function to set up and run the distributed FCNN."""
    start_time = time.time()

    # Load configuration and inputs
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        layers_config = config["layers"]
        distribution = config.get("layer_distribution", DEFAULT_LAYER_DISTRIBUTION)
        logging.info(f"Loaded model config from: {config_file}")
        logging.info(f"Layer distribution: {distribution}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        return
    except (KeyError, json.JSONDecodeError) as e:
        logging.error(f"Error loading or parsing config file {config_file}: {e}")
        return

    try:
        with open(inputs_file, "r") as f:
            # Load only the first example to determine input dimension
            input_examples = json.load(f).get("examples", [])
            logging.info(f"Loaded example inputs from: {inputs_file}")
    except FileNotFoundError:
        logging.error(f"Inputs file not found: {inputs_file}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing inputs file {inputs_file}: {e}")
        return

    # Calculate container mappings
    try:
        layer_mappings = calculate_layer_mappings(layers_config, distribution, input_examples)
    except (ValueError, IndexError) as e:
         logging.error(f"Error calculating layer mappings: {e}")
         return

    # Initialize Docker client
    try:
        client = get_docker_client()
    except Exception:
        return # Error already logged in get_docker_client

    network = None
    spawned_containers = {}
    try:
        # Setup Docker network
        network = manage_docker_network(client, DOCKER_NETWORK_NAME)

        # Clean up old containers
        remove_existing_containers(client, CONTAINER_NAME_PREFIX)

        # Spawn new containers
        spawned_containers = spawn_layer_containers(client, layer_mappings, DOCKER_NETWORK_NAME, DOCKER_IMAGE_NAME)

        if not spawned_containers:
             logging.error("No containers were spawned. Exiting.")
             return

        # Wait for the first layer to be ready
        first_layer_port = layer_mappings[0]["listen_port"]
        wait_for_first_layer(first_layer_port, READINESS_CHECK_TIMEOUT, READINESS_CHECK_INTERVAL)

        elapsed_time = time.time() - start_time
        logging.info(f"Distributed FCNN setup completed in {elapsed_time:.3f} seconds.")
        logging.info("Containers are running. Press Ctrl+C to shut down.")

        # Keep the script running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Shutdown signal received (Ctrl+C).")
    except Exception as e:
         logging.error(f"An unexpected error occurred during setup/runtime: {e}")
    finally:
        # Cleanup Docker resources
        logging.info("Shutting down and cleaning up Docker resources...")
        if spawned_containers:
             remove_existing_containers(client, CONTAINER_NAME_PREFIX) # Use the same function for cleanup
        if network:
            try:
                logging.info(f"Removing network: {network.name}")
                network.remove()
            except Exception as e:
                logging.error(f"Failed to remove network {network.name}: {e}")
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a distributed FCNN using Docker containers and gRPC.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help=f"Path to the model configuration JSON file (default: {DEFAULT_CONFIG_FILE})"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=DEFAULT_INPUTS_FILE,
        help=f"Path to the example inputs JSON file (default: {DEFAULT_INPUTS_FILE})"
    )
    args = parser.parse_args()

    main(args.config, args.inputs)
