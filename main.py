import docker

NUM_CONTAINERS = 3  # For example, create a chain of 3 containers
FIRST_CONTAINER_HOST_PORT = 8001  # We'll map the first container's internal port 8001 to host's 8001
CALLBACK_PORT = 9000  # Port on which the controller listens for the final result

# Create a Docker client
client = docker.from_env()

# Create a dedicated Docker network so containers can resolve each other by name.
network_name = "chain_network"
try:
    network = client.networks.create(network_name, driver="bridge")
    # After creating the network, inspect it to get the gateway IP
    network_details = network.attrs
    gateway_ip = network_details['IPAM']['Config'][0]['Gateway']
    print("Using gateway IP for final container:", gateway_ip)
except Exception as e:
    network = client.networks.get(network_name)

containers = []
for i in range(1, NUM_CONTAINERS + 1):
    container_name = f"chain_container_{i}"
    listen_port = 8000 + i  # Each container listens on its own port (e.g., 8001, 8002, …)
    
    # Determine next destination:
    if i < NUM_CONTAINERS:
        # Next container is the one with name chain_container_{i+1} on port 8000+(i+1)
        next_host = f"chain_container_{i+1}"
        next_port = str(8000 + i + 1)
    else:
        # For the final container, set the next host to the host machine
        # Docker Desktop on Mac/Windows supports "host.docker.internal".
        # On Linux, you might pass the host IP explicitly.
        
        # next_host = "host.docker.internal"
        # For the final container on Linux, use the gateway IP from the Docker network
        next_host = gateway_ip
        next_port = str(CALLBACK_PORT)
    
    env = {
        "LISTEN_PORT": str(listen_port),
        "NEXT_HOST": next_host,
        "NEXT_PORT": next_port,
        "MOD_STR": f" -> C{i}"
    }
    
    # Only publish the first container's port to the host
    ports = {}
    if i == 1:
        ports = {f"{listen_port}/tcp": FIRST_CONTAINER_HOST_PORT}
    
    print(f"Starting container {container_name} on internal port {listen_port} with NEXT_HOST={next_host}, NEXT_PORT={next_port}")
    container = client.containers.run(
        "chain_image",  # the image built from the Dockerfile above
        detach=True,
        name=container_name,
        network=network_name,
        environment=env,
        ports=ports  # only container 1 is accessible from the host
    )
    containers.append(container)

# Start a callback server to receive the final output
final_result = None
