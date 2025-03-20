import docker
import socket
import json
import time
import threading

num_containers = 5
container_port = 8001  # Port no. of the first container
callback_port = 9000  # Port no. of the final output

# Create a Docker client
client = docker.from_env()

network_name = "chain_network"
try:
    # Create a Docker network
    network = client.networks.create(network_name, driver="bridge")
    network_details = network.attrs
    gateway_ip = network_details['IPAM']['Config'][0]['Gateway'] # Get the gateway IP of the Docker network
    print("Using gateway IP for final container:", gateway_ip)
except Exception as e:
    print("Network already exists, using existing network...")
    network = client.networks.get(network_name)

containers = []
for i in range(1, num_containers + 1):
    container_name = f"chain_container_{i}"
    listen_port = 8000 + i  # Each container listens on its own port (e.g., 8001, 8002, …)
    
    # Determine next destination:
    if i < num_containers:
        # Next container is the one with name chain_container_{i+1} on port 8000+(i+1)
        next_host = f"chain_container_{i+1}"
        next_port = str(8000 + i + 1)
    else:
        # For the final container, set the next host to the host machine
        next_host = gateway_ip  # for Mac/Windows: next_host = "host.docker.internal".
        next_port = str(callback_port)
    
    # Set environment variabless to pass to the container
    env = {
        "LISTEN_PORT": str(listen_port),
        "NEXT_HOST": next_host,
        "NEXT_PORT": next_port,
        "PAYLOAD_STR": f" >> Container {i}"
    }
    
    ports = {}
    if i == 1: # Only the first container is accessible from the host
        '''
        If i == 1, it binds listen_port from the host to container_port inside the container.
        Otherwise, the container is only accessible within the Docker network.
        '''
        ports = {f"{listen_port}/tcp": container_port}
    
    print(f"Starting container {container_name} on port {listen_port} with NEXT_HOST={next_host}, NEXT_PORT={next_port}")
    container = client.containers.run(
        "chain_image",  # the image built from the Dockerfile above
        detach=True,
        name=container_name,
        network=network_name,
        environment=env,
        ports=ports  # only container 1 is accessible from the host script
    )
    containers.append(container)

# Start a callback server to receive the final output
final_result = None
def callback_server():
    global final_result
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", callback_port))
        s.listen()
        print(f"Callback server listening on port {callback_port}...")
        conn, addr = s.accept()
        with conn:
            data = conn.recv(4096).decode()
            if data:
                final_result = json.loads(data)
                print("Final result received:", final_result)

# # Run the callback server in a separate thread
callback_thread = threading.Thread(target=callback_server, daemon=True)
callback_thread.start()

# Give the containers a moment to start up
time.sleep(num_containers)

# Prepare the input payload in JSON 
input_data = {"message": "Hello", "counter": 0}
input_json = json.dumps(input_data).encode()

start_time = time.time() # Start timer

# Send the input to the first container
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("0.0.0.0", container_port))
    s.sendall(input_json)
print("Input sent to first container.")

# Wait for the callback to receive the final result
timeout = 10  # timeout to receive the result (10 seconds)
callback_thread.join(timeout) # Wait for the callback thread to finish
elapsed_time = time.time() - start_time # Calculate elapsed time

if final_result is not None:
    print("Final output:", final_result)
    print("Total time taken: {:.3f} seconds".format(elapsed_time))
else:
    print("No response received within the timeout period.")

# Clean up the containers and network when done.
for container in containers:
    print(f"Stopping and removing container {container.name}")
    container.stop()
    container.remove()
print("Removing network", network_name)
network.remove()
print("All containers and network removed.")