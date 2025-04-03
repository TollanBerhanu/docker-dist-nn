import os
import socket
import json
import threading
import math
import time
from datetime import datetime

class Neuron:
    def __init__(self):
        # Configuration from environment variables
        self.container_name = os.environ.get("CONTAINER_NAME", "container")
        self.listen_port = int(os.environ.get("LISTEN_PORT", "8000"))
        self.expected_inputs = int(os.environ.get("EXPECTED_INPUTS", "1"))
        self.weights = json.loads(os.environ.get("WEIGHTS", "[]"))
        self.bias = float(os.environ.get("BIAS", "0"))
        self.activation = os.environ.get("ACTIVATION", "relu")
        self.next_nodes = json.loads(os.environ.get("NEXT_NODES", "[]"))
        
        # State variables
        self.inputs_lock = threading.Lock()
        self.collected_inputs = []
        self.connections = {}  # Store persistent connections to next nodes
        
        self.log_msg(f"""Neuron initialized with configuration - 
            LISTEN_PORT: {self.listen_port},
            EXPECTED_INPUT_LENGTH: {self.expected_inputs},
            WEIGHTS: {self.weights[:2] + ["..."] + self.weights[-2:] if len(self.weights) > 4 else self.weights}, 
            BIAS: {self.bias}, 
            ACTIVATION: {self.activation}, 
            NEXT_NODES: {self.next_nodes}
            """, 0)
        
        # Establish connections to downstream neurons
        self.establish_connections()

    def establish_connections(self):
        """Establish persistent connections to all downstream neurons"""
        max_retries = 50
        retry_interval = 2  # seconds
        
        for node in self.next_nodes:
            host = node["host"]
            port = int(node["port"])
            connected = False
            retries = 0
            
            while not connected and retries < max_retries:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) # Enable keep-alive
                    s.connect((host, port))
                    self.connections[(host, port)] = s
                    self.log_msg(f"Established persistent connection to {host}:{port}", 0)
                    connected = True
                except Exception as e:
                    retries += 1
                    self.log_msg(f"Connection attempt {retries} to {host}:{port} failed: {e}. Retrying in {retry_interval}s...", 0)
                    time.sleep(retry_interval)
            
            if not connected:
                self.log_msg(f"Failed to establish connection to {host}:{port} after {max_retries} attempts", 0)
        self.log_msg(f"Connections established", 0)

    def log_msg(self, message, priority=0):
        if priority > 0:
            return
        timestamped = f"{datetime.now().isoformat()} | ({self.container_name}) {message}"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as log_sock:
                # Using host.docker.internal to reach the host machine
                log_sock.connect(("host.docker.internal", 2345))
                log_sock.sendall(timestamped.encode())
        except Exception as e:
            # If logging fails, fallback to writing locally
            print(f"Logging failed: {e}")
        # Also log locally for container debugging
        print(timestamped)

    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def activate(self, x):
        self.log_msg(f"Activating with {self.activation} function on value: {x}", 1)
        if self.activation == "relu":
            return self.relu(x)
        elif self.activation == "sigmoid":
            return self.sigmoid(x)
        else:
            # Default: linear activation
            return x

    def process_and_forward(self):
        self.log_msg(f"\t---> Processing {len(self.collected_inputs)} inputs", 0)
        
        # Compute weighted sum
        weighted_sum = sum(float(val) * float(w) for val, w in zip(self.collected_inputs, self.weights)) + self.bias
        self.log_msg(f"\t---> Computed weighted sum: {weighted_sum}", 1)
        
        output = self.activate(weighted_sum)
        self.log_msg(f"\t---> Activation output: {output}", 1)
        
        # Reset inputs for the next inference round
        self.collected_inputs = []
        
        # Prepare output value
        output_values = json.dumps({"value": output}).encode()
        
        # Forward the result to all next nodes using persistent connections
        for node in self.next_nodes:
            host = node["host"]
            port = int(node["port"])
            try:
                conn = self.connections.get((host, port))
                if conn:
                    conn.sendall(output_values)
                    self.log_msg(f"\t---> Forwarded output to {host}:{port} with value: {output}", 0)
                else:
                    # If connection doesn't exist or was closed, try to establish it
                    self.log_msg(f"\t---> Connection to {host}:{port} not found, attempting to reconnect", 0)
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect((host, port))
                        self.connections[(host, port)] = s
                        s.sendall(output_values)
                        self.log_msg(f"\t---> Reconnected and forwarded output to {host}:{port}", 0)
                    except Exception as e:
                        self.log_msg(f"\t---> Error reconnecting to {host}:{port}: {e}", 0)
            except Exception as e:
                self.log_msg(f"\t---> Error forwarding to {host}:{port}: {e}", 0)

    def handle_neuron(self, conn, addr):
        self.log_msg(f"{self.container_name} handling client on {addr}", 1)
        try:
            while True:
                data = conn.recv(1024).decode()
                if not data:
                    break
                try:
                    msg = json.loads(data)
                    value = msg.get("value")

                    # For the input layer, we expect a list of inputs
                    with self.inputs_lock:
                        if isinstance(value, list):
                            self.log_msg(f"\t---> Received input: [{value[0]}, ... {value[-1]}]", 0)
                            self.collected_inputs = value
                        else: # For hidden layers, we expect a single input value
                            self.log_msg(f"\t---> Received input: {value}", 0)
                            self.collected_inputs.append(value)
                    if len(self.collected_inputs) == self.expected_inputs:
                        try:
                            self.process_and_forward()
                        except Exception as e:
                            self.log_msg(f"\t---> Error processing and forwarding data: {e}", 0)
                except Exception as e:
                    self.log_msg(f"\t---> Error loading data: {e}", 0)
                        
        except Exception as e:
            self.log_msg(f"\t---> Error handleing neuron: {e}", 0)
        finally:
            conn.close()

    def start_server(self):
        self.log_msg(f"Listening on port {self.listen_port} ...", 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", self.listen_port))
            s.listen()
            while True:
                conn, addr = s.accept()
                self.log_msg(f"\t---> Accepted connection from {addr}", 0)
                threading.Thread(target=self.handle_neuron, args=(conn, addr), daemon=True).start()

    def cleanup(self):
        """Close all persistent connections"""
        for (host, port), conn in self.connections.items():
            try:
                conn.close()
                self.log_msg(f"Closed connection to {host}:{port}", 0)
            except Exception as e:
                self.log_msg(f"Error closing connection to {host}:{port}: {e}", 0)

if __name__ == "__main__":
    print("Starting Neuron...")
    neuron = Neuron()
    try:
        neuron.start_server()
    except KeyboardInterrupt:
        neuron.cleanup()