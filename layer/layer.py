import os
import socket
import json
import threading
import time
import math
from datetime import datetime

class Layer:
    def __init__(self):
        self.container_name = os.environ.get("CONTAINER_NAME", "layer")
        self.listen_port = int(os.environ.get("LISTEN_PORT", "8000"))
        self.expected_input_dim = int(os.environ.get("EXPECTED_INPUT_DIM", "0"))
        self.neurons = json.loads(os.environ.get("NEURONS", "[]"))
        self.next_nodes = json.loads(os.environ.get("NEXT_NODES", "[]"))
        self.connections = {}
        self.establish_connections()
        self.log_msg(f"Layer initialized with {len(self.neurons)} neurons expecting input dimension {self.expected_input_dim}", 0)
    
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


    def connect_to_node(self, host, port, max_retries=50, retry_interval=2):
        retries = 0
        while retries < max_retries:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                self.log_msg(f"Established persistent connection to {host}:{port}", 0)
                return s
            except Exception as e:
                retries += 1
                self.log_msg(f"Attempt {retries} to connect to {host}:{port} failed: {e}", 0)
                time.sleep(retry_interval)
        self.log_msg(f"Failed to establish connection to {host}:{port} after {max_retries} attempts", 0)
        return None

    def establish_connections(self):
        for node in self.next_nodes:
            host = node["host"]
            port = int(node["port"])
            conn = self.connect_to_node(host, port)
            if conn:
                self.connections[(host, port)] = conn
        self.log_msg("Connections established", 0)

    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def activate(self, x, func):
        if func == "relu":
            return self.relu(x)
        elif func == "sigmoid":
            return self.sigmoid(x)
        else:
            return x  # linear

    def process_matrix(self, input_matrix):
        output_matrix = []
        for row in input_matrix:
            if len(row) != self.expected_input_dim:
                self.log_msg("Input dimension mismatch", 0)
                continue
            outputs = []
            for neuron in self.neurons:
                weights = neuron.get("weights", [])
                bias = neuron.get("bias", 0)
                activation = neuron.get("activation", "linear")
                dot = sum(float(val) * float(w) for val, w in zip(row, weights)) + bias
                outputs.append(self.activate(dot, activation))
            output_matrix.append(outputs)
        self.log_msg(f"Processed matrix: {output_matrix}", 0)
        return output_matrix

    def forward_result(self, output_matrix):
        msg = json.dumps({"matrix": output_matrix}).encode()
        for node in self.next_nodes:
            host = node["host"]
            port = int(node["port"])
            conn = self.connections.get((host, port))
            if conn:
                try:
                    conn.sendall(msg)
                    self.log_msg(f"Forwarded output to {host}:{port}", 0)
                except Exception as e:
                    self.log_msg(f"Error forwarding to {host}:{port}: {e}", 0)

    def handle_connection(self, conn, addr):
        self.log_msg(f"Accepted connection from {addr}", 0)
        try:
            data = conn.recv(10240).decode()
            if data:
                try:
                    content = json.loads(data)
                    matrix = content.get("matrix", [])
                    output = self.process_matrix(matrix)
                    self.forward_result(output)
                except Exception as e:
                    self.log_msg(f"Error processing data from {addr}: {e}", 0)
        except Exception as e:
            self.log_msg(f"Error handling connection from {addr}: {e}", 0)
        finally:
            conn.close()

    def start_server(self):
        self.log_msg(f"Listening on port {self.listen_port} ...", 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", self.listen_port))
            s.listen()
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    layer = Layer()
    try:
        layer.start_server()
    except KeyboardInterrupt:
        print("Shutting down layer.")
