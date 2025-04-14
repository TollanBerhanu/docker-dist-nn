import os
import socket
import json
import threading
import time
import math
from datetime import datetime

def log_to_host(message, container_name):
    timestamped = f"{datetime.now().isoformat()} | ({container_name}) {message}"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as log_sock:
            log_sock.connect(("host.docker.internal", 2345))
            log_sock.sendall(timestamped.encode())
    except Exception as e:
        print(f"Logging failed: {e}")
    print(timestamped)

class Layer:
    def __init__(self):
        self.container_name = os.environ.get("CONTAINER_NAME", "layer")
        self.listen_port = int(os.environ.get("LISTEN_PORT", "8000"))
        self.expected_input_dim = int(os.environ.get("EXPECTED_INPUT_DIM", "0"))
        if "NEURONS_CONFIG" in os.environ:
            self.neurons_config = json.loads(os.environ["NEURONS_CONFIG"])
            self.layers = [self.neurons_config[key] for key in sorted(self.neurons_config, key=lambda x: int(x.split('_')[1]))]
            log_to_host(f"Container with multiple layers: {len(self.layers)} layers", self.container_name)
        elif "NEURONS_FILE_CONFIG" in os.environ:
            with open(os.environ["NEURONS_FILE_CONFIG"], "r") as f:
                self.neurons_config = json.load(f)
            self.layers = [self.neurons_config[key] for key in sorted(self.neurons_config, key=lambda x: int(x.split('_')[1]))]
            log_to_host(f"Container with multiple layers (via file): {len(self.layers)} layers", self.container_name)
        else:
            with open(os.environ["NEURONS_FILE"], "r") as f:
                self.layers = [json.load(f)]  # single layer wrapped in a list
        self.next_nodes = json.loads(os.environ.get("NEXT_NODES", "[]"))
        self.connections = {}
        self.establish_connections()
        log_to_host(f"Layer initialized with expected input dim {self.expected_input_dim}", self.container_name)
    
    def log_msg(self, message, priority=0):
        if priority > 0:
            return
        log_to_host(message, self.container_name)

    def connect_to_node(self, host, port, max_retries=50, retry_interval=2):
        retries = 0
        while retries < max_retries:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                log_to_host(f"Established persistent connection to {host}:{port}", self.container_name)
                return s
            except Exception as e:
                retries += 1
                log_to_host(f"Attempt {retries} to connect to {host}:{port} failed: {e}", self.container_name)
                time.sleep(retry_interval)
        log_to_host(f"Failed to establish connection to {host}:{port} after {max_retries} attempts", self.container_name)
        return None

    def establish_connections(self):
        for node in self.next_nodes:
            host = node["host"]
            port = int(node["port"])
            conn = self.connect_to_node(host, port)
            if conn:
                self.connections[(host, port)] = conn
        log_to_host("Connections established", self.container_name)

    def activate(self, values, func):
        if func == "relu":
            return [max(0, v) for v in values]
        elif func == "sigmoid":
            return [1/(1+math.exp(-v)) for v in values]
        elif func == "softmax":
            m = max(values)
            exps = [math.exp(v - m) for v in values]
            s = sum(exps)
            return [e/s for e in exps]
        else:  # linear
            return values


    def process_matrix(self, input_matrix):
        current_matrix = input_matrix
        expected_dim = self.expected_input_dim   # use local variable instead of self.expected_input_dim
        layer_index = 1
        for layer in self.layers:
            output_matrix = []
            for row in current_matrix:
                if len(row) != expected_dim:
                    raise ValueError(f"Layer {layer_index}: expected input dim {expected_dim}, got {len(row)}")
                raw = [sum(val * w for val, w in zip(row, neuron["weights"])) + neuron["bias"] for neuron in layer]
                activated = self.activate(raw, layer[0].get("activation", "linear"))
                output_matrix.append(activated)
            if output_matrix:
                expected_dim = len(output_matrix[0])
            current_matrix = output_matrix
            layer_index += 1
        self.log_msg(f"Processed matrix through {layer_index-1} layers. Final output dimension: [{len(current_matrix)}, {len(current_matrix[0]) if current_matrix else 0}]", 0)
        return current_matrix

    def forward_result(self, output_matrix):
        # --- New code to compute dimensions and data size ---
        data_dimensions = "unknown"
        if isinstance(output_matrix, list) and output_matrix and isinstance(output_matrix[0], list):
            rows = len(output_matrix)
            cols = len(output_matrix[0])
            data_dimensions = f"[{rows} x {cols}]"
        msg = json.dumps({"matrix": output_matrix}).encode()
        data_size = len(msg)
        self.log_msg(f"Forwarding data with dimensions {data_dimensions} and size {data_size} bytes", 0)
        # --- End new code ---
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
            buffer = []
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer.append(data)
                if len(data) < 4096:  # assume end-of-message
                    break
            full_data = b"".join(buffer).decode()
            if full_data:
                try:
                    content = json.loads(full_data)
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
    try:
        layer = Layer()
        layer.start_server()
    except KeyboardInterrupt:
        print("Shutting down layer.")
    except Exception as e:
        print(f"Error starting layer: {e}")
        log_to_host(f"Error starting layer: {e}", layer.container_name)
