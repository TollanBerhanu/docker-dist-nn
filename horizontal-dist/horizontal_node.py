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
            self.layers = [self.neurons_config[key] for key in sorted(self.neurons_config, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else float('inf'))]
            log_to_host(f"Container with multiple layers: {len(self.layers)} layers", self.container_name)
        elif "NEURONS_FILE_CONFIG" in os.environ:
            with open(os.environ["NEURONS_FILE_CONFIG"], "r") as f:
                self.neurons_config = json.load(f)
            self.layers = [self.neurons_config[key] for key in sorted(self.neurons_config, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else float('inf'))]
            log_to_host(f"Container with multiple layers (via file): {len(self.layers)} layers", self.container_name)
        else:
            with open(os.environ["NEURONS_FILE"], "r") as f:
                self.layers = [json.load(f)]  # single layer wrapped in a list
        self.next_nodes = json.loads(os.environ.get("NEXT_NODES", "[]"))
        log_to_host(f"Layer initialized with expected input dim {self.expected_input_dim}", self.container_name)
        # New variables for assembling inputs from multiple nodes:
        self.num_input_nodes = int(os.environ.get("NUM_INPUT_NODES", "1"))
        self.input_chunks = []
        self.input_lock = threading.Lock()
        # New flag to indicate if this container receives raw input directly from inference.
        self.is_input_layer = os.environ.get("IS_INPUT_LAYER", "false").lower() == "true"
        # Remove any empty layer entries (e.g. when a container has no neurons for that layer)
        self.layers = [layer for layer in self.layers if layer]
        log_to_host(f"Filtered layers; now {len(self.layers)} non-empty layers", self.container_name)

    def log_msg(self, message, priority=0):
        if priority > 0:
            return
        log_to_host(message, self.container_name)

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
        # If this layer receives raw data, use self.expected_input_dim; otherwise, use the received row length.
        if self.is_input_layer and len(input_matrix[0]) == self.expected_input_dim:
            expected_dim = self.expected_input_dim
        else:
            expected_dim = len(input_matrix[0])
        layer_index = 1
        current_matrix = input_matrix
        for layer in self.layers:
            output_matrix = []
            for row in current_matrix:
                if len(row) != expected_dim:
                    raise ValueError(f"Layer {layer_index}: expected input dim {expected_dim}, got {len(row)}")
                raw = [sum(val * w for val, w in zip(row, neuron["weights"])) + neuron["bias"] for neuron in layer]
                # Safe activation lookup (we know layer is non-empty)
                func = layer[0].get("activation", "linear")
                activated = self.activate(raw, func)
                output_matrix.append(activated)
            if output_matrix:
                expected_dim = len(output_matrix[0])
            current_matrix = output_matrix
            layer_index += 1
        self.log_msg(f"Processed matrix through {layer_index-1} layers. Final output dimension: [{len(current_matrix)}, {len(current_matrix[0]) if current_matrix else 0}]", 0)
        return current_matrix

    def forward_result(self, output_matrix):
        data_dimensions = "unknown"
        if isinstance(output_matrix, list) and output_matrix and isinstance(output_matrix[0], list):
            rows = len(output_matrix)
            cols = len(output_matrix[0])
            data_dimensions = f"[{rows} x {cols}]"
        msg = json.dumps({"matrix": output_matrix}).encode()
        data_size = len(msg)
        self.log_msg(f"Forwarding data with dimensions {data_dimensions} and size {data_size} bytes", 0)
        for node in self.next_nodes:
            host = node["host"]
            port = int(node["port"])
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((host, port))
                    s.sendall(msg)
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
                    matrix_fragment = content.get("matrix", [])
                    with self.input_lock:
                        self.input_chunks.append(matrix_fragment)
                        self.log_msg(f"Received input fragment from {addr}. Total fragments: {len(self.input_chunks)}", 0)
                        if len(self.input_chunks) >= self.num_input_nodes:
                            if self.num_input_nodes == 1:
                                full_matrix = self.input_chunks[0]
                            else:
                                # Use the minimum number of rows among fragments to avoid index errors.
                                num_rows = min(len(frag) for frag in self.input_chunks)
                                full_matrix = []
                                for i in range(num_rows):
                                    combined_row = []
                                    for frag in self.input_chunks:
                                        combined_row.extend(frag[i])
                                    full_matrix.append(combined_row)
                            self.input_chunks.clear()
                            output = self.process_matrix(full_matrix)
                            self.forward_result(output)
                except Exception as e:
                    self.log_msg(f"Error processing data from {addr}: {e}", 0)
            else:
                self.log_msg(f"Received empty data from {addr}", 0)
        except Exception as e:
            self.log_msg(f"Error handling connection from {addr}: {e}", 0)
        finally:
            conn.close()

    def start_server(self):
        self.log_msg(f"Listening on port {self.listen_port} ...", 0)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", self.listen_port))
                s.listen()
                while True:
                    conn, addr = s.accept()
                    threading.Thread(target=self.handle_connection, args=(conn, addr), daemon=True).start()
        except Exception as e:
            self.log_msg(f"Error starting server: {e}", 0)

if __name__ == "__main__":
    try:
        layer = Layer()
        log_to_host(f"Starting node server...", layer.container_name)
        layer.start_server()
    except KeyboardInterrupt:
        print("Shutting down layer.")
    except Exception as e:
        print(f"Error starting layer: {e}")
        log_to_host(f"Error starting layer: {e}", layer.container_name)