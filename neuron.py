import os
import socket
import json
import threading
import math

# Get configuration from environment variables
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))
EXPECTED_INPUTS = int(os.environ.get("EXPECTED_INPUTS", "1"))
WEIGHTS = json.loads(os.environ.get("WEIGHTS", "[]"))
BIAS = float(os.environ.get("BIAS", "0"))
ACTIVATION = os.environ.get("ACTIVATION", "relu")
# NEXT_NODES is a JSON string representing a list of dicts, e.g.:
#   [{"host": "nn_output_0", "port": "8201"}] or for final neuron: [{"host": "<gateway_ip>", "port": "9000"}]
NEXT_NODES = json.loads(os.environ.get("NEXT_NODES", "[]"))

def relu(x):
    return max(0, x)

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def activate(x):
    if ACTIVATION == "relu":
        return relu(x)
    elif ACTIVATION == "sigmoid":
        return sigmoid(x)
    else:
        # Default: linear activation
        return x

# A lock to protect access to our input collection
inputs_lock = threading.Lock()
collected_inputs = []

def process_and_forward():
    global collected_inputs
    # Compute weighted sum
    weighted_sum = sum(float(val) * float(w) for val, w in zip(collected_inputs, WEIGHTS)) + BIAS
    output = activate(weighted_sum)
    print(f"Processed inputs: {collected_inputs}, Weights: {WEIGHTS}, Bias: {BIAS}, Weighted sum: {weighted_sum}, Output: {output}")
    # Reset inputs for the next inference round
    collected_inputs = []
    # Prepare output message
    message = json.dumps({"value": output}).encode()
    # Forward the result to all next nodes
    for node in NEXT_NODES:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((node["host"], int(node["port"])))
                s.sendall(message)
                print(f"Forwarded output to {node['host']}:{node['port']} with message: {message}")
        except Exception as e:
            print(f"Error forwarding to {node}: {e}")

def handle_client(conn, addr):
    global collected_inputs
    try:
        data = conn.recv(1024).decode()
        if data:
            msg = json.loads(data)
            value = msg.get("value")
            print(f"Received input from {addr}: {value}")
            with inputs_lock:
                collected_inputs.append(value)
                if len(collected_inputs) == EXPECTED_INPUTS:
                    process_and_forward()
    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()

def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", LISTEN_PORT))
        s.listen()
        print(f"Neuron listening on port {LISTEN_PORT} (expecting {EXPECTED_INPUTS} inputs).")
        while True:
            conn, addr = s.accept()
            print(f"Accepted connection from {addr}")
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    server()
