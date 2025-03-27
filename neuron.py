import os
import socket
import json
import threading
import math
from datetime import datetime  # changed code

# Get configuration from environment variables
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "container")  # Added container name for logging
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))
EXPECTED_INPUTS = int(os.environ.get("EXPECTED_INPUTS", "1"))
WEIGHTS = json.loads(os.environ.get("WEIGHTS", "[]"))
BIAS = float(os.environ.get("BIAS", "0"))
ACTIVATION = os.environ.get("ACTIVATION", "relu")
# NEXT_NODES is a JSON string representing a list of dicts, e.g.:
#   [{"host": "nn_output_0", "port": "8201"}] or for final neuron: [{"host": "<gateway_ip>", "port": "9000"}]
NEXT_NODES = json.loads(os.environ.get("NEXT_NODES", "[]"))

# New log_msg helper that sends log messages to the host logging server
def log_msg(message, priority=0):
    if priority > 0:
        return
    timestamped = f"{datetime.now().isoformat()} | {message}"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as log_sock:
            # Using host.docker.internal to reach the host machine on Mac
            log_sock.connect(("host.docker.internal", 2345))
            log_sock.sendall(timestamped.encode())
    except Exception as e:
        # If logging fails, fallback to writing locally (or silently ignore)
        print(f"Logging failed: {e}")
        pass
    # Also log locally for container debugging.
    print(timestamped)

def relu(x):
    return max(0, x)

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def activate(x):
    log_msg(f"Activating with {ACTIVATION} function on value: {x}", 1)
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
    log_msg(f"\t---> {CONTAINER_NAME} processing inputs: {collected_inputs[:2] + ['...'] + collected_inputs[-2:]}", 0)
    # Compute weighted sum
    weighted_sum = sum(float(val) * float(w) for val, w in zip(collected_inputs, WEIGHTS)) + BIAS
    log_msg(f"\t---> Computed weighted sum: {weighted_sum}", 0)
    
    output = activate(weighted_sum)
    log_msg(f"\t---> Activation output: {output}", 2)
    log_msg(f"\t---> Processed inputs: {collected_inputs}, Weights: {WEIGHTS}, Bias: {BIAS}, Weighted sum: {weighted_sum}, Output: {output}", 1)
    # Reset inputs for the next inference round
    collected_inputs = []
    # Prepare output output_values
    output_values = json.dumps({"value": output}).encode()
    log_msg(f"\t---> Prepared output_values to forward: {output_values}", 2)
    # Forward the result to all next nodes
    for node in NEXT_NODES:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((node["host"], int(node["port"])))
                s.sendall(output_values)
                log_msg(f"\t---> Forwarded output to {node['host']}:{node['port']} with value: {output_values}", 0)
        except Exception as e:
            log_msg(f"\t---> Error forwarding to {node}: {e}", 0)

def handle_neuron(conn, addr):
    global collected_inputs
    log_msg(f"{CONTAINER_NAME} handling neuron on {addr}", 1)
    try:
        while True:
            data = conn.recv(1024).decode()
            if not data:
                break
            try:
                msg = json.loads(data)
                value = msg.get("value")
                log_msg(f"\t---> {CONTAINER_NAME} received input: {value}", 0)
                with inputs_lock:
                    collected_inputs.append(value)
                    log_msg(f"\t---> Collected inputs: {collected_inputs}", 2)
                    if len(collected_inputs) == EXPECTED_INPUTS:
                        process_and_forward()
            except Exception as e:
                log_msg(f"\t---> Error processing data: {e}", 0)
    except Exception as e:
        log_msg(f"\t---> Error in handle_neuron: {e}", 0)
    # finally:
    #     conn.close()
    #     log_msg(f"\t---> Closed connection from {addr}", 0)

def server():
    log_msg(f"{CONTAINER_NAME} listening on port {LISTEN_PORT} ...", 0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", LISTEN_PORT))
        s.listen()
        # log_msg(f"{CONTAINER_NAME} neuron listening on port {LISTEN_PORT}.", 0)
        while True:
            conn, addr = s.accept()
            log_msg(f"\t---> Accepted connection from {addr}", 0)
            threading.Thread(target=handle_neuron, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":

    log_msg(f"""Configuration - 
        LISTEN_PORT: {LISTEN_PORT},
        EXPECTED_INPUT_LENGTH: {EXPECTED_INPUTS},
        WEIGHTS: {WEIGHTS[:2] + ["..."] + WEIGHTS[-2:]}, 
        BIAS: {BIAS}, 
        ACTIVATION: {ACTIVATION}, 
        NEXT_NODES: {NEXT_NODES}
        """, 0)

    server()