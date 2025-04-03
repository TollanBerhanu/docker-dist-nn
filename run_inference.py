import socket
import json
import time
import sys
import threading
from tqdm import tqdm
from run_fcnn import BASE_PORT_FIRST_HIDDEN, BASE_PORT_INCREMENT, CONFIG_FILE, INPUTS_FILE

def wait_for_results():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 9500))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024).decode()
            if data:
                msg = json.loads(data)
                print(f"Received final results: {msg.get('results')}")

def run_inference(input_index=1):
    # Load Inputs
    with open(INPUTS_FILE, "r") as f:
        inputs = json.load(f)["examples"]
    
    if input_index >= len(inputs):
        print(f"Error: Input index {input_index} out of range. Available inputs: 0-{len(inputs)-1}")
        return
    
    input_values = inputs[input_index]
    print(f"Running inference with input values: {input_values}")

    # Load configuration to compute neuron mappings
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    layers = config["layers"]
    
    # The first inference is sent to the first hidden layer (layer index 0)
    first_layer = layers[0]
    base_port = BASE_PORT_FIRST_HIDDEN
    first_hidden_mapping = []
    for j in range(first_layer["nodes"]):
        port = base_port + j + 1
        first_hidden_mapping.append(port)

    start_inference = time.time()
    print(f"Sending inputs to {len(first_hidden_mapping)} neurons in first hidden layer...")
    
    # Create progress bar for tracking the input sending process
    total_messages = len(first_hidden_mapping) #* len(input_values)
    with tqdm(total=total_messages, desc="Sending inputs") as pbar:
        # Send each input value to every neuron in the first hidden layer
        for port in first_hidden_mapping:
            # for idx, val in enumerate(input_values):
                # msg = json.dumps({"value": val, "index": idx}).encode()
            msg = json.dumps({"value": input_values}).encode()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)  # Set a timeout for the connection
                    s.connect(("127.0.0.1", port))
                    s.sendall(msg)
                    # Brief delay to prevent socket exhaustion
                    time.sleep(0.005)
                pbar.update(1)
            except ConnectionRefusedError:
                print(f"Error: Could not connect to neuron on port {port}. Is the neural network running?")
                return
            except OSError as e:
                print(f"Socket error when connecting to port {port}: {e}")
                print("Make sure Docker containers are running and ports are correctly mapped.")
                return
            except Exception as e:
                print(f"Unexpected error when connecting to port {port}: {e}")
                return

    inference_time = time.time() - start_inference
    print("Inference processing initiated in {:.3f} seconds".format(inference_time))

if __name__ == "__main__":
    # Allow specifying which input example to use (default is 1)
    input_index = 1
    if len(sys.argv) > 1:
        try:
            input_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid input index: {sys.argv[1]}. Using default index 1.")

    try:
        run_inference(input_index)
        
        start_time = time.time()
        print("Waiting for results...")
        wait_for_results()
        elapsed_time = time.time() - start_time
        print(f"Time for inference: {elapsed_time:.4f} seconds")

    except Exception as e:
        print(f"Error running inference: {e}")
