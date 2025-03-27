import socket
import json
import time
import sys
from tqdm import tqdm

def run_inference(input_index=1):
    # Load Inputs
    with open("config/example_inputs/example_inputs_cali.json", "r") as f:
        inputs = json.load(f)["examples"]
    
    if input_index >= len(inputs):
        print(f"Error: Input index {input_index} out of range. Available inputs: 0-{len(inputs)-1}")
        return
    
    input_values = inputs[input_index]
    num_input = len(input_values)
    print(f"Running inference with input values: {input_values}")

    # Load configuration to compute neuron mappings
    with open("config/config_cali.json", "r") as f:
        config = json.load(f)
    layers = config["layers"]

    BASE_PORT_FIRST_HIDDEN = 1100
    BASE_PORT_INCREMENT = 800
    
    # The first inference is sent to the first hidden layer (layer index 1)
    layer_index = 1
    layer = layers[layer_index]
    base_port = BASE_PORT_FIRST_HIDDEN + (layer_index * BASE_PORT_INCREMENT)
    first_hidden_mapping = []
    for j in range(layer["nodes"]):
        port = base_port + j + 1
        first_hidden_mapping.append(port)

    start_inference = time.time()
    print(f"Sending inputs to {len(first_hidden_mapping)} neurons in first hidden layer...")
    
    # Create progress bar for tracking the input sending process
    total_messages = len(first_hidden_mapping) * len(input_values)
    with tqdm(total=total_messages, desc="Sending inputs") as pbar:
        # Send each input value to every neuron in the first hidden layer
        for port in first_hidden_mapping:
            for idx, val in enumerate(input_values):
                msg = json.dumps({"value": val, "index": idx}).encode()
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect(("localhost", port))
                        s.sendall(msg)
                    pbar.update(1)
                except ConnectionRefusedError:
                    print(f"Error: Could not connect to neuron on port {port}. Is the neural network running?")
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
    
    run_inference(input_index)