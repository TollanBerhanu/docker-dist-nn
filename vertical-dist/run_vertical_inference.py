import socket
import json
import time
import sys
import threading
from tqdm import tqdm
from run_vertical_fcnn import INPUTS_FILE, CONFIG_FILE

# Constants specific to layer-based FCNN inference
FIRST_LAYER_PORT = 5101    # Must match published port for layer_0 in run_fcnn_layer.py
CALLBACK_PORT = 9500         # Callback port for inference results

def load_example_inputs():
    """Helper to load example_inputs from INPUTS_FILE."""
    with open(INPUTS_FILE, "r") as f:
        return json.load(f)["examples"]

def wait_for_results():
    start_time = time.time()
    # Callback server to capture the final output matrix.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", CALLBACK_PORT))
        s.listen()
        print(f"Waiting for results on port {CALLBACK_PORT}...")
        conn, addr = s.accept()
        with conn:
            data = conn.recv(10240).decode()
            if data:
                try:
                    msg = json.loads(data)
                    # print(f"Received final output matrix: {msg.get('results')}")
                    predicted_matrix = msg.get("results", {}).get("matrix")[0]
                    max_value = max(predicted_matrix)
                    max_index = predicted_matrix.index(max_value)
                    print(f'Predicted Value: {max_index}, Probability: {max_value:.4f}')
                except Exception as e:
                    print(f"Error decoding callback message: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Time for inference: {elapsed_time:.4f} seconds")

def run_inference(input_index=0):
    example_inputs = load_example_inputs()  # load once
    if input_index >= len(example_inputs):
        print(f"Error: Input index {input_index} out of range.")
        return
    print(f"Running layer inference with input no. {input_index+1}")

    # NOTE: Uncomment the next lines to run inference for all test inputs
    # for i in range(len(example_inputs)):
        # print(f"Running inference on input index {i}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(("127.0.0.1", FIRST_LAYER_PORT))

            # NOTE: Use input_index as 'i' to run inference for all test inputs
            test_input = example_inputs[input_index]["input"]
            test_label = example_inputs[input_index]["label"]
            s.sendall(json.dumps({"matrix": [test_input]}).encode())
            print(f"Sent input matrix to layer_0 on port {FIRST_LAYER_PORT}")
            
            # NOTE: Comment the next line to run inference for all test inputs
            return test_input, test_label
    except Exception as e:
        print(f"Error sending input to first layer: {e}")
        return None, None

if __name__ == "__main__":
    input_index = 0
    if len(sys.argv) > 1:
        try:
            input_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid input index: {sys.argv[1]}. Using default index 0.")
    
    try:
        # start listener thread first
        listener = threading.Thread(target=wait_for_results, daemon=True)
        listener.start()
        
        # NOTE: Uncomment the next line to run inference for all test inputs
        # run_inference()

        test_input, test_label = run_inference(input_index)
        print(f"Input matrix: {test_input[0:2]}..., True Label: {test_label}")
        listener.join()

    except Exception as e:
        print(f"Error running inference: {e}")
