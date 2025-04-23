# grpc-dist/run_grpc_inference.py
import os
import sys
import json
import time
import grpc
import argparse
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Ensure local 'proto' directory is on sys.path for generated stubs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'proto'))
try:
    import dist_nn_pb2
    import dist_nn_pb2_grpc
except ImportError:
    logging.error("Failed to import generated gRPC modules. "
                  "Ensure they are generated in the 'proto' directory "
                  "and the directory contains an '__init__.py' file.")
    sys.exit(1)

# --- Configuration ---
DEFAULT_INPUTS_FILE = os.path.join(SCRIPT_DIR, "../config/example_inputs/mnist_examples_5.json")
DEFAULT_FIRST_LAYER_PORT = 5101 # Must match published port in run_grpc_fcnn.py
DEFAULT_GRPC_TIMEOUT = 10 # seconds

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_example_inputs(inputs_file: str) -> List[Dict[str, Any]]:
    """Loads example inputs from the specified JSON file."""
    try:
        with open(inputs_file, "r") as f:
            data = json.load(f)
            examples = data.get("examples", [])
            if not examples:
                 logging.warning(f"No 'examples' found in {inputs_file}.")
            return examples
    except FileNotFoundError:
        logging.error(f"Inputs file not found: {inputs_file}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing inputs file {inputs_file}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading inputs: {e}")
        raise

def run_single_inference(
    input_data: List[float],
    true_label: Optional[Any],
    first_layer_address: str,
    timeout: float
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Runs a single inference pass via gRPC."""
    
    # Prepare gRPC request message
    # Input data is expected to be a single row (list of floats)
    if not isinstance(input_data, list):
        logging.error(f"Invalid input data format. Expected list[float], got {type(input_data)}")
        return None, None
        
    row_msg = dist_nn_pb2.Row(values=input_data)
    matrix_msg = dist_nn_pb2.Matrix(rows=[row_msg]) # Send as a 1-row matrix

    start_time = time.monotonic()
    try:
        # Establish insecure channel and create stub
        with grpc.insecure_channel(first_layer_address) as channel:
            # Wait for channel to be ready (optional, but good practice)
            try:
                 grpc.channel_ready_future(channel).result(timeout=timeout/2) # Wait half the total timeout
                 logging.info(f"gRPC channel to {first_layer_address} is ready.")
            except grpc.FutureTimeoutError:
                 logging.error(f"Timeout waiting for gRPC channel to {first_layer_address} to become ready.")
                 return None, None

            stub = dist_nn_pb2_grpc.LayerServiceStub(channel)
            logging.info(f"Sending inference request to {first_layer_address}...")
            
            # Make the RPC call with timeout
            response = stub.Process(matrix_msg, timeout=timeout)
            
            elapsed_time = time.monotonic() - start_time
            logging.info(f"Received response from gRPC server in {elapsed_time:.4f} seconds.")

            # Process the response
            if not response.rows:
                logging.warning("Received empty matrix in response.")
                return None, elapsed_time
                
            # Assuming the output is also a single row
            output_values = np.array(response.rows[0].values)
            return output_values, elapsed_time

    except grpc.RpcError as e:
        elapsed_time = time.monotonic() - start_time
        logging.error(f"gRPC call failed after {elapsed_time:.4f}s: {e.code()} - {e.details()}")
        return None, elapsed_time
    except Exception as e:
        elapsed_time = time.monotonic() - start_time
        logging.error(f"An unexpected error occurred during inference after {elapsed_time:.4f}s: {e}")
        return None, elapsed_time


# --- Main Execution ---

def main(inputs_file: str, input_index: int, first_layer_port: int, grpc_timeout: float):
    """Loads data and runs inference for the specified input index."""
    
    try:
        examples = load_example_inputs(inputs_file)
    except Exception:
        return # Error already logged

    if not examples:
        logging.error("No examples loaded, cannot run inference.")
        return

    if not 0 <= input_index < len(examples):
        logging.error(f"Input index {input_index} is out of range (0-{len(examples)-1}).")
        return

    selected_example = examples[input_index]
    test_input = selected_example.get("input")
    test_label = selected_example.get("label") # Optional

    if test_input is None:
        logging.error(f"Example at index {input_index} does not contain 'input' data.")
        return

    logging.info(f"Running inference for input index {input_index} from {inputs_file}.")
    if test_label is not None:
        logging.info(f"True Label: {test_label}")
    
    first_layer_address = f"127.0.0.1:{first_layer_port}"

    output_values, elapsed_time = run_single_inference(test_input, test_label, first_layer_address, grpc_timeout)

    if output_values is not None and elapsed_time is not None:
        logging.info(f"Inference completed in {elapsed_time:.4f} seconds.")
        logging.info(f"Raw output vector (first 10 elements): {output_values[:10]}")
        
        # Calculate prediction (index of max value)
        if output_values.size > 0:
             predicted_index = np.argmax(output_values)
             probability = output_values[predicted_index]
             logging.info(f"Predicted Class: {predicted_index}, Probability: {probability:.4f}")
        else:
             logging.warning("Output vector is empty.")

    else:
        logging.error("Inference failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a distributed FCNN via gRPC.")
    parser.add_argument(
        "input_index",
        type=int,
        nargs='?', # Make index optional
        default=0,
        help="Index of the input example to use from the inputs file (default: 0)."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=DEFAULT_INPUTS_FILE,
        help=f"Path to the example inputs JSON file (default: {DEFAULT_INPUTS_FILE})."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_FIRST_LAYER_PORT,
        help=f"Port number of the first layer container (default: {DEFAULT_FIRST_LAYER_PORT})."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_GRPC_TIMEOUT,
        help=f"gRPC call timeout in seconds (default: {DEFAULT_GRPC_TIMEOUT})."
    )
    args = parser.parse_args()

    main(args.inputs, args.input_index, args.port, args.timeout)