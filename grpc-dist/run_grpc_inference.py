# grpc-dist/run_grpc_inference.py
import os
import sys
import json
import time
import grpc
import argparse
import logging
import numpy as np
import math
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
DEFAULT_INPUTS_FILE = os.path.join(SCRIPT_DIR, "../config/example_inputs/mnist_examples_60000.json")
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

def run_batch_inference(
    input_batch: list,
    first_layer_address: str,
    timeout: float
) -> tuple:
    """Runs a batch inference pass via gRPC."""
    rows = [dist_nn_pb2.Row(values=inp) for inp in input_batch]
    matrix_msg = dist_nn_pb2.Matrix(rows=rows)
    start_time = time.monotonic()
    try:
        with grpc.insecure_channel(first_layer_address) as channel:
            grpc.channel_ready_future(channel).result(timeout=timeout/2)
            stub = dist_nn_pb2_grpc.LayerServiceStub(channel)
            response = stub.Process(matrix_msg, timeout=timeout)
            elapsed = time.monotonic() - start_time
            if not response.rows:
                logging.warning("Received empty matrix in batch response.")
                return None, elapsed
            output = np.array([row.values for row in response.rows])
            return output, elapsed
    except grpc.RpcError as e:
        elapsed = time.monotonic() - start_time
        logging.error(f"Batch gRPC call failed after {elapsed:.4f}s: {e.code()} - {e.details()}")
        return None, elapsed
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logging.error(f"Unexpected batch error after {elapsed:.4f}s: {e}")
        return None, elapsed

# --- Main Execution ---

def main(inputs_file: str, input_index: Optional[int], first_layer_port: int, grpc_timeout: float, batch_size: Optional[int]):
    """Loads data and runs inference for the specified input index or in batches."""
    
    try:
        examples = load_example_inputs(inputs_file)
        num_examples = len(examples)
        if not examples:
            logging.error("No examples loaded, cannot run inference.")
            return
    except Exception:
        return # Error already logged

    if input_index is not None:
        if not 0 <= input_index < len(examples):
            logging.error(f"Input index {input_index} is out of range (0-{len(examples)-1}).")
            return
        examples = [examples[input_index]] # Select specific example

    correct_predictions = 0
    total_start = time.monotonic()

    first_layer_address = f"127.0.0.1:{first_layer_port}"
    # determine batch processing
    if batch_size is None or batch_size >= num_examples:
        # single batch of all examples
        inputs = [ex.get("input") for ex in examples]
        labels = [ex.get("label") for ex in examples]
        outputs, elapsed = run_batch_inference(inputs, first_layer_address, grpc_timeout)
        if outputs is not None:
            for idx, output_vals in enumerate(outputs):
                pred = np.argmax(output_vals) if output_vals.size>0 else None
                if labels[idx] == pred:
                    correct_predictions += 1
        logging.info(f"Batch inference completed in {elapsed:.4f} seconds for {num_examples} examples.")
    else:
        # process in chunks
        num_batches = math.ceil(num_examples / batch_size)
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, num_examples)
            batch_ex = examples[start:end]
            inputs = [ex.get("input") for ex in batch_ex]
            labels = [ex.get("label") for ex in batch_ex]
            outputs, elapsed = run_batch_inference(inputs, first_layer_address, grpc_timeout)
            if outputs is not None:
                for idx, output_vals in enumerate(outputs):
                    pred = np.argmax(output_vals) if output_vals.size>0 else None
                    if labels[idx] == pred:
                        correct_predictions += 1
            logging.info(f"Batch {b+1}/{num_batches} completed in {elapsed:.4f} seconds ({end-start} examples).")

    total_elapsed = time.monotonic() - total_start
    logging.info(f"Total inference time: {total_elapsed:.4f} seconds")
    logging.info(f"Correct predictions: {correct_predictions} out of {num_examples}")
    logging.info("Inference process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a distributed FCNN via gRPC.")
    parser.add_argument(
        "input_index",
        type=int,
        nargs='?', 
        help="Index of the input example to use (default: all examples)."
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of examples per batch (default: all examples in one batch)."
    )
    args = parser.parse_args()

    main(args.inputs, args.input_index, args.port, args.timeout, args.batch_size)