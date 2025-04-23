# grpc-dist/grpc_node.py
import os
import json
import sys
from concurrent import futures
import grpc
import numpy as np  # Import numpy

# Add proto/ to Python path so generated stubs can be imported
sys.path.insert(0, os.path.join(os.getcwd(), 'proto'))
import dist_nn_pb2
import dist_nn_pb2_grpc

# Removed log_to_host function - use standard print for container logs

class Layer:
    def __init__(self):
        self.container_name = os.environ.get("CONTAINER_NAME", "layer")
        self.listen_port = int(os.environ.get("LISTEN_PORT", "8000"))
        self.expected_input_dim = int(os.environ.get("EXPECTED_INPUT_DIM", "0"))
        
        neurons_config_data = None
        if "NEURONS_CONFIG" in os.environ:
            neurons_config_data = json.loads(os.environ["NEURONS_CONFIG"])
            print(f"({self.container_name}) Loaded neurons config from env var.")
        elif "NEURONS_FILE_CONFIG" in os.environ:
            config_path = os.environ["NEURONS_FILE_CONFIG"]
            try:
                with open(config_path, "r") as f:
                    neurons_config_data = json.load(f)
                print(f"({self.container_name}) Loaded neurons config from file: {config_path}")
            except FileNotFoundError:
                print(f"({self.container_name}) ERROR: Neurons config file not found at {config_path}")
                sys.exit(1)
            except json.JSONDecodeError:
                print(f"({self.container_name}) ERROR: Failed to decode JSON from {config_path}")
                sys.exit(1)
        else:
             # Fallback or error if no config is provided - adjust as needed
             print(f"({self.container_name}) ERROR: No neuron configuration provided (NEURONS_CONFIG or NEURONS_FILE_CONFIG env var).")
             sys.exit(1)

        # Store layers' weights, biases, and activation functions
        self.layer_params = []
        if neurons_config_data:
             sorted_layer_keys = sorted(neurons_config_data, key=lambda x: int(x.split('_')[1]))
             for key in sorted_layer_keys:
                 layer_neurons = neurons_config_data[key]
                 if not layer_neurons: continue # Skip empty layers
                 
                 weights = np.array([neuron["weights"] for neuron in layer_neurons]).T # Transpose for (input_dim, num_neurons)
                 biases = np.array([neuron["bias"] for neuron in layer_neurons])
                 activation = layer_neurons[0].get("activation", "linear") # Assume all neurons in a layer have the same activation
                 self.layer_params.append({"weights": weights, "biases": biases, "activation": activation})
             print(f"({self.container_name}) Initialized {len(self.layer_params)} layer(s).")
        
        self.next_nodes = json.loads(os.environ.get("NEXT_NODES", "[]"))
        print(f"({self.container_name}) Layer initialized. Expected input dim: {self.expected_input_dim}. Next nodes: {self.next_nodes}")

    # Removed log_msg function

    def activate(self, values: np.ndarray, func: str) -> np.ndarray:
        """Applies activation function using numpy."""
        if func == "relu":
            return np.maximum(0, values)
        elif func == "sigmoid":
            return 1 / (1 + np.exp(-values))
        elif func == "softmax":
            # Subtract max for numerical stability before exponentiating
            exps = np.exp(values - np.max(values, axis=-1, keepdims=True))
            return exps / np.sum(exps, axis=-1, keepdims=True)
        else:  # linear
            return values

    def process_matrix(self, input_matrix: np.ndarray) -> np.ndarray:
        """Processes the input matrix through the configured layers using numpy."""
        current_matrix = input_matrix
        expected_dim = self.expected_input_dim
        
        for i, params in enumerate(self.layer_params):
            layer_num = i + 1
            # Input dimension check
            if current_matrix.shape[1] != expected_dim:
                 raise ValueError(f"({self.container_name}) Layer {layer_num}: expected input dim {expected_dim}, got {current_matrix.shape[1]}")

            # Matrix multiplication and bias addition
            raw_output = np.dot(current_matrix, params["weights"]) + params["biases"]
            
            # Activation
            activated_output = self.activate(raw_output, params["activation"])
            
            # Update expected dimension for the next layer
            expected_dim = activated_output.shape[1]
            current_matrix = activated_output
            
        print(f"({self.container_name}) Processed matrix through {len(self.layer_params)} layers. Final output shape: {current_matrix.shape}")
        return current_matrix

class LayerServiceServicer(dist_nn_pb2_grpc.LayerServiceServicer):
    def __init__(self, layer_processor: Layer):
        self.layer_processor = layer_processor

    def Process(self, request: dist_nn_pb2.Matrix, context) -> dist_nn_pb2.Matrix:
        try:
            # Convert request Matrix to numpy array
            # Assuming request.rows is a list of Row messages, each with a 'values' field
            input_list = [list(row.values) for row in request.rows]
            if not input_list:
                 print(f"({self.layer_processor.container_name}) Received empty input matrix.")
                 # Return an empty matrix or handle as an error depending on requirements
                 return dist_nn_pb2.Matrix()
                 
            input_matrix_np = np.array(input_list)
            print(f"({self.layer_processor.container_name}) Received matrix with shape: {input_matrix_np.shape}")

            # Process through local layers using numpy
            output_matrix_np = self.layer_processor.process_matrix(input_matrix_np)

            # If next nodes exist, forward via gRPC
            if self.layer_processor.next_nodes:
                next_node = self.layer_processor.next_nodes[0]
                addr = f"{next_node['host']}:{next_node['port']}"
                print(f"({self.layer_processor.container_name}) Forwarding to next node: {addr}")
                
                # Convert numpy output back to gRPC Matrix for forwarding
                rows_to_send = [dist_nn_pb2.Row(values=row) for row in output_matrix_np.tolist()]
                matrix_to_send = dist_nn_pb2.Matrix(rows=rows_to_send)

                try:
                    with grpc.insecure_channel(addr) as channel:
                        stub = dist_nn_pb2_grpc.LayerServiceStub(channel)
                        # Set a timeout for the RPC call (e.g., 10 seconds)
                        response = stub.Process(matrix_to_send, timeout=10) 
                        print(f"({self.layer_processor.container_name}) Received response from {addr}")
                        return response
                except grpc.RpcError as e:
                     print(f"({self.layer_processor.container_name}) ERROR: gRPC call to {addr} failed: {e.code()} - {e.details()}")
                     context.set_code(e.code())
                     context.set_details(f"Failed to forward request to {addr}: {e.details()}")
                     return dist_nn_pb2.Matrix() # Return empty matrix on error

            # Otherwise (this is the last node), return the local output
            else:
                print(f"({self.layer_processor.container_name}) Reached final node. Returning result.")
                # Convert final numpy output back to gRPC Matrix
                final_rows = [dist_nn_pb2.Row(values=row) for row in output_matrix_np.tolist()]
                return dist_nn_pb2.Matrix(rows=final_rows)

        except ValueError as ve:
             print(f"({self.layer_processor.container_name}) ERROR processing matrix: {ve}")
             context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
             context.set_details(str(ve))
             return dist_nn_pb2.Matrix()
        except Exception as e:
             print(f"({self.layer_processor.container_name}) UNEXPECTED ERROR: {e}")
             context.set_code(grpc.StatusCode.INTERNAL)
             context.set_details(f"An internal error occurred: {e}")
             return dist_nn_pb2.Matrix()


def serve():
    """Starts the gRPC server."""
    try:
        layer_processor = Layer()
    except Exception as e:
         print(f"Failed to initialize Layer: {e}")
         sys.exit(1)
         
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dist_nn_pb2_grpc.add_LayerServiceServicer_to_server(
        LayerServiceServicer(layer_processor), server
    )
    serve_addr = f"0.0.0.0:{layer_processor.listen_port}"
    server.add_insecure_port(serve_addr)
    print(f"({layer_processor.container_name}) gRPC LayerService starting on {serve_addr}...")
    server.start()
    print(f"({layer_processor.container_name}) Server started.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()