import os
import socket
import json

# Read configuration from environment
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))
NEXT_HOST = os.environ.get("NEXT_HOST")  # For final container, this will be the callback host (e.g. "host.docker.internal")
NEXT_PORT = os.environ.get("NEXT_PORT")  # For final container, this will be the callback port (e.g. "9000")
MOD_STR = os.environ.get("MOD_STR", " processed")

def process_data(data):
    # data is expected to be a dictionary: {"message": str, "counter": int}
    data["message"] += MOD_STR
    data["counter"] += 1
    return data

def handle_client(conn, addr):
    try:
        # Read the full message (for simplicity, we assume the entire JSON is sent at once)
        data = conn.recv(4096).decode() # 4KB buffer
        if not data: # Connection closed
            return
        message = json.loads(data)
        print(f"Received from {addr}: {message}")
        
        # Modify the data
        new_message = process_data(message)
        out_data = json.dumps(new_message).encode()
        
        # Connect to the next host if provided
        if NEXT_HOST and NEXT_PORT:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((NEXT_HOST, int(NEXT_PORT)))
                s.sendall(out_data)
            print(f"Forwarded to {NEXT_HOST}:{NEXT_PORT}: {new_message}")
        else:
            # If no next host is provided, this worker is final.
            print("Final container reached; no forwarding configured.")
    except Exception as e:
        print("Error handling connection:", e)
    finally:
        conn.close()
