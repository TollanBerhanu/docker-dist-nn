import os
import socket
import json
import threading

# Get environment variables
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))
NEXT_HOST = os.environ.get("NEXT_HOST")
NEXT_PORT = os.environ.get("NEXT_PORT")
PAYLOAD_STR = os.environ.get("PAYLOAD_STR")

def process_payload(payload: dict) -> dict:
    payload["message"] += PAYLOAD_STR
    payload["counter"] += 1
    return payload

def handle_client(conn, addr):
    try:
        payload = conn.recv(1024).decode() # use 1KB buffer
        if not payload:
            return
        message = json.loads(payload)
        print(f"Received from {addr}: {message}")
        
        # Modify the payload
        new_message = process_payload(message)
        out_payload = json.dumps(new_message).encode()
        
        if NEXT_HOST and NEXT_PORT:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((NEXT_HOST, int(NEXT_PORT)))  # Connect to the next host
                s.sendall(out_payload)                  # Send modified payload to next host
            print(f"Forwarded to {NEXT_HOST}:{NEXT_PORT}: {new_message}")
        else:
            # If no next host is provided in envrionment variables, this is the final container.
            print("*** Final container reached  ***")
    except Exception as e:
        print("Error: ", e)
    finally:
        conn.close()

def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", LISTEN_PORT))    # Bind to all ips (if left empty)
        s.listen()
        print(f"Neuron is listening on port {LISTEN_PORT}...")
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)
            # threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    server()
