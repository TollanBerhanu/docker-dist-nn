import socket

HOST = '' # Listen on all interfaces 
PORT = 2345 
LOGFILE = 'neuron_logs.txt'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
    s.bind((HOST, PORT)) 
    s.listen() 
    print(f"Logging server listening on port {PORT}, logging to {LOGFILE}") 
    
    # Create and/or clear the logfile first
    with open(LOGFILE, 'w') as f:
        f.write('')

    while True: 
        conn, addr = s.accept() 
        with conn: 
            data = conn.recv(4096) 
            if data: 
                with open(LOGFILE, 'a') as f: 
                    f.write(data.decode() + "\n") 
                    print(data.decode())

# To kill the server, run:
# lsof -i :2345
# kill -9 <PID>