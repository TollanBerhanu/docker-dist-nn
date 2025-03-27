# Docker Distributed Neural Network
This repo contains a distributed neural network implementation using docker containers.

Run this commands in the root directory to get started:

- Build the image
``` bash
docker build -t fcnn_image .
```

- Install required dependencies using poetry (Linux)
``` bash
poetry install && poetry shell
```
- Install required dependencies using poetry (MacOS)
``` bash
poetry install --no-root && eval $(poetry env activate)
```

- Run the contaier chain
``` bash
python3 run_fcnn.py
```