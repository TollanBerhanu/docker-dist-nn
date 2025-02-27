# Docker Distributed Neural Network
This repo contains a distributed neural network implementation using docker containers.

Run this commands in the root directory to get started:

- Build the image
``` bash
docker build -t chain_image .
```
- Install required dependencies using poetry
``` bash
poetry install && poetry shell
```

- Run the contaier chain
``` bash
python3 run_chain.py
```