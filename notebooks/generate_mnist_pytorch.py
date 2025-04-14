import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, time

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])  # Convert images to tensors (scales to [0, 1])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Extract 5 example inputs
examples = []
for i in range(5):
    image, label = trainset[i]
    image = image.view(-1).tolist()  # Flatten 28x28 image to a list of 784 elements
    examples.append(image)

# Define the Fully Connected Neural Network (FCNN) model
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input: 784 (28x28), Hidden layer 1: 128 nodes
        self.fc2 = nn.Linear(128, 64)   # Hidden layer 2: 64 nodes
        self.fc3 = nn.Linear(64, 10)    # Output layer: 10 nodes (one per digit)

    def forward(self, x):
        x = F.relu(self.fc1(x))         # ReLU activation for first hidden layer
        x = F.relu(self.fc2(x))         # ReLU activation for second hidden layer
        x = self.fc3(x)                 # Output logits (no softmax here, handled by loss function)
        return x

# Instantiate the model, loss function, and optimizer
model = FCNN()
criterion = nn.CrossEntropyLoss()  # Expects logits, applies log_softmax internally
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for 5 epochs
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1)  # Flatten images to (batch_size, 784)
        optimizer.zero_grad()                    # Clear gradients
        outputs = model(inputs)                  # Forward pass
        loss = criterion(outputs, labels)        # Compute loss
        loss.backward()                          # Backward pass
        optimizer.step()                         # Update weights
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}')

start_time = time.time()
# Run inference on the first example
first_example = torch.tensor(examples[0]).unsqueeze(0)  # Add batch dimension (1, 784)
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    logits = model(first_example)
    predicted_class = torch.argmax(logits, dim=1).item()
    true_class = trainset[0][1]
    print(f'Predicted class: {predicted_class}, True class: {true_class}')

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# # Extract and format model weights for JSON
# layers = [model.fc1, model.fc2, model.fc3]
# layers_list = []
# for i, layer in enumerate(layers):
#     weights = layer.weight.data
#     biases = layer.bias.data
#     num_nodes = biases.shape[0]
#     neurons = []
#     for j in range(num_nodes):
#         neuron_weights = weights[j, :].tolist()
#         neuron_bias = biases[j].item()
#         # Set activation based on layer position
#         activation = "relu" if i < len(layers) - 1 else "softmax"
#         neurons.append({
#             "weights": neuron_weights,
#             "bias": neuron_bias,
#             "activation": activation
#         })
#     layer_type = "hidden" if i < len(layers) - 1 else "output"
#     layers_list.append({
#         "type": layer_type,
#         "nodes": num_nodes,
#         "neurons": neurons
#     })

# # Create the final dictionary for JSON
# final_dict = {
#     "examples": examples,
#     "model": {"layers": layers_list}
# }

# # Save to JSON file
# with open('mnist_model_and_examples.json', 'w') as f:
#     json.dump(final_dict, f, indent=4)

# print("Example inputs and model weights have been saved to 'mnist_model_and_examples.json'")