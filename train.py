import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from model import LSTM_Model
from dataset import get_data

path = "data1.txt"
data_loader , vocab , not_used_in_this_code = get_data(path)

model = LSTM_Model(len(vocab))

epochs = 120
learning_rate = 0.001

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters() , lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()



losses = []
for epoch in range(epochs):
    total_loss = 0

    for (batch_input , target) in data_loader:
        optimizer.zero_grad()
        batch_input = batch_input.to(device)
        target = target.to(device)

        pred = model(batch_input)
        loss = loss_fn(pred , target)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    losses.append(total_loss)
    print(f"Epoch: {epoch+1} , Loss: {total_loss}")

plt.plot(range(1, epochs + 1), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.savefig("results/loss_graph.png")
# plt.show()



torch.save(model.state_dict(), "checkpoint/next-word-predictor.pth")


