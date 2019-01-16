import torch
import torch.nn as nn

n_in, n_h, n_out, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_in)                                                         # batch_size * n_in
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])  # batch_size * n_out

# input -> linear -> relu -> linear -> sigmoid
model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid()
)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train
for epoch in range(50):
    # forward
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    