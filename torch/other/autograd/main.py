import torch
from torch.autograd import Variable


x = Variable(torch.tensor(1, dtype=torch.float32), requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.001)
for _ in range(10000):
    optimizer.zero_grad()
    y = x * x
    y.backward()
    optimizer.step()
    print(x)

