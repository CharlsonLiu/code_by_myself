import torch


x = torch.tensor([[1,2],
                  [3,4],
                  [5,6]])
y = torch.tensor([1,2,3])

out = torch.einsum('fe,f->e',x,y)
print(out)