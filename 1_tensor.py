import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"x_ones: {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"x_rand: {x_rand}")

shape = (
    2,
    3,
)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


print(f"rand_tensor: {rand_tensor}")
print(f"ones_tensor: {ones_tensor}")
print(f"zeros_tensor: {zeros_tensor}")

tensor = torch.rand(3, 4)

print(f"tensor.shape: {tensor.shape}")
print(f"tensor.dtype: {tensor.dtype}")
print(f"tensor.device: {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"tensor[0]: {tensor[0]}")
print(f"tensor[:, 0]: {tensor[:, 0]}")
print(f"tensor[..., -1]: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(y3)
print(z3)

agg = tensor.sum()
print(f"agg: {agg}")
agg_item = agg.item()
print(f"agg_item: {agg_item}")
print(agg_item, type(agg_item))

print(f"tensor: {tensor}")
tensor.add_(5)
print(f"tensor: {tensor}")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(t)
print(n)

n = np.ones(5)
t = torch.from_numpy(n)

print(f"n: {n}")
print(f"t: {t}")
