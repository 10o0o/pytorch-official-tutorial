import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()

print(f"w.grad: {w.grad}")
print(f"b.grad: {b.grad}")


z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

x = torch.tensor([1.0, 2.0], requires_grad=True)
a, b = x
y = torch.stack([a + 2 * b, 3 * a + b])

# y.backward(torch.ones_like(y))
y.backward(torch.tensor([1.0, 0.0]))
print(x.grad)
