# Peachygrad

A tiny autograd library.

## Supported Ops
Arithmetic:
- Add
- Sub
- Mul
- Div
- Exp
- Log
- Matmul
- Rcp (reciprocal)
- Neg (- operator)
- Softmax

Data:
- Transpose
- ReduceSum
- ReduceMean
- Broadcast

Initializers:
- Zeros
- Ones
- Uniform
- Tensor

Transform to other libraries:
- Numpy

## Example

Peachygrad is currently lacking good abstractions, so usage will be clunky. As a minimal example:

```
import peachygrad as pg

# peachygrad is lazy. This function builds a computational graph, and does not
# evaluate anything.
def graph(x: Value, W1: Value, W2: Value):
  return (x @ W1) @ W2

x = pg.random((128, 784))
W1 = pg.random((784, 64))
W2 = pg.random((64, 10))
logits = graph(x, W1, W2)

# grad_map is a dict{pg.Node: list[pg.Node]}. To calculate the overall gradient,
# sum together all partials.
grad_leaves, grad_map = pg.autograd.backward(logits)

# forward eval.
logits.eval()

# backward eval.
for grad in grad_leaves:
    grad.eval()

# reset computation.
logits.reset()
for grad in grad_leaves:
    grad.reset()


# apply gradients
grad_w0 = None
for grad in grad_map[W0]:
    if grad_w0 is None:
        grad_w0 = grad
        continue
    grad_w0 = grad_w0 + grad

grad_w1 = None
for grad in grad_map[W1]:
    if grad_w1 is None:
        grad_w1 = grad
        continue
    grad_w1 = grad_w1 + grad

# accumulate.
grad_w0.eval()
grad_w1.eval()
```

For a more advanced example, check `train_mnist.py`

## Project Status

Peachygrad was mainly a toy project for me to learn more about autograd and kernel optimization. I am not planning on developing it any further.
