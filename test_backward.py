import pytest
import peachygrad_cc as pg_cc
import peachygrad as pg
import math
from autograd import *


#### Data Op Tests ####
def test_transpose():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.transpose(x)

    y_grad = pg.tensor([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    (x_grad,) = y.grad(y_grad)
    x_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_reduce_sum():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.reduce_sum(x, axis=0)

    y_grad = pg.tensor([[1, 2, 3]])
    (x_grad,) = y.grad(y_grad)
    x_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[1, 2, 3], [1, 2, 3]])


def test_reduce_mean():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.reduce_mean(x, axis=0)

    y_grad = pg.tensor([[1, 2, 3]])
    (x_grad,) = y.grad(y_grad)
    x_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[0.5, 1, 1.5], [0.5, 1, 1.5]])


def test_broadcast():
    x = pg.tensor([[1, 2, 3]])
    y = pg.broadcast(x, axis=0, size=4)

    y_grad = pg.ones(y.shape)
    (x_grad,) = y.grad(y_grad)
    x_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[4, 4, 4]])


#### Arithmetic Tests ####
def test_neg():
    x = pg.ones((2, 3))
    neg = -x
    neg_grad = pg.tensor([[1, 2, 3], [4, 5, 6]])
    (x_grad,) = neg.grad(neg_grad)
    x_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[-1, -2, -3], [-4, -5, -6]])


def test_exp():
    x = pg.tensor([1, 2, 3])
    exp = pg.exp(x)
    exp_grad = pg.tensor([4, 5, 6])
    (x_grad,) = exp.grad(exp_grad)
    x_grad.eval()
    assert pg.testing.allclose(
        x_grad.out_tensor,
        pg_cc.tensor([math.exp(1) * 4, math.exp(2) * 5, math.exp(3) * 6]),
        atol=1e-3,
        rtol=1e-4,
    )


def test_log():
    x = pg.tensor([1, 2, 3])
    log = pg.log(x)
    log_grad = pg.tensor([4, 5, 6])
    (x_grad,) = log.grad(log_grad)
    x_grad.eval()
    assert pg.testing.allclose(
        x_grad.out_tensor,
        pg_cc.tensor([4, 2.5, 2]),
    )


def test_add():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.tensor([[1, 2, 3], [4, 5, 6]])
    add = x + y
    add_grad = pg.tensor([[1, 2, 3], [4, 5, 6]])
    x_grad, y_grad = add.grad(add_grad)
    x_grad.eval()
    y_grad.eval()
    assert x_grad.out_tensor == add_grad.out_tensor
    assert y_grad.out_tensor == add_grad.out_tensor


def test_sub():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.tensor([[1, 2, 3], [4, 5, 6]])
    sub = x - y
    sub_grad = pg.tensor([[1, 2, 3], [4, 5, 6]])
    x_grad, y_grad = sub.grad(sub_grad)
    x_grad.eval()
    y_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[1, 2, 3], [4, 5, 6]])
    assert y_grad.out_tensor == pg_cc.tensor([[-1, -2, -3], [-4, -5, -6]])


def test_mul():
    x = pg.tensor([[2, 2, 2], [2, 2, 2]])
    y = pg.tensor([[3, 3, 3], [3, 3, 3]])
    mul = x * y
    mul_grad = pg.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    x_grad, y_grad = mul.grad(mul_grad)
    x_grad.eval()
    y_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[1.5, 3, 4.5], [6, 7.5, 9.0]])
    assert y_grad.out_tensor == pg_cc.tensor([[1, 2, 3], [4, 5, 6]])


def test_div():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.tensor([[7, 8, 9], [10, 11, 12]])
    div = x / y
    div_grad = pg.ones((2, 3))
    x_grad, y_grad = div.grad(div_grad)
    x_grad.eval()
    y_grad.eval()
    assert pg.testing.allclose(
        x_grad.out_tensor,
        pg_cc.tensor(
            [[0.14285715, 0.12500000, 0.11111111], [0.10000000, 0.09090909, 0.08333334]]
        ),
    )
    assert pg.testing.allclose(
        y_grad.out_tensor,
        pg_cc.tensor(
            [
                [-0.02040816, -0.03125000, -0.03703704],
                [-0.04000000, -0.04132232, -0.04166667],
            ]
        ),
    )


def test_matmul():
    x = pg.tensor([[0.25, 0.5, 0.75], [1.0, 1.25, 1.5]])
    y = pg.tensor([[1, 2], [3, 4], [5, 6]])
    mm = x @ y
    mm_grad = pg.tensor([[1, 1], [1, 1]])
    x_grad, y_grad = mm.grad(mm_grad)
    x_grad.eval()
    y_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[3, 7, 11], [3, 7, 11]])
    assert y_grad.out_tensor == pg_cc.tensor([[1.25, 1.25], [1.75, 1.75], [2.25, 2.25]])


#### Autograd Tests ####
def test_ag_simple():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.tensor([[1, 2, 3], [4, 5, 6]])
    res = x + y
    grad_sinks, grad_map = backward(res)

    assert len(grad_map[x]) == 1
    assert len(grad_map[y]) == 1

    for grad_sink in grad_sinks:
        grad_sink.eval()

    grad_x_tensor = grad_map[x][0].out_tensor
    grad_y_tensor = grad_map[y][0].out_tensor
    assert grad_x_tensor == pg_cc.tensor([[1, 1, 1], [1, 1, 1]])
    assert grad_y_tensor == pg_cc.tensor([[1, 1, 1], [1, 1, 1]])

    # test reset logic.
    for grad_sink in grad_sinks:
        grad_sink.reset()

    for grad_sink in grad_sinks:
        grad_sink.eval()

    assert grad_map[x][0].out_tensor == pg_cc.tensor([[1, 1, 1], [1, 1, 1]])
    assert grad_map[y][0].out_tensor == pg_cc.tensor([[1, 1, 1], [1, 1, 1]])
    assert id(grad_map[x][0].out_tensor) == id(grad_x_tensor)
    assert id(grad_map[y][0].out_tensor) == id(grad_y_tensor)


def test_ag_tree():
    x = pg.tensor([1, 2])
    y = pg.tensor([3, 4])
    z = pg.tensor([0.5, 0.75])
    mul0 = x * y
    mul1 = y * z
    sink = mul0 * mul1
    grad_sinks, grad_map = backward(sink)

    assert len(grad_map[x]) == 1
    assert len(grad_map[y]) == 2
    assert len(grad_map[z]) == 1

    for grad_sink in grad_sinks:
        grad_sink.eval()

    grad_x_tensor = grad_map[x][0].out_tensor
    grad_y_tensor0 = grad_map[y][0].out_tensor
    grad_y_tensor1 = grad_map[y][1].out_tensor
    grad_z_tensor = grad_map[z][0].out_tensor
    assert grad_x_tensor == pg_cc.tensor([4.5, 12])
    assert pg_cc.add(grad_y_tensor0, grad_y_tensor1) == pg_cc.tensor([3, 12])
    assert grad_z_tensor == pg_cc.tensor([9, 32])

    # test reset logic.
    for grad_sink in grad_sinks:
        grad_sink.reset()

    for grad_sink in grad_sinks:
        grad_sink.eval()

    assert id(grad_map[x][0].out_tensor) == id(grad_x_tensor)
    assert id(grad_map[y][0].out_tensor) == id(grad_y_tensor0)
    assert id(grad_map[y][1].out_tensor) == id(grad_y_tensor1)
    assert id(grad_map[z][0].out_tensor) == id(grad_z_tensor)
    assert grad_map[x][0].out_tensor == pg_cc.tensor([4.5, 12])
    assert pg_cc.add(
        grad_map[y][0].out_tensor, grad_map[y][1].out_tensor
    ) == pg_cc.tensor([3, 12])
    assert grad_map[z][0].out_tensor == pg_cc.tensor([9, 32])

def test_ag_mlp():
    x = pg.tensor([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    w0 = pg.tensor([[0.5, 0.25], [0.01, 1.2]])
    logits = pg.relu(x @ w0)
    labels = pg.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    loss = pg.loss.categorical_crossentropy(logits, labels)
    _, grad_map = pg.autograd.backward(loss)

    assert len(grad_map[w0]) == 2
    grad_map[w0][0].eval()
    grad_map[w0][1].eval()

    grad_expected = pg_cc.tensor([[-0.0784, 0.0784], [-0.0857, 0.0857]])
    assert pg.testing.allclose(
        pg_cc.add(grad_map[w0][0].out_tensor, grad_map[w0][1].out_tensor),
        grad_expected,
        atol=1e-3,
        rtol=1e-3,
    )
