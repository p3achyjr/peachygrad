import pytest
import peachygrad_cc as pg_cc
import peachygrad as pg
from autograd import *


#### Data Op Tests ####
def test_transpose():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.transpose(x)

    y_grad = pg.tensor([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    (x_grad,) = y.grad(y_grad)
    x_grad.eval()
    assert x_grad.out_tensor == pg_cc.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


#### Arithmetic Tests ####
def test_add():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.tensor([[1, 2, 3], [4, 5, 6]])
    add = x + y
    add_grad = pg.tensor([[1, 2, 3], [4, 5, 6]])
    x_grad, y_grad = add.grad(add_grad)
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
