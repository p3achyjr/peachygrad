import pytest
import peachygrad_cc as pg_cc
import peachygrad as pg
import numpy as np


#### Value Test ####
def test_value():
    value_node = pg.tensor([1, 2, 3])
    assert value_node.eval().out_tensor == pg_cc.tensor([1, 2, 3])


def test_value_np():
    value_node = pg.tensor(
        np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    )
    assert value_node.eval().out_tensor == pg_cc.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )


#### Data Op Tests ####
def test_id():
    x = pg.tensor([1, 2, 3])
    y = pg_cc.identity(x.out_tensor)
    assert y == x.out_tensor


def test_transpose():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.transpose(x)
    y.eval()
    assert y.out_tensor == pg_cc.tensor([[1, 4], [2, 5], [3, 6]])

    z = pg.transpose(y)
    z.eval()
    assert z.out_tensor == x.out_tensor

    z.reset()
    z.eval()
    assert z.out_tensor == x.out_tensor


#### Basic Arithmetic Tests ####
def test_neg():
    x = pg.tensor([1, 2, 3])

    res = -x
    res.eval()

    assert res.out_tensor == pg_cc.tensor([-1, -2, -3])


def test_add():
    x = pg.tensor([1, 2, 3])
    y = pg.tensor([4, 5, 6])

    res = x + y
    res.eval()

    assert res.out_tensor == pg_cc.tensor([5, 7, 9])


def test_sub():
    x = pg.tensor([3, 2, 1])
    y = pg.tensor([4, 5, 6])

    res = x - y
    res.eval()

    assert res.out_tensor == pg_cc.tensor([-1, -3, -5])


def test_mul():
    x = pg.tensor([1, 2, 3])
    y = pg.tensor([4, 5, 6])

    res = x * y
    res.eval()

    assert res.out_tensor == pg_cc.tensor([4, 10, 18])


def test_div():
    x = pg.tensor([1, 2, 3])
    y = pg.tensor([4, 5, 6])

    res = x / y
    res.eval()

    assert res.out_tensor == pg_cc.tensor([0.25, 0.4, 0.5])


def test_matmul():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.tensor([[1, 2], [3, 4], [5, 6]])

    res = x @ y
    assert res.shape == pg_cc.shape((2, 2))

    res.eval()
    assert res.out_tensor == pg_cc.tensor([[22, 28], [49, 64]])


def test_exp():
    x = pg.tensor([1, 2, 3])
    y = pg.exp(x)
    y.eval()
    assert pg.testing.allclose(
        y.out_tensor,
        pg_cc.tensor([2.7183, 7.3891, 20.0855]),
        rtol=1e-3,
        atol=1e-3,
    )

    y.reset()
    y.eval()
    assert pg.testing.allclose(
        y.out_tensor,
        pg_cc.tensor([2.7183, 7.3891, 20.0855]),
        rtol=1e-3,
        atol=1e-3,
    )

    x.out_tensor = pg_cc.tensor([-1, -2, -3])
    y.reset()
    y.eval()
    assert pg.testing.allclose(
        y.out_tensor,
        pg_cc.tensor([0.3679, 0.1353, 0.0498]),
        rtol=1e-3,
        atol=1e-3,
    )


def test_reduce_sum():
    x = pg.tensor([[1, 2, 3], [4, 5, 6]])
    y = pg.reduce_sum(x, axis=0)
    y.eval()
    assert y.out_tensor == pg_cc.tensor([[5, 7, 9]])

    x.out_tensor = pg_cc.tensor([[1, 2, 3], [1, 2, 3]])
    y.reset()
    y.eval()
    assert y.out_tensor == pg_cc.tensor([[2, 4, 6]])


#### Vec Tests ####
def test_neg():
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    res = -x
    assert res.shape == x.shape

    res.eval()

    assert res.out_tensor == pg_cc.tensor(
        [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11]
    )


def test_vec_add():
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = pg.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    res = x + y
    assert res.shape == x.shape
    assert res.shape == y.shape

    res.eval()

    assert res.out_tensor == pg_cc.tensor([13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33])


def test_vec_sub():
    x = pg.tensor([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    y = pg.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    res = x - y
    assert res.shape == x.shape
    assert res.shape == y.shape

    res.eval()

    assert res.out_tensor == pg_cc.tensor(
        [-1, -3, -5, -7, -9, -11, -13, -15, -17, -19, -21]
    )


def test_vec_mul():
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = pg.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    res = x * y
    assert res.shape == x.shape
    assert res.shape == y.shape

    res.eval()

    assert res.out_tensor == pg_cc.tensor(
        [12, 26, 42, 60, 80, 102, 126, 152, 180, 210, 242]
    )


def test_vec_div():
    # keep these rational.
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = pg.tensor([2, 5, 8, 5, 2, 3, 4, 2, 5, 2.5, 5])

    res = x / y
    assert res.shape == x.shape
    assert res.shape == y.shape

    res.eval()

    assert res.out_tensor == pg_cc.tensor(
        [0.5, 0.4, 0.375, 0.8, 2.5, 2, 1.75, 4, 1.8, 4, 2.2]
    )


def test_vec_matmul():
    x = pg.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    y = pg.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    res = x @ y
    assert res.shape == pg_cc.shape((2, 10))

    res.eval()
    assert res.out_tensor == pg_cc.tensor(
        [[3, 6, 9, 12, 15, 18, 21, 24, 27, 30], [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]]
    )


def test_vec_matmul_aligned():
    x = pg.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )
    y = pg.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )

    res = x @ y
    assert res.shape == pg_cc.shape((8, 8))

    res.eval()
    assert res.out_tensor == pg_cc.tensor(
        [
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
            [36, 72, 108, 144, 180, 216, 252, 288],
        ]
    )


def test_vec_softmax():
    x = pg.tensor([[1, 2, 1, 1, 3, 4, 2, 2]])
    ex = pg.exp(x)
    ex_sum = pg.reduce_sum(ex, axis=1)
    ex_sum_bcast = pg.broadcast(ex_sum, axis=1, size=8)
    probs = ex / ex_sum_bcast

    probs_op = pg.softmax(x)
    probs.eval()
    probs_op.eval()
    assert pg.testing.allclose(
        probs.out_tensor,
        pg_cc.tensor(
            [[0.0259, 0.0704, 0.0259, 0.0259, 0.1913, 0.5200, 0.0704, 0.0704]]
        ),
        atol=1e-3,
        rtol=1e-3,
    )
    assert pg.testing.allclose(
        probs_op.out_tensor,
        pg_cc.tensor(
            [[0.0259, 0.0704, 0.0259, 0.0259, 0.1913, 0.5200, 0.0704, 0.0704]]
        ),
        atol=1e-3,
        rtol=1e-3,
    )

    # test reset logic
    probs_op.reset()
    probs.reset()
    probs_op.eval()
    probs.eval()
    assert pg.testing.allclose(
        probs.out_tensor,
        pg_cc.tensor(
            [[0.0259, 0.0704, 0.0259, 0.0259, 0.1913, 0.5200, 0.0704, 0.0704]]
        ),
        atol=1e-3,
        rtol=1e-3,
    )
    assert pg.testing.allclose(
        probs_op.out_tensor,
        pg_cc.tensor(
            [[0.0259, 0.0704, 0.0259, 0.0259, 0.1913, 0.5200, 0.0704, 0.0704]]
        ),
        atol=1e-3,
        rtol=1e-3,
    )

    # test replacement.
    x.out_tensor = pg_cc.tensor([[1, 0, 0, 1, 2, 0, 0, 3]])
    probs_op.reset()
    probs.reset()
    probs_op.eval()
    probs.eval()
    assert pg.testing.allclose(
        probs.out_tensor,
        pg_cc.tensor(
            [[0.0736, 0.0271, 0.0271, 0.0736, 0.2002, 0.0271, 0.0271, 0.5442]]
        ),
        atol=1e-3,
        rtol=1e-3,
    )
    assert pg.testing.allclose(
        probs_op.out_tensor,
        pg_cc.tensor(
            [[0.0736, 0.0271, 0.0271, 0.0736, 0.2002, 0.0271, 0.0271, 0.5442]]
        ),
        atol=1e-3,
        rtol=1e-3,
    )


#### ND Tests ####
def test_nd_add():
    x = pg.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = pg.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

    res = x + y
    assert res.shape == x.shape
    assert res.shape == y.shape

    res.eval()

    assert res.out_tensor.shape() == pg_cc.shape((2, 2, 2))
    assert res.out_tensor == pg_cc.tensor([[[10, 12], [14, 16]], [[18, 20], [22, 24]]])


def test_nd_mul():
    x = pg.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = pg.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

    res = x * y
    assert res.shape == x.shape
    assert res.shape == y.shape

    res.eval()

    assert res.out_tensor.shape() == pg_cc.shape((2, 2, 2))
    assert res.out_tensor == pg_cc.tensor([[[9, 20], [33, 48]], [[65, 84], [105, 128]]])
