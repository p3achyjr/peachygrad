import pytest
import peachygrad_cc as pg_cc
import peachygrad as pg


#### Basic Tests ####
def test_value():
    value_node = pg.tensor([1, 2, 3])
    assert value_node.forward().out_tensor == pg_cc.tensor([1, 2, 3])


def test_add():
    x = pg.tensor([1, 2, 3])
    y = pg.tensor([4, 5, 6])

    res = x + y
    res.forward()

    assert res.out_tensor == pg_cc.tensor([5, 7, 9])


def test_sub():
    x = pg.tensor([3, 2, 1])
    y = pg.tensor([4, 5, 6])

    res = x - y
    res.forward()

    assert res.out_tensor == pg_cc.tensor([-1, -3, -5])


def test_mul():
    x = pg.tensor([1, 2, 3])
    y = pg.tensor([4, 5, 6])

    res = x * y
    res.forward()

    assert res.out_tensor == pg_cc.tensor([4, 10, 18])


def test_div():
    x = pg.tensor([1, 2, 3])
    y = pg.tensor([4, 5, 6])

    res = x / y
    res.forward()

    assert res.out_tensor == pg_cc.tensor([0.25, 0.4, 0.5])


def test_matmul():
    x = pg.tensor([[1, 2], [3, 4]])
    y = pg.tensor([[5, 6], [7, 8]])

    res = x @ y
    res.forward()

    assert res.out_tensor == pg_cc.tensor([[19, 22], [43, 50]])


#### Vec Tests ####
def test_vec_add():
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = pg.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    res = x + y
    res.forward()

    assert res.out_tensor == pg_cc.tensor([13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33])


def test_vec_sub():
    x = pg.tensor([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    y = pg.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    res = x - y
    res.forward()

    assert res.out_tensor == pg_cc.tensor(
        [-1, -3, -5, -7, -9, -11, -13, -15, -17, -19, -21]
    )


def test_vec_mul():
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = pg.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    res = x * y
    res.forward()

    assert res.out_tensor == pg_cc.tensor(
        [12, 26, 42, 60, 80, 102, 126, 152, 180, 210, 242]
    )


def test_vec_div():
    # keep these rational.
    x = pg.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = pg.tensor([2, 5, 8, 5, 2, 3, 4, 2, 5, 2.5, 5])

    res = x / y
    res.forward()

    assert res.out_tensor == pg_cc.tensor(
        [0.5, 0.4, 0.375, 0.8, 2.5, 2, 1.75, 4, 1.8, 4, 2.2]
    )


#### ND Tests ####
def test_nd_add():
    x = pg.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = pg.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

    res = x + y
    res.forward()

    assert res.out_tensor.shape() == pg_cc.shape((2, 2, 2))
    assert res.out_tensor == pg_cc.tensor([[[10, 12], [14, 16]], [[18, 20], [22, 24]]])


def test_nd_mul():
    x = pg.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = pg.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

    res = x * y
    res.forward()

    assert res.out_tensor.shape() == pg_cc.shape((2, 2, 2))
    assert res.out_tensor == pg_cc.tensor([[[9, 20], [33, 48]], [[65, 84], [105, 128]]])
