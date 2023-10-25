from typing import Optional, Union

import peachygrad_cc as pg_cc

dtype = pg_cc.dtype


def arity_check(arity: int, expected_arity: int, op: str):
    if arity != expected_arity:
        raise Exception(f"{op}, Expected `{expected_arity}` args, got `{arity}.")


class Node(object):
    name: str
    out_tensor: Optional[pg_cc.Tensor]
    evaluated: bool
    shape: pg_cc.shape
    dtype: pg_cc.dtype
    is_unop: bool
    is_binop: bool

    def __init__(self, name: str):
        self.name = name
        self.evaluated = False
        self.out_tensor = None
        self.is_unop = False
        self.is_binop = False

    def __call__(self, *_):
        return self.eval()

    def eval(self):
        raise NotImplementedError("Called `eval` on `Node` base class.")

    def grad(self, out_grad):
        raise NotImplementedError("Called `backward` on `Node` base class.")

    def reset(self):
        raise NotImplementedError("Called `reset` on `Node` base class.")

    def __neg__(self):
        return Neg(self)

    def __add__(self, x):
        return Add(self, x)

    def __sub__(self, x):
        return Sub(self, x)

    def __mul__(self, x):
        return Mul(self, x)

    def __truediv__(self, x):
        return Div(self, x)

    def __matmul__(self, x):
        return Matmul(self, x)

    def __hash__(self):
        return id(self)


class Value(Node):
    """Represents single (mutable) value."""

    def __init__(self, x: pg_cc.Tensor):
        super().__init__("Value")
        self.out_tensor = x
        self.shape = x.shape()
        self.dtype = x.dtype()

    def eval(self):
        self.evaluated = True
        return self

    def grad(self, out_grad: Node):
        return ()

    def reset(self):
        self.evaluated = False

    def __repr__(self):
        return f"pg.Value(\n{self.out_tensor})"


def elemwise_check(op_name: str, lhs: Node, rhs: Node):
    if lhs.dtype != rhs.dtype:
        raise Exception(f"`{op_name}` Mismatched Types: {lhs.dtype}, {rhs.dtype}")
    if lhs.shape != rhs.shape:
        raise Exception(
            f"`{op_name}` Mismatched Shapes: {lhs.shape}, {rhs.shape}."
            + " Shapes must be the same."
        )


#### Unary Operations ####
def unop_init(op: Node, in_node: Node):
    op.in_node = in_node
    op.dtype = in_node.dtype
    op.is_unop = True


def unop_eval(op: Node, kernel, *args):
    if op.evaluated:
        return

    if not op.in_node.evaluated:
        op.in_node.eval()

    if op.out_tensor is None:
        op.out_tensor = kernel(op.in_node.out_tensor, *args)
    else:
        kernel(op.out_tensor, op.in_node.out_tensor, *args)

    op.evaluated = True
    return op


def unop_reset(op: Node):
    if op.in_node.evaluated:
        op.in_node.reset()

    op.evaluated = False


def unop_str(op: Node):
    if op.evaluated:
        return f"pg.{op.name}({op.out_tensor} (evaluated))"

    return f"pg.{op.name}({op.in_node})"


class Identity(Node):
    """Represents Identity Function."""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Identity")
        unop_init(self, x)
        self.shape = x.shape

    def eval(self, x):
        return unop_eval(self, pg_cc.identity(x))

    def grad(self, out_grad: Node):
        return (out_grad,)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Transpose(Node):
    """Represents Transpose"""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Transpose")
        unop_init(self, x)
        self.shape = x.shape.transpose()

    def eval(self):
        return unop_eval(self, pg_cc.transpose)

    def grad(self, out_grad: Node):
        return (Transpose(out_grad),)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Reduce(Node):
    """Reduce sum over a single axis. Multiple axes can be represented as
    multiple of these operations."""

    in_node: Node

    SUM = "sum"
    MEAN = "mean"

    def __init__(self, x: Node, axis: int, mode=SUM):
        if 0 > axis >= len(x.shape):
            raise Exception(
                f"`Reduce{mode.capitalize()}` axis out of bounds: {axis}."
                + f"Num axes: {len(x.shape)}"
            )
        super().__init__(f"Reduce{mode.capitalize()}")
        unop_init(self, x)
        new_shape = list(x.shape)
        new_shape[axis] = 1
        self.axis = axis
        self.size = x.shape[self.axis]
        self.shape = pg_cc.shape(tuple(new_shape))
        self.mode = mode

        if mode == self.SUM:
            self.cc_fn = pg_cc.reduce_sum
        elif mode == self.MEAN:
            self.cc_fn = pg_cc.reduce_mean
        else:
            raise Exception(f"None or Invalid `Reduce` Mode. {mode}")

    def eval(self):
        unop_eval(self, self.cc_fn, self.axis)
        return self

    def grad(self, out_grad: Node):
        if self.mode == self.SUM:
            return (Broadcast(out_grad, self.axis, self.size),)
        elif self.mode == self.MEAN:
            return (
                Broadcast(out_grad, self.axis, self.size)
                * constant(self.in_node.shape, 1.0 / self.size),
            )

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Broadcast(Node):
    """Broadcast along a single axis. Axis must have size 1."""

    in_node: Node

    def __init__(self, x: Node, axis: int, size: int):
        if 0 > axis >= len(x.shape):
            raise Exception(
                f"`Broadcast` axis out of bounds: {axis}. Num axes: {len(x.shape)}"
            )
        if x.shape[axis] != 1:
            raise Exception(f"`Broadcast` must expand an axis with length 1.")
        super().__init__("Broadcast")
        unop_init(self, x)
        new_shape = list(x.shape)
        new_shape[axis] = size
        self.axis = axis
        self.size = size
        self.shape = pg_cc.shape(tuple(new_shape))

    def eval(self):
        unop_eval(self, pg_cc.broadcast, self.axis, self.size)
        return self

    def grad(self, out_grad: Node):
        return (Reduce(out_grad, axis=self.axis, mode=Reduce.SUM),)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Neg(Node):
    """Represents Negation (-x)"""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Neg")
        unop_init(self, x)
        self.shape = x.shape

    def eval(self):
        return unop_eval(self, pg_cc.neg)

    def grad(self, out_grad: Node):
        return (-out_grad,)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Exp(Node):
    """Represents e^x"""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Exp")
        unop_init(self, x)
        self.shape = x.shape

    def eval(self):
        unop_eval(self, pg_cc.exp)
        return self

    def grad(self, out_grad: Node):
        return (out_grad * self,)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Log(Node):
    """Represents ln(x)"""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Log")
        unop_init(self, x)
        self.shape = x.shape

    def eval(self):
        unop_eval(self, pg_cc.log)
        return self

    def grad(self, out_grad: Node):
        return (out_grad * Rcp(self.in_node),)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Rcp(Node):
    """Represents 1 / x"""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Rcp")
        unop_init(self, x)
        self.shape = x.shape

    def eval(self):
        return unop_eval(self, pg_cc.rcp)

    def grad(self, out_grad: Node):
        return (out_grad * -self * self,)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class MaskGt(Node):
    """Creates mask where x = 1.0 if x > c, 0.0 ow."""

    in_node: Node

    def __init__(self, x: Node, c: float):
        super().__init__("MaskGt")
        unop_init(self, x)
        self.c = c
        self.shape = x.shape

    def eval(self):
        return unop_eval(self, pg_cc.mask_gt, self.c)

    def grad(self, _: Node):
        raise NotImplementedError("`grad` not yet implemented for `MaskGt`.")

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


class Relu(Node):
    """Represents ReLU(x)"""

    in_node: Node

    def __init__(self, x: Node):
        super().__init__("Relu")
        unop_init(self, x)
        self.shape = x.shape

    def eval(self):
        return unop_eval(self, pg_cc.max, 0)

    def grad(self, out_grad: Node):
        return (MaskGt(self.in_node, 0) * out_grad,)

    def reset(self):
        unop_reset(self)

    def __repr__(self):
        return unop_str(self)


#### Binary Operations ####
def binop_init(op: Node, lhs: Node, rhs: Node):
    op.lhs = lhs
    op.rhs = rhs
    op.is_binop = True


def binop_eval(op: Node, kernel):
    if op.evaluated:
        return

    if not op.lhs.evaluated:
        op.lhs.eval()
    if not op.rhs.evaluated:
        op.rhs.eval()

    if op.out_tensor is None:
        op.out_tensor = kernel(op.lhs.out_tensor, op.rhs.out_tensor)
    else:
        kernel(op.out_tensor, op.lhs.out_tensor, op.rhs.out_tensor)
    op.evaluated = True
    return op


def binop_reset(op: Node):
    if op.lhs.evaluated:
        op.lhs.reset()
    if op.rhs.evaluated:
        op.rhs.reset()

    op.evaluated = False


def binop_str(op: Node):
    if op.evaluated:
        return f"pg.{op.name}({op.out_tensor} (evaluated))"

    return f"pg.{op.name}({op.lhs}, {op.rhs})"


class Add(Node):
    """Node representing addition."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Add", lhs, rhs)

        super().__init__("Add")
        binop_init(self, lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def eval(self):
        return binop_eval(self, pg_cc.add)

    def grad(self, out_grad: Node):
        return out_grad, out_grad

    def reset(self):
        binop_reset(self)

    def __repr__(self):
        return binop_str(self)


class Sub(Node):
    """Node representing subtraction."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Sub", lhs, rhs)

        super().__init__("Sub")
        binop_init(self, lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def eval(self):
        return binop_eval(self, pg_cc.sub)

    def grad(self, out_grad: Node):
        return out_grad, -out_grad

    def reset(self):
        binop_reset(self)

    def __repr__(self):
        return binop_str(self)


class Mul(Node):
    """Node representing multiplication."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Mul", lhs, rhs)

        super().__init__("Mul")
        binop_init(self, lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def eval(self):
        binop_eval(self, pg_cc.mul)
        return self

    def grad(self, out_grad: Node):
        return out_grad * self.rhs, out_grad * self.lhs

    def reset(self):
        binop_reset(self)

    def __repr__(self):
        return binop_str(self)


class Div(Node):
    """Node representing division."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Div", lhs, rhs)

        super().__init__("Div")
        binop_init(self, lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def eval(self):
        binop_eval(self, pg_cc.div)
        return self

    def grad(self, out_grad: Node):
        # precompute notes to avoid recomputation.
        rcp_rhs = Rcp(self.rhs)

        # 1 / y, x * -(1 / y ^ 2)
        return (out_grad * rcp_rhs, out_grad * self.lhs * rcp_rhs * -rcp_rhs)

    def reset(self):
        binop_reset(self)

    def __repr__(self):
        return binop_str(self)


class Matmul(Node):
    """Node representing matrix multiplication."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        if lhs.dtype != rhs.dtype:
            raise Exception(f"`Matmul` Mismatched Types: {lhs.dtype}, {rhs.dtype}")

        lshape = lhs.shape
        rshape = rhs.shape
        if not 1 <= len(lshape) <= 2:
            raise Exception(f"`Matmul` LHS has wrong number of dims: {len(lshape)}")
        if not 1 <= len(rshape) <= 2:
            raise Exception(f"`Matmul` RHS has wrong number of dims: {len(rshape)}")

        # shape inference.
        if len(rshape) == 1:
            self.shape = pg_cc.shape((lshape[0], rshape[0]))
        else:
            self.shape = pg_cc.shape((lshape[0], rshape[1]))

        self.dtype = lhs.dtype

        super().__init__("Matmul")
        binop_init(self, lhs, rhs)

    def eval(self):
        return binop_eval(self, pg_cc.matmul)

    def grad(self, out_grad: Node):
        grad_lhs = out_grad @ Transpose(self.rhs)
        grad_rhs = Transpose(self.lhs) @ out_grad
        return grad_lhs, grad_rhs

    def reset(self):
        binop_reset(self)

    def __repr__(self):
        return binop_str(self)


# Initializers.
def tensor(x: list):
    return Value(pg_cc.tensor(x))


def placeholder(shape: Union[tuple, pg_cc.shape], dtype=pg_cc.f32):
    if isinstance(shape, tuple):
        shape = pg_cc.shape(shape)
    return Value(pg_cc.zeros(shape, dtype))


def zeros(shape: Union[tuple, pg_cc.shape], dtype=pg_cc.f32):
    if isinstance(shape, tuple):
        shape = pg_cc.shape(shape)
    return Value(pg_cc.zeros(shape, dtype))


def ones(shape: Union[tuple, pg_cc.shape], dtype=pg_cc.f32):
    if isinstance(shape, tuple):
        shape = pg_cc.shape(shape)
    return Value(pg_cc.ones(shape, dtype))


def constant(shape: Union[tuple, pg_cc.shape], c: float, dtype=pg_cc.f32):
    if isinstance(shape, tuple):
        shape = pg_cc.shape(shape)
    return Value(pg_cc.constant(shape, dtype, c))


def uniform(shape: Union[tuple, pg_cc.shape], lo=-1.0, hi=1.0):
    if isinstance(shape, tuple):
        shape = pg_cc.shape(shape)
    return Value(pg_cc.uniform(shape, lo, hi))


# Data Transform
def transpose(x: Node):
    return Transpose(x)


def reduce_sum(x: Node, axis: int):
    return Reduce(x, axis, mode=Reduce.SUM)


def reduce_mean(x: Node, axis: int):
    return Reduce(x, axis, mode=Reduce.MEAN)


def broadcast(x: Node, axis: int, size: int):
    return Broadcast(x, axis, size)


# Arithmetic
def neg(x: Node):
    return Neg(x)


def exp(x: Node):
    return Exp(x)


def log(x: Node):
    return Log(x)


def rcp(x: Node):
    return Rcp(x)


def relu(x: Node):
    return Relu(x)


def add(x: Node, y: Node):
    return Add(x, y)


def sub(x: Node, y: Node):
    return Sub(x, y)


def mul(x: Node, y: Node):
    return Mul(x, y)


def div(x: Node, y: Node):
    return Div(x, y)


def matmul(x: Node, y: Node):
    return Matmul(x, y)


def softmax(x: Node):
    """Assumes `x` is of shape (B, L)"""
    _, l = x.shape[0], x.shape[1]
    ex = exp(x)
    ex_sum = reduce_sum(ex, axis=1)
    ex_sum_bcast = broadcast(ex_sum, axis=1, size=l)
    probs = ex / ex_sum_bcast
    return probs
