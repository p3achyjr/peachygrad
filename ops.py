from typing import Optional

import peachygrad_cc as pg_cc

dtype = pg_cc.dtype


def arity_check(arity: int, expected_arity: int, op: str):
    if arity != expected_arity:
        raise Exception(f"{op}, Expected `{expected_arity}` args, got `{arity}.")


class Node:
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
    op.unop = True


def unop_eval(op: Node, kernel):
    if op.evaluated:
        return

    if not op.in_node.evaluated:
        op.in_node.eval()

    if op.out_tensor is None:
        op.out_tensor = kernel(op.in_node.out_tensor)
    else:
        kernel(op.out_tensor, op.in_node.out_tensor)

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
    if not op.rhs.out_tensor:
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
        return binop_eval(self, pg_cc.mul)

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
        return binop_eval(self, pg_cc.div)

    def grad(self, out_grad: Node):
        raise NotImplementedError("Div Grad not yet implemented.")

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


def zeros(shape: tuple, dtype=pg_cc.f32):
    shape = pg_cc.shape(shape)
    return pg_cc.zeros(shape, dtype)


def ones(shape: tuple, dtype=pg_cc.f32):
    shape = pg_cc.shape(shape)
    return pg_cc.ones(shape, dtype)


# Data Transform
def transpose(x: Node):
    return Transpose(x)


# Arithmetic
def neg(x: Node):
    return Neg(x)


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
