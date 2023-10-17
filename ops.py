from typing import Optional

import peachygrad_cc as pg_cc


def arity_check(arity: int, expected_arity: int, op: str):
    if arity != expected_arity:
        raise Exception(f"{op}, Expected `{expected_arity}` args, got `{arity}.")


class Node:
    name: str
    out_tensor: Optional[pg_cc.Tensor]
    shape: tuple
    dtype: pg_cc.dtype

    def __init__(self, name: str):
        self.name = name
        self.out_tensor = None

    def __call__(self, *_):
        return self.forward()

    def forward(self):
        raise NotImplementedError("Called `forward` on `Node` base class.")

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


class Value(Node):
    """Represents single (mutable) value."""

    def __init__(self, x: pg_cc.Tensor):
        super().__init__("Value")
        self.out_tensor = x
        self.shape = tuple(x.shape())
        self.dtype = x.dtype()

    def forward(self):
        return self

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


def binop_init(op: Node, op_name: str, lhs: Node, rhs: Node):
    op.name = op_name
    op.lhs = lhs
    op.rhs = rhs
    op.out_tensor = None


def binop_forward(op: Node, kernel):
    if op.lhs.out_tensor is None:
        op.lhs.forward()
    if op.rhs.out_tensor is None:
        op.rhs.forward()

    if op.out_tensor is None:
        op.out_tensor = kernel(op.lhs.out_tensor, op.rhs.out_tensor)
    else:
        kernel(op.out_tensor, op.lhs.out_tensor, op.rhs.out_tensor)
    return op


def binop_str(op: Node):
    if op.out_tensor:
        return f"pg.{op.name}({op.out_tensor} (evaluated))"

    return f"pg.{op.name}({op.lhs}, {op.rhs})"


class Add(Node):
    """Node representing addition."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Add", lhs, rhs)
        binop_init(self, "Add", lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def forward(self):
        return binop_forward(self, pg_cc.add)

    def __repr__(self):
        return binop_str(self)


class Sub(Node):
    """Node representing subtraction."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Sub", lhs, rhs)
        binop_init(self, "Sub", lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def forward(self):
        return binop_forward(self, pg_cc.sub)

    def __repr__(self):
        return binop_str(self)


class Mul(Node):
    """Node representing multiplication."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Mul", lhs, rhs)
        binop_init(self, "Mul", lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def forward(self):
        return binop_forward(self, pg_cc.mul)

    def __repr__(self):
        return binop_str(self)


class Div(Node):
    """Node representing division."""

    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        elemwise_check("Mul", lhs, rhs)
        binop_init(self, "Div", lhs, rhs)
        self.shape = lhs.shape
        self.dtype = lhs.dtype

    def forward(self):
        return binop_forward(self, pg_cc.div)

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
            self.shape = (lshape[0], rshape[0])
        else:
            self.shape = (lshape[0], rshape[1])

        self.dtype = lhs.dtype
        binop_init(self, "Matmul", lhs, rhs)

    def forward(self):
        return binop_forward(self, pg_cc.matmul)

    def __repr__(self):
        return binop_str(self)


# Functional API.
def tensor(x: list):
    return Value(pg_cc.tensor(x))


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
