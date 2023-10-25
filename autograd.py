import peachygrad as pg
import peachygrad_cc as pg_cc

import collections


def backward(sink: pg.Node):
    """
    Initially, add sink to worklist. At each iteration, pop one element off the
    worklist, construct its gradient graphs, link its gradients to its corresponding
    nodes, and continue. Stop when worklist is empty (Values return an empty tuple).

    To start, we can actually just worry about `Value` nodes, since those are the
    only nodes we should be applying gradients to.
    """

    grad_map = collections.defaultdict(list)
    backward_sinks = []
    out_grad = pg.Value(pg_cc.ones(sink.shape, sink.dtype))
    worklist = collections.deque([(sink, out_grad)])
    while len(worklist) > 0:
        node, node_grad = worklist.popleft()
        grads = node.grad(node_grad)
        if node.is_binop:
            lhs_grad, rhs_grad = grads
            if isinstance(node.lhs, pg.Value):
                grad_map[node.lhs].append(lhs_grad)
                backward_sinks.append(lhs_grad)
            if isinstance(node.rhs, pg.Value):
                grad_map[node.rhs].append(rhs_grad)
                backward_sinks.append(rhs_grad)

            worklist.append((node.lhs, lhs_grad))
            worklist.append((node.rhs, rhs_grad))

        if node.is_unop:
            (in_node_grad,) = grads
            if isinstance(node.in_node, pg.Value):
                grad_map[node.in_node].append(in_node_grad)
                backward_sinks.append(in_node_grad)
            worklist.append((node.in_node, in_node_grad))

    return backward_sinks, grad_map
