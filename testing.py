import peachygrad_cc as pg_cc


def allclose(ref: pg_cc.Tensor, x: pg_cc.Tensor, atol=1e-5, rtol=1e-5) -> bool:
    return pg_cc.testing.allclose(ref, x, atol, rtol)
