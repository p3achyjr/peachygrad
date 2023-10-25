import peachygrad as pg


def categorical_crossentropy(logits: pg.Node, labels: pg.Node):
    """
    logits: (B, L)
    labels: (B, L)
    """
    probs = pg.softmax(logits)
    log_probs = pg.log(probs)
    return pg.reduce_mean(pg.reduce_sum(-log_probs * labels, axis=1), axis=0)
