import peachygrad as pg
import peachygrad_cc as pg_cc
import numpy as np

from pathlib import Path

LR = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_TEST_BATCHES = 50
NUM_TRAIN_BATCHES = 60000 // BATCH_SIZE


def normalize_images(data: np.ndarray, n: int):
    return np.reshape(data.astype(dtype=np.float32) / 255.0, (n, 784))


def normalize_labels(data: np.ndarray, n: int):
    one_hot = np.zeros((n, 10), dtype=np.float32)
    one_hot[range(n), data] = 1.0
    return one_hot


def main():
    mnist_path = Path(".", "mnist", "mnist.npz").absolute()
    with np.load(str(mnist_path)) as data:
        x_train_np = normalize_images(data["x_train"], 60000)
        y_train_np = normalize_labels(data["y_train"], 60000)
        x_test_np = normalize_images(data["x_test"], 10000)
        y_test_np = normalize_labels(data["y_test"], 10000)

    def build_grad_map(ag_map, *nodes):
        grad_map = {}
        for node in nodes:
            partials = ag_map[node]
            grad_accum_graph = None
            for partial in partials:
                if grad_accum_graph is None:
                    grad_accum_graph = partial
                    continue

                grad_accum_graph = grad_accum_graph + partial

            grad_map[node] = grad_accum_graph
        return grad_map

    def apply_gradients(grad_map):
        for node, grad in grad_map.items():
            grad.reset()
            new_node = node - (pg.constant(grad.shape, LR) * grad)
            new_node.eval()
            node.out_tensor = new_node.out_tensor.copy()

    kaiming_w0 = (6 / 784) ** 0.5
    kaiming_w1 = (6 / 64) ** 0.5

    x = pg.zeros((BATCH_SIZE, 784))
    w0 = pg.uniform((784, 64), lo=-kaiming_w0, hi=kaiming_w0)
    w1 = pg.uniform((64, 10), lo=-kaiming_w1, hi=kaiming_w1)
    h0 = x @ w0
    logits = pg.relu(h0 @ w1)

    # softmax.
    _, l = logits.shape[0], logits.shape[1]
    ex = pg.exp(logits)
    ex_sum = pg.reduce_sum(ex, axis=1)
    ex_sum_bcast = pg.broadcast(ex_sum, axis=1, size=l)
    probs = ex / ex_sum_bcast
    log_probs = pg.log(probs)

    # loss.
    labels = pg.zeros((BATCH_SIZE, 10))
    loss = pg.reduce_mean(-pg.reduce_sum(log_probs * labels, axis=1), axis=0)

    _, ag_map = pg.autograd.backward(loss)
    grad_map = build_grad_map(ag_map, w0, w1)

    for epoch in range(NUM_EPOCHS):
        for batch_id in range(NUM_TRAIN_BATCHES):
            x_batch_np = x_train_np[
                batch_id * BATCH_SIZE : (batch_id + 1) * BATCH_SIZE, ...
            ]
            y_batch_np = y_train_np[
                batch_id * BATCH_SIZE : (batch_id + 1) * BATCH_SIZE, ...
            ]
            x_batch = pg_cc.tensor(x_batch_np)
            y_batch = pg_cc.tensor(y_batch_np)

            x.out_tensor = x_batch
            labels.out_tensor = y_batch
            loss.reset()
            loss.eval()

            # backward passes.
            apply_gradients(grad_map)

        num_correct = 0
        num_total = 0
        for test_batch_id in range(NUM_TEST_BATCHES):
            x_batch_np = x_test_np[
                test_batch_id * BATCH_SIZE : (test_batch_id + 1) * BATCH_SIZE, ...
            ]
            y_batch_np = y_test_np[
                test_batch_id * BATCH_SIZE : (test_batch_id + 1) * BATCH_SIZE, ...
            ]
            x_batch = pg_cc.tensor(x_batch_np)
            x.out_tensor = x_batch
            logits.reset()
            logits.eval()
            logits_np = pg_cc.numpy(logits.out_tensor)
            logits_argmax = np.argmax(logits_np, axis=1)
            y_argmax = np.argmax(y_batch_np, axis=1)
            num_correct += np.sum(np.equal(logits_argmax, y_argmax))
            num_total += BATCH_SIZE

        print(f"Epoch {epoch}, Loss: {loss}. Num Correct: {num_correct} of {num_total}")


if __name__ == "__main__":
    main()
