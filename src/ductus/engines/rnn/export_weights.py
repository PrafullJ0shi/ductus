"""Export the pretrained Graves RNN checkpoint to a plain .npz, so the batched PyTorch sampler
(graves.py) can run it without TensorFlow. Runs once, in the TensorFlow Python:

  python export_weights.py MODEL_DIR        -> MODEL_DIR/pretrained/model-29.npz
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main(root):
    import numpy as np
    import tensorflow as tf
    path = os.path.join(root, "pretrained", "model-29")
    out = {}
    for name, _ in tf.train.list_variables(path):
        if name.startswith("model/") and "Adam" not in name and ("lstm_cell" in name or "/window/" in name or "mixture_output" in name):
            out[name[len("model/"):]] = tf.train.load_variable(path, name)
    tmp = path + ".npz.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path + ".npz")
    print(path + ".npz")


if __name__ == "__main__":
    main(sys.argv[1])
