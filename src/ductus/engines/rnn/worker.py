"""Stroke sampler for the Graves (2013) handwriting RNN, pretrained by Grzegorz Opoka
(github.com/Grzego/handwriting-generation, MIT). Runs as a child process so TensorFlow stays
out of the main ductus environment.

  python worker.py MODEL_DIR        (MODEL_DIR holds data/ and pretrained/model-29*)
Reads one JSON request per line on stdin: {"text", "bias", "style", "seed"}; writes one JSON
line per request: {"coords": [[dx, dy, end], ...], "ok": bool}.
"""
import json, os, pickle, sys, warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402


class Model:
    def __init__(self, root):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.logging.set_verbosity(tf.logging.ERROR)
        with open(os.path.join(root, "data", "translation.pkl"), "rb") as f:
            self.translation = pickle.load(f)
        with open(os.path.join(root, "data", "styles.pkl"), "rb") as f:
            self.styles = pickle.load(f, encoding="latin1")
        self.charset = sorted(k for k in self.translation if k != "<NULL>")
        n = int(os.environ.get("DUCTUS_RNN_THREADS", "2"))
        self.sess = tf.Session(config=tf.ConfigProto(device_count={"GPU": 0}, intra_op_parallelism_threads=n,
                                                     inter_op_parallelism_threads=1))
        path = os.path.join(root, "pretrained", "model-29")
        tf.train.import_meta_graph(path + ".meta").restore(self.sess, path)
        fields = ["coordinates", "sequence", "bias", "e", "pi", "mu1", "mu2", "std1", "std2",
                  "rho", "finish", "phi", "zero_states"]
        self.v = {k: tf.get_collection(k)[0] for k in fields}

    def line(self, text, bias, style, rng, pad="   "):
        if style is not None:
            text = " " + text
        n_real = len(text)
        text = text + pad
        v, T = self.v, self.translation
        nT = len(T)
        ids = np.array([T.get(c, 0) for c in text])
        coord = np.array([0., 0., 1.])
        coords, att, pos = [coord], [0], [0.0]
        prime_len = style_len = 0
        seq_prime = prime = None
        if style is not None:
            sc, st = self.styles[0][style], self.styles[1][style]
            prime_len, style_len = len(sc), len(st)
            prime = list(sc)
            coord = prime[0]
            ids = np.r_[st, ids]
            seq_prime = np.concatenate([np.eye(nT, dtype=np.float32)[st], np.zeros((1, nT))])[None]
        seq = np.concatenate([np.eye(nT, dtype=np.float32)[ids], np.zeros((1, nT))])[None]
        self.sess.run(v["zero_states"])
        ok = False
        for s in range(1, 60 * (len(text) + style_len) + 1):
            priming = s < prime_len
            e, pi, mu1, mu2, s1, s2, rho, fin, phi = self.sess.run(
                [v["e"], v["pi"], v["mu1"], v["mu2"], v["std1"], v["std2"], v["rho"], v["finish"], v["phi"]],
                feed_dict={v["coordinates"]: coord[None, None, ...],
                           v["sequence"]: seq_prime if priming else seq, v["bias"]: bias})
            if priming:
                coord = prime[s]
                continue
            g = rng.choice(np.arange(pi.shape[1]), p=pi[0] / pi[0].sum())
            c = s1[0, g] * s2[0, g] * rho[0, g]
            x, y = rng.multivariate_normal([mu1[0, g], mu2[0, g]], [[s1[0, g] ** 2, c], [c, s2[0, g] ** 2]])
            coord = np.array([x, y, rng.binomial(1, e[0, 0])])
            if np.argmax(phi[0]) >= style_len + n_real + 1:
                if len(coords) > 1:
                    coords[-1][2] = 1.
                ok = True
                break
            coords.append(coord)
            att.append(int(np.argmax(phi[0])))
            ph = phi[0].ravel(); pos.append(float((ph * np.arange(len(ph))).sum() / max(ph.sum(), 1e-9)))
            if fin[0, 0] > 0.8:
                ok = True
                break
        coords = np.array(coords)
        coords[-1, 2] = 1.
        att = np.array(att)
        while True:     # drop stray marks drawn while attending the padding
            ends = np.where(coords[:-1, 2] == 1.)[0]
            if len(ends) < 2:
                break
            start = ends[-1] + 1
            absp = np.cumsum(coords[:, :2], axis=0)
            absx = absp[start:]
            extent = np.ptp(absx, axis=0).max() if len(absx) > 1 else 0.
            right = absx[:, 0].min() > absp[1:start, 0].max() + 0.3
            if att[start:].min() >= style_len + n_real and extent < 1.2 and right:
                coords, att, pos = coords[:start], att[:start], pos[:start]
                coords[-1, 2] = 1.
            else:
                break
        off = style_len + (1 if style is not None else 0)   # attention index -> index in the caller's text
        return coords, ok, (att - off).tolist(), (np.array(pos) - off).round(3).tolist()


def main():
    m = Model(sys.argv[1])
    print(json.dumps({"ready": True, "charset": m.charset}), flush=True)
    for raw in sys.stdin:
        if not raw.strip():
            continue
        q = json.loads(raw)
        text = "".join(c for c in q["text"] if c in m.translation)
        coords, ok, att, pos = m.line(text, float(q.get("bias", 1.5)), q.get("style"), np.random.RandomState(int(q.get("seed", 0))))
        print(json.dumps({"coords": np.round(coords, 4).tolist(), "ok": bool(ok), "text": text, "att": att, "pos": pos}), flush=True)


if __name__ == "__main__":
    main()
