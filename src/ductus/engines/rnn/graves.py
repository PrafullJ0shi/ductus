"""Batched PyTorch sampler for the pretrained Graves (2013) handwriting RNN.

Same network and the same sampling as worker.py (TensorFlow, one line at a time, one
session call per pen point), but:
  * many lines are sampled together, one LSTM step for the whole batch;
  * the style priming (the 400-900 pen points of the chosen writer style) is run once per
    style as one sequence and its end state reused for every line;
  * it runs in the main process, on a CUDA or Apple (MPS) GPU when one is present.

The weights come from the TensorFlow checkpoint, exported once to model-29.npz by
export_weights.py (needs the TensorFlow Python the worker uses; done automatically).
Every line keeps its own numpy RandomState(seed), so a line is sampled the same way as by
the worker for the same seed, up to float rounding.
"""
from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path

import numpy as np

NU = 400          # LSTM units per layer
NW = 10           # window mixtures
NM = 20           # output mixtures


def weights_file(root: Path) -> Path:
    return Path(root) / "pretrained" / "model-29.npz"


def ensure_weights(root: Path, python: str) -> Path:
    f = weights_file(root)
    if not f.exists():
        subprocess.run([python, str(Path(__file__).with_name("export_weights.py")), str(root)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return f


class Graves:
    def __init__(self, root: Path, python: str, device=None):
        import torch
        self.torch = torch
        from ... import speed
        self.speed = speed
        self.dev = device or speed.device()
        w = np.load(ensure_weights(root, python))
        T = lambda k: torch.from_numpy(np.ascontiguousarray(w[k])).float().to(self.dev)
        p = "rnn/rnn_model/"
        self.lk = [T(f"{p}lstm_{i}/lstm_cell/kernel") for i in range(3)]
        self.lb = [T(f"{p}lstm_{i}/lstm_cell/bias") for i in range(3)]
        # window: alpha, beta, kappa stacked into one [400, 30] layer
        self.wk = torch.cat([T(f"{p}window/{n}/kernel") for n in ("alpha", "beta", "kappa")], 1)
        self.wb = torch.cat([T(f"{p}window/{n}/bias") for n in ("alpha", "beta", "kappa")], 0)
        names = ("e", "pi", "mu1", "mu2", "std1", "std2", "rho")
        self.mk = torch.cat([T(f"mixture_output/{n}/kernel") for n in names], 1)   # [400, 121]
        self.mb = torch.cat([T(f"mixture_output/{n}/bias") for n in names], 0)
        with open(Path(root) / "data" / "translation.pkl", "rb") as f:
            self.translation = pickle.load(f)
        with open(Path(root) / "data" / "styles.pkl", "rb") as f:
            self.styles = pickle.load(f, encoding="latin1")
        self.charset = sorted(k for k in self.translation if k != "<NULL>")
        self.nT = len(self.translation)
        self._primed = {}

    # ---- network ---------------------------------------------------------------------
    def _lstm(self, i, x, h, c):
        z = self.torch.cat([x, h], 1) @ self.lk[i] + self.lb[i]
        ig, j, f, o = z.chunk(4, 1)
        c = self.torch.sigmoid(f + 1.0) * c + self.torch.sigmoid(ig) * self.torch.tanh(j)
        return self.torch.sigmoid(o) * self.torch.tanh(c), c

    def _step(self, x, st, seq, u):
        """One pen point for the batch. x [B,3]; st = (h0,c0,h1,c1,h2,c2,window,kappa);
        seq [B,L,nT] one-hot; u [L] positions. Returns new state, phi [B,L], h2."""
        torch = self.torch
        h0, c0, h1, c1, h2, c2, win, kap = st
        h0, c0 = self._lstm(0, torch.cat([x, win], 1), h0, c0)
        abk = torch.exp(h0 @ self.wk + self.wb)
        a, b, k = abk[:, :NW], abk[:, NW:2 * NW], abk[:, 2 * NW:]
        kap = kap + k
        phi = (a[:, :, None] * torch.exp(-((u[None, None, :] - kap[:, :, None]) ** 2) * b[:, :, None])).sum(1)
        win = torch.bmm(phi[:, None, :], seq)[:, 0]
        h1, c1 = self._lstm(1, torch.cat([x, win, h0], 1), h1, c1)
        h2, c2 = self._lstm(2, torch.cat([x, win, h1], 1), h2, c2)
        return (h0, c0, h1, c1, h2, c2, win, kap), phi, h2

    def _zero(self, B):
        z = lambda n: self.torch.zeros(B, n, device=self.dev)
        return (z(NU), z(NU), z(NU), z(NU), z(NU), z(NU), z(self.nT), z(NW))

    def _onehot(self, ids, L):
        o = np.zeros((L, self.nT), np.float32)
        o[np.arange(len(ids)), ids] = 1.0
        return o

    def _prime(self, style):
        """End state after the style's priming strokes (same for every line of that style)."""
        if style in self._primed:
            return self._primed[style]
        torch = self.torch
        sc, stx = self.styles[0][style], self.styles[1][style]
        seq = torch.from_numpy(self._onehot(np.asarray(stx), len(stx) + 1))[None].to(self.dev)
        u = torch.arange(seq.shape[1], dtype=torch.float32, device=self.dev)
        st = self._zero(1)
        prime = torch.from_numpy(np.asarray(sc, np.float32)).to(self.dev)
        with torch.inference_mode():
            for s in range(len(sc) - 1):          # the worker feeds prime[0 .. n-2]
                st, _, _ = self._step(prime[s][None], st, seq, u)
        self._primed[style] = (st, np.asarray(sc[-1], np.float64), list(stx))
        return self._primed[style]

    # ---- sampling --------------------------------------------------------------------
    def lines(self, reqs: list[dict]) -> list[dict]:
        """reqs: [{"text", "bias", "style", "seed"}] -> [{"coords", "ok", "text", "att", "pos"}],
        the worker's reply format. Lines are grouped by style and sampled as one batch."""
        with self.speed.threads(int(os.environ.get("DUCTUS_RNN_THREADS", "2"))):
            return self._lines(reqs)

    def _lines(self, reqs):
        out = [None] * len(reqs)
        groups = {}
        for i, q in enumerate(reqs):
            groups.setdefault(q.get("style"), []).append(i)
        for style, idx in groups.items():
            for i, r in zip(idx, self._batch([reqs[i] for i in idx], style)):
                out[i] = r
        return out

    def _batch(self, reqs, style):
        torch = self.torch
        B = len(reqs)
        T = self.translation
        texts = ["".join(c for c in q["text"] if c in T) for q in reqs]
        biases = np.array([float(q.get("bias", 1.5)) for q in reqs], np.float32)
        rngs = [np.random.RandomState(int(q.get("seed", 0))) for q in reqs]
        pad = "   "
        style_len = 0
        if style is not None:
            st1, last, stx = self._prime(style)
            style_len = len(stx)
            st = tuple(s.expand(B, -1).contiguous() for s in st1)
            full = [stx + [T.get(c, 0) for c in " " + t + pad] for t in texts]
            n_real = [len(t) + 1 for t in texts]
            coord = np.tile(last, (B, 1))
        else:
            st = self._zero(B)
            full = [[T.get(c, 0) for c in t + pad] for t in texts]
            n_real = [len(t) for t in texts]
            coord = np.tile(np.array([0., 0., 1.]), (B, 1))
        Lmax = max(len(f) for f in full) + 1
        seq = torch.from_numpy(np.stack([self._onehot(np.asarray(f), Lmax) for f in full]))
        seq = seq.to(self.dev)
        u = torch.arange(Lmax, dtype=torch.float32, device=self.dev)
        lens = np.array([len(f) for f in full])               # the worker's sequence is len(f)+1 long
        mask = torch.from_numpy((np.arange(Lmax)[None, :] <= lens[:, None]).astype(np.float32)).to(self.dev)
        bias_t = torch.from_numpy(biases).to(self.dev)[:, None]
        first = np.asarray(self.styles[0][style][0], np.float64) if style is not None else np.array([0., 0., 1.])
        coords = [[first.copy()] for _ in range(B)]
        att = [[0] for _ in range(B)]
        pos = [[0.0] for _ in range(B)]
        ok = np.zeros(B, bool)
        # the worker's step budget, which also pays for the priming points
        n_prime = len(self.styles[0][style]) - 1 if style is not None else 0
        limit = [60 * (len(t) + len(pad) + (1 if style is not None else 0) + style_len) - n_prime for t in texts]
        steps = 0
        ar = np.arange(NM)
        act = np.arange(B)                 # rows still writing; finished rows leave the batch
        with torch.inference_mode():
            while len(act):
                steps += 1
                x = torch.from_numpy(coord[act].astype(np.float32)).to(self.dev)
                st, phi, h2 = self._step(x, st, seq, u)
                m = h2 @ self.mk + self.mb
                e = torch.sigmoid(m[:, :1])
                pi = torch.softmax(m[:, 1:1 + NM] * (1. + bias_t), 1)
                mu1, mu2 = m[:, 1 + NM:1 + 2 * NM], m[:, 1 + 2 * NM:1 + 3 * NM]
                s1 = torch.exp(m[:, 1 + 3 * NM:1 + 4 * NM] - bias_t)
                s2 = torch.exp(m[:, 1 + 4 * NM:1 + 5 * NM] - bias_t)
                rho = torch.tanh(m[:, 1 + 5 * NM:1 + 6 * NM])
                ph = (phi * mask).double().cpu().numpy()
                e, pi, mu1, mu2, s1, s2, rho = (t.double().cpu().numpy() for t in (e, pi, mu1, mu2, s1, s2, rho))
                keep = []
                for r, b in enumerate(act):
                    L = lens[b] + 1
                    pb = ph[r, :L]
                    # finish: the window's last position outweighs every earlier one
                    fin = float(pb[-1] > pb[:-1].max())
                    g = rngs[b].choice(ar, p=pi[r] / pi[r].sum())
                    c = s1[r, g] * s2[r, g] * rho[r, g]
                    xx, yy = rngs[b].multivariate_normal([mu1[r, g], mu2[r, g]], [[s1[r, g] ** 2, c], [c, s2[r, g] ** 2]])
                    nc = np.array([xx, yy, rngs[b].binomial(1, e[r, 0])])
                    am = int(np.argmax(pb))
                    if am >= style_len + n_real[b] + 1:
                        if len(coords[b]) > 1:
                            coords[b][-1][2] = 1.
                        ok[b] = True
                        continue
                    coords[b].append(nc)
                    att[b].append(am)
                    pos[b].append(float((pb * np.arange(L)).sum() / max(pb.sum(), 1e-9)))
                    coord[b] = nc
                    if fin > 0.8:
                        ok[b] = True
                    elif steps < limit[b]:
                        keep.append(r)
                if len(keep) < len(act):
                    if not keep:
                        break
                    k = torch.tensor(keep, device=self.dev)
                    st = tuple(t.index_select(0, k) for t in st)
                    seq, mask, bias_t = seq.index_select(0, k), mask.index_select(0, k), bias_t.index_select(0, k)
                    act = act[keep]
        res = []
        for b in range(B):
            res.append(_finish(np.array(coords[b]), np.array(att[b]), pos[b], style_len, n_real[b], style, texts[b], bool(ok[b])))
        return res


def _finish(coords, att, pos, style_len, n_real, style, text, ok):
    """The worker's tail clean-up: drop stray marks drawn while attending the padding."""
    coords[-1, 2] = 1.
    while True:
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
    off = style_len + (1 if style is not None else 0)
    return {"coords": np.round(coords, 4).tolist(), "ok": ok, "text": text,
            "att": (np.asarray(att) - off).tolist(), "pos": (np.array(pos) - off).round(3).tolist()}
