"""
pointerbot_configurable.py -- your pointerbot, same teach/learn/:stabilize
workflow, same copy/pointer mechanism, same BPE tokenizer -- but the
transformer's internal architecture is now a CONFIG block instead of a
fixed design. Flip each flag on/off and the model that gets built
changes shape accordingly:

    use_rope     -- rotary position embeddings (RoFormer) instead of a
                    learned absolute position table
    use_rmsnorm  -- RMSNorm (LLaMA/Qwen/DeepSeek-style) instead of
                    LayerNorm
    use_mla      -- Multi-Head Latent Attention (DeepSeek-V2-style):
                    compress K/V through a shared low-rank bottleneck
                    (width d_latent) instead of full-size per-head K/V
    n_experts    -- Mixture-of-Experts feedforward with a shared expert
                    (DeepSeek-V2/V3-style). n_experts=1 approximates a
                    plain dense FFN (shared expert + one always-chosen
                    routed expert -- see MoEFeedForward docstring for the
                    exact tradeoff of leaving it at 1 vs truly going
                    dense); n_experts>1 gives real sparse-style routing.
    use_mtp      -- an auxiliary head trained to predict token t+2
                    alongside the usual t+1 (DeepSeek-V3/Meta-style
                    Multi-Token Prediction), weighted by mtp_weight
    use_pointer  -- the original pointerbot copy/pointer mechanism
                    (See et al. 2017-style): blend the vocabulary
                    softmax with a distribution that copies straight
                    from the input, gated per-position. This is
                    orthogonal to all the flags above -- it sits on top
                    of whatever backbone you configure.

Every combination of these flags produces a real, running, trainable
model -- this file doesn't fake any of them. What none of these flags
change is DATA or COMPUTE: turning more of them on gives the model more
*mechanisms* it could in principle use, not more *knowledge*. It still
only knows whatever you teach it.

Run:
    python3 pointerbot_configurable.py
Requires:
    pkg install python-numpy   (Termux)  /  pip install numpy   (elsewhere)
"""
import os
import re
import json
import time
import random
import collections
import numpy as np


# ======================================================================
# ---- Tensor: your original autograd core, plus rmsnorm + sigmoid ----
# ======================================================================
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad,
                      _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.data * other, requires_grad=self.requires_grad,
                          _children=(self,), _op="mulc")

            def _backward():
                if self.requires_grad:
                    self.grad += out.grad * other
            out._backward = _backward
            return out

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def matmul(self, other):
        out = Tensor(self.data @ other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            if other.requires_grad:
                other.grad += np.swapaxes(self.data, -1, -2) @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self, axes):
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                      _children=(self,), _op="transpose")
        inv = tuple(np.argsort(axes))

        def _backward():
            if self.requires_grad:
                self.grad += np.transpose(out.grad, inv)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                      _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(orig_shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                      requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad,
                      _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(x)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                dy = out.grad
                dot = np.sum(dy * s, axis=axis, keepdims=True)
                self.grad += s * (dy - dot)
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * std_inv
        y = xhat * gamma.data + beta.data
        out = Tensor(y, requires_grad=True, _children=(self, gamma, beta), _op="layernorm")
        D = x.shape[-1]

        def _backward():
            dy = out.grad
            if gamma.requires_grad:
                gamma.grad += np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
            if beta.requires_grad:
                beta.grad += np.sum(dy, axis=tuple(range(dy.ndim - 1)))
            if self.requires_grad:
                dxhat = dy * gamma.data
                dvar = np.sum(dxhat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
                dmu = (np.sum(dxhat * -std_inv, axis=-1, keepdims=True) +
                       dvar * np.mean(-2 * (x - mu), axis=-1, keepdims=True))
                dx = dxhat * std_inv + dvar * 2 * (x - mu) / D + dmu / D
                self.grad += dx
        out._backward = _backward
        return out

    def rmsnorm(self, weight, eps=1e-6):
        x = self.data
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(ms + eps)
        xhat = x * inv
        y = xhat * weight.data
        out = Tensor(y, requires_grad=True, _children=(self, weight), _op="rmsnorm")
        D = x.shape[-1]

        def _backward():
            dy = out.grad
            if weight.requires_grad:
                weight.grad += np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
            if self.requires_grad:
                dxhat = dy * weight.data
                dms = np.sum(dxhat * x * -0.5 * inv ** 3, axis=-1, keepdims=True)
                dx = dxhat * inv + dms * 2 * x / D
                self.grad += dx
        out._backward = _backward
        return out

    def embedding_lookup(self, ids):
        out_data = self.data[ids]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="embed")

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, ids, out.grad)
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


def softmax_cross_entropy(logits, target_ids):
    x = logits.data
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    T = x.shape[0]
    logp = -np.log(probs[np.arange(T), target_ids] + 1e-12)
    loss_val = np.mean(logp)
    out = Tensor(loss_val, requires_grad=True, _children=(logits,), _op="softmax_ce")

    def _backward():
        grad = probs.copy()
        grad[np.arange(T), target_ids] -= 1
        grad /= T
        logits.grad += grad * out.grad
    out._backward = _backward
    return out


def nll_from_probs(probs: Tensor, target_ids) -> Tensor:
    T = len(target_ids)
    target_ids = np.array(target_ids)
    p_target = probs.data[np.arange(T), target_ids]
    loss_val = np.mean(-np.log(p_target + 1e-9))

    out = Tensor(loss_val, requires_grad=True, _children=(probs,), _op="nll")

    def _backward():
        grad = np.zeros_like(probs.data)
        grad[np.arange(T), target_ids] = (-1.0 / (p_target + 1e-9)) / T
        probs.grad += grad * out.grad
    out._backward = _backward
    return out


def silu(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def _param(shape, scale=0.02):
    return Tensor(np.random.randn(*shape) * scale, requires_grad=True)


# ======================================================================
# ---- RoPE ----
# ======================================================================
def build_rope_cache(T, d_head, base=10000.0):
    assert d_head % 2 == 0
    inv_freq = 1.0 / (base ** (np.arange(0, d_head, 2) / d_head))
    pos = np.arange(T)
    freqs = np.outer(pos, inv_freq)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    return Tensor(np.cos(freqs)), Tensor(np.sin(freqs))


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.data
    Dh = d.shape[-1]
    half = Dh // 2
    x1, x2 = d[..., :half], d[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    out_data = d * cos.data + rot * sin.data
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="rope")

    def _backward():
        if x.requires_grad:
            dout = out.grad
            grad_direct = dout * cos.data
            dg = dout * sin.data
            dg1, dg2 = dg[..., :half], dg[..., half:]
            grad_x1 = grad_direct[..., :half] + dg2
            grad_x2 = grad_direct[..., half:] - dg1
            x.grad += np.concatenate([grad_x1, grad_x2], axis=-1)
    out._backward = _backward
    return out


# ======================================================================
# ---- tokenizer.py -- unchanged from your version ----
# ======================================================================
class BPETokenizer:
    SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self):
        printable = [chr(i) for i in range(32, 127)]
        extra = ["\n", "\t"]
        self.base_chars = sorted(set(printable + extra))
        self.merges = []
        self._build_vocab()

    def _build_vocab(self):
        tokens = list(self.SPECIALS) + list(self.base_chars)
        for a, b in self.merges:
            tokens.append(a + b)
        self.stoi = {}
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = len(self.stoi)
        self.itos = {i: t for t, i in self.stoi.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def eos_id(self):
        return self.stoi["<EOS>"]

    @property
    def bos_id(self):
        return self.stoi["<BOS>"]

    @staticmethod
    def _pretokenize(text):
        return re.findall(r"\S+|\s+", text)

    def train(self, text, vocab_size=600, min_pair_freq=2, verbose=True):
        words = self._pretokenize(text)
        word_freq = collections.Counter(words)
        vocab = {tuple(w): f for w, f in word_freq.items()}

        n_base = len(self.SPECIALS) + len(self.base_chars)
        target_merges = max(0, vocab_size - n_base)

        merges = []
        for i in range(target_merges):
            pairs = collections.defaultdict(int)
            for word, freq in vocab.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j + 1])] += freq
            if not pairs:
                break
            best_pair, best_freq = max(pairs.items(), key=lambda kv: kv[1])
            if best_freq < min_pair_freq:
                break
            merges.append(best_pair)

            a, b = best_pair
            merged = a + b
            new_vocab = {}
            for word, freq in vocab.items():
                w = list(word)
                out, k = [], 0
                while k < len(w):
                    if k < len(w) - 1 and w[k] == a and w[k + 1] == b:
                        out.append(merged)
                        k += 2
                    else:
                        out.append(w[k])
                        k += 1
                key = tuple(out)
                new_vocab[key] = new_vocab.get(key, 0) + freq
            vocab = new_vocab

            if verbose and (i + 1) % 50 == 0:
                print(f"  [bpe merge {i+1}/{target_merges}: {a!r}+{b!r} -> "
                      f"{merged!r} (freq={best_freq})]")

        self.merges = merges
        self._build_vocab()
        if verbose:
            print(f"[bpe trained: {len(self.base_chars)} base chars + "
                  f"{len(self.merges)} merges = {self.vocab_size} tokens total]")

    def _encode_word(self, word):
        symbols = list(word)
        if len(symbols) == 1 or not self.merge_ranks:
            return symbols
        while True:
            best_rank, best_i = None, None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i is None:
                break
            symbols = (symbols[:best_i] + [symbols[best_i] + symbols[best_i + 1]]
                       + symbols[best_i + 2:])
        return symbols

    def encode(self, text, add_bos=True, add_eos=False):
        ids = [self.stoi["<BOS>"]] if add_bos else []
        unk = self.stoi["<UNK>"]
        for word in self._pretokenize(text):
            for piece in self._encode_word(word):
                if piece in self.stoi:
                    ids.append(self.stoi[piece])
                else:
                    for ch in piece:
                        ids.append(self.stoi.get(ch, unk))
        if add_eos:
            ids.append(self.stoi["<EOS>"])
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos.get(int(i), "")
            if t in ("<BOS>", "<PAD>"):
                continue
            if t == "<EOS>":
                break
            out.append(t)
        return "".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"base_chars": self.base_chars,
                        "merges": [list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.base_chars = data["base_chars"]
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._build_vocab()
        return tok


# ======================================================================
# ---- feedforward: SwiGLU building block, used by every expert ----
# ======================================================================
class SwiGLUFeedForward:
    def __init__(self, d_model, d_ff):
        self.W_gate = _param((d_model, d_ff))
        self.W_up = _param((d_model, d_ff))
        self.W_down = _param((d_ff, d_model))

    def params(self):
        return [self.W_gate, self.W_up, self.W_down]

    def __call__(self, x):
        gate = silu(x.matmul(self.W_gate))
        up = x.matmul(self.W_up)
        return (gate * up).matmul(self.W_down)


# ======================================================================
# ---- Mixture-of-Experts feedforward, with a shared expert ----
# ======================================================================
class MoEFeedForward:
    """n_experts=1 is a valid config, but note what it actually costs:
    every token still pays for the shared expert AND the one routed
    expert (since with only one choice, routing is trivial but still
    happens) -- roughly 2x a single dense FFN's compute, not 1x. If you
    want a truly plain dense FFN with no MoE overhead at all, that's a
    trivial edit (just call self.shared(x) and skip the router/experts
    loop entirely) -- left as MoE-with-1-expert here so every config
    knob in CONFIG stays meaningful and comparable."""
    def __init__(self, d_model, d_ff_expert, n_experts):
        self.n_experts = n_experts
        self.shared = SwiGLUFeedForward(d_model, d_ff_expert)
        self.experts = [SwiGLUFeedForward(d_model, d_ff_expert) for _ in range(n_experts)]
        self.W_router = _param((d_model, n_experts))

    def params(self):
        ps = list(self.shared.params()) + [self.W_router]
        for e in self.experts:
            ps += e.params()
        return ps

    def __call__(self, x):
        T, D = x.shape
        shared_out = self.shared(x)

        router_logits = x.matmul(self.W_router)
        router_probs = router_logits.softmax(axis=-1)
        chosen = np.argmax(router_probs.data, axis=-1)

        expert_outs = [e(x) for e in self.experts]
        combined = Tensor(np.zeros((T, D)), requires_grad=True,
                           _children=tuple(expert_outs) + (router_probs,))
        gate_weight = router_probs.data[np.arange(T), chosen]

        out_data = np.zeros((T, D))
        for e_idx in range(self.n_experts):
            mask = (chosen == e_idx)
            if not mask.any():
                continue
            out_data[mask] = expert_outs[e_idx].data[mask] * gate_weight[mask, None]
        combined.data = out_data

        def _backward():
            dout = combined.grad
            for e_idx, eo in enumerate(expert_outs):
                if eo.requires_grad:
                    mask = (chosen == e_idx)
                    g = np.zeros_like(eo.data)
                    g[mask] = dout[mask] * gate_weight[mask, None]
                    eo.grad += g
            if router_probs.requires_grad:
                g = np.zeros_like(router_probs.data)
                contrib = np.sum(dout * np.stack([eo.data for eo in expert_outs], axis=1)[
                    np.arange(T), chosen], axis=-1)
                g[np.arange(T), chosen] = contrib
                router_probs.grad += g
        combined._backward = _backward

        return shared_out + combined


# ======================================================================
# ---- Attention: plain full-size, or MLA (compressed K/V) ----
# ======================================================================
class MultiHeadAttention:
    """Plain attention, full-size per-head K/V. Used when use_mla=False."""
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.Wq, self.bq = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.Wk, self.bk = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.Wv, self.bv = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.Wo, self.bo = _param((d_model, d_model)), _param((d_model,), 0.0)

    def params(self):
        return [self.Wq, self.bq, self.Wk, self.bk, self.Wv, self.bv, self.Wo, self.bo]

    def __call__(self, x, causal_mask, rope):
        T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = x.matmul(self.Wq) + self.bq
        k = x.matmul(self.Wk) + self.bk
        v = x.matmul(self.Wv) + self.bv
        q = q.reshape(T, H, Dh).transpose((1, 0, 2))
        k = k.reshape(T, H, Dh).transpose((1, 0, 2))
        v = v.reshape(T, H, Dh).transpose((1, 0, 2))
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        kt = k.transpose((0, 2, 1))
        scores = q.matmul(kt) * (1.0 / np.sqrt(Dh))
        scores = scores + causal_mask
        attn = scores.softmax(axis=-1)
        out = attn.matmul(v)
        out = out.transpose((1, 0, 2)).reshape(T, D)
        return out.matmul(self.Wo) + self.bo


class MultiHeadLatentAttention:
    """Compressed K/V via a shared low-rank latent. Used when use_mla=True.
    See advanced_architecture_reference.py's version of this class for the
    fuller explanation of what's simplified vs. the DeepSeek-V2 paper."""
    def __init__(self, d_model, n_heads, d_latent):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.Wq, self.bq = _param((d_model, d_model)), _param((d_model,), 0.0)
        self.W_down_kv = _param((d_model, d_latent))
        self.W_up_k = _param((d_latent, d_model))
        self.W_up_v = _param((d_latent, d_model))
        self.Wo, self.bo = _param((d_model, d_model)), _param((d_model,), 0.0)

    def params(self):
        return [self.Wq, self.bq, self.W_down_kv, self.W_up_k, self.W_up_v, self.Wo, self.bo]

    def __call__(self, x, causal_mask, rope):
        T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = x.matmul(self.Wq) + self.bq
        c_kv = x.matmul(self.W_down_kv)
        k = c_kv.matmul(self.W_up_k)
        v = c_kv.matmul(self.W_up_v)
        q = q.reshape(T, H, Dh).transpose((1, 0, 2))
        k = k.reshape(T, H, Dh).transpose((1, 0, 2))
        v = v.reshape(T, H, Dh).transpose((1, 0, 2))
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        kt = k.transpose((0, 2, 1))
        scores = q.matmul(kt) * (1.0 / np.sqrt(Dh))
        scores = scores + causal_mask
        attn = scores.softmax(axis=-1)
        out = attn.matmul(v)
        out = out.transpose((1, 0, 2)).reshape(T, D)
        return out.matmul(self.Wo) + self.bo


# ======================================================================
# ---- TransformerBlock: assembled from whatever CONFIG picks ----
# ======================================================================
class TransformerBlock:
    def __init__(self, cfg):
        d = cfg["d_model"]
        self.use_rmsnorm = cfg["use_rmsnorm"]
        if self.use_rmsnorm:
            self.norm1_w = Tensor(np.ones(d), requires_grad=True)
            self.norm2_w = Tensor(np.ones(d), requires_grad=True)
        else:
            self.norm1_g = Tensor(np.ones(d), requires_grad=True)
            self.norm1_b = Tensor(np.zeros(d), requires_grad=True)
            self.norm2_g = Tensor(np.ones(d), requires_grad=True)
            self.norm2_b = Tensor(np.zeros(d), requires_grad=True)

        if cfg["use_mla"]:
            self.attn = MultiHeadLatentAttention(d, cfg["n_heads"], cfg["d_latent"])
        else:
            self.attn = MultiHeadAttention(d, cfg["n_heads"])

        self.ffn = MoEFeedForward(d, cfg["d_ff_expert"], cfg["n_experts"])

    def params(self):
        if self.use_rmsnorm:
            ps = [self.norm1_w, self.norm2_w]
        else:
            ps = [self.norm1_g, self.norm1_b, self.norm2_g, self.norm2_b]
        return ps + self.attn.params() + self.ffn.params()

    def _norm(self, x, which):
        if self.use_rmsnorm:
            w = self.norm1_w if which == 1 else self.norm2_w
            return x.rmsnorm(w)
        g, b = (self.norm1_g, self.norm1_b) if which == 1 else (self.norm2_g, self.norm2_b)
        return x.layernorm(g, b)

    def __call__(self, x, causal_mask, rope):
        x = x + self.attn(self._norm(x, 1), causal_mask, rope)
        x = x + self.ffn(self._norm(x, 2))
        return x


# ======================================================================
# ---- ConfigurableGPT: the backbone + pointer/copy head + MTP head ----
# ======================================================================
class ConfigurableGPT:
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3,
                 d_latent=16, d_ff_expert=96, n_experts=4, max_seq_len=256,
                 use_rope=True, use_rmsnorm=True, use_mla=True,
                 use_pointer=True, use_mtp=True, mtp_weight=0.3, seed=0):
        np.random.seed(seed)
        self.cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                         d_latent=d_latent, d_ff_expert=d_ff_expert, n_experts=n_experts,
                         max_seq_len=max_seq_len, use_rope=use_rope, use_rmsnorm=use_rmsnorm,
                         use_mla=use_mla, use_pointer=use_pointer, use_mtp=use_mtp,
                         mtp_weight=mtp_weight)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_pointer = use_pointer
        self.use_mtp = use_mtp
        self.mtp_weight = mtp_weight

        self.token_emb = _param((vocab_size, d_model))
        if not use_rope:
            self.pos_emb = _param((max_seq_len, d_model))

        self.blocks = [TransformerBlock(self.cfg) for _ in range(n_layers)]

        if use_rmsnorm:
            self.lnf_w = Tensor(np.ones(d_model), requires_grad=True)
        else:
            self.lnf_g = Tensor(np.ones(d_model), requires_grad=True)
            self.lnf_b = Tensor(np.zeros(d_model), requires_grad=True)

        if use_pointer:
            self.Wq_ptr = _param((d_model, d_model))
            self.Wk_ptr = _param((d_model, d_model))
            self.Wg = _param((d_model, 1))
            self.bg = Tensor(np.zeros(1), requires_grad=True)

        if use_mtp:
            self.mtp_block = TransformerBlock(self.cfg)
            if use_rmsnorm:
                self.mtp_norm_w = Tensor(np.ones(d_model), requires_grad=True)
            else:
                self.mtp_norm_g = Tensor(np.ones(d_model), requires_grad=True)
                self.mtp_norm_b = Tensor(np.zeros(d_model), requires_grad=True)

        self._mask_cache = {}
        self._rope_cache = {}

    # ---------------------------- params ----------------------------
    def named_params(self):
        names, ps = [], []
        def add(n, p):
            names.append(n); ps.append(p)

        add("token_emb", self.token_emb)
        if not self.use_rope:
            add("pos_emb", self.pos_emb)
        if self.use_rmsnorm:
            add("lnf_w", self.lnf_w)
        else:
            add("lnf_g", self.lnf_g); add("lnf_b", self.lnf_b)
        if self.use_pointer:
            add("Wq_ptr", self.Wq_ptr); add("Wk_ptr", self.Wk_ptr)
            add("Wg", self.Wg); add("bg", self.bg)
        for i, b in enumerate(self.blocks):
            for j, p in enumerate(b.params()):
                add(f"block{i}_p{j}", p)
        if self.use_mtp:
            for j, p in enumerate(self.mtp_block.params()):
                add(f"mtp_p{j}", p)
            if self.use_rmsnorm:
                add("mtp_norm_w", self.mtp_norm_w)
            else:
                add("mtp_norm_g", self.mtp_norm_g); add("mtp_norm_b", self.mtp_norm_b)
        return names, ps

    def params(self):
        return self.named_params()[1]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def _causal_mask(self, T):
        if T not in self._mask_cache:
            m = np.triu(np.ones((T, T)) * -1e9, k=1)
            self._mask_cache[T] = Tensor(m, requires_grad=False)
        return self._mask_cache[T]

    def _rope(self, T):
        if not self.use_rope:
            return None
        if T not in self._rope_cache:
            self._rope_cache[T] = build_rope_cache(T, self.d_model // self.n_heads)
        return self._rope_cache[T]

    def _final_norm(self, x):
        if self.use_rmsnorm:
            return x.rmsnorm(self.lnf_w)
        return x.layernorm(self.lnf_g, self.lnf_b)

    def _mtp_norm(self, x):
        if self.use_rmsnorm:
            return x.rmsnorm(self.mtp_norm_w)
        return x.layernorm(self.mtp_norm_g, self.mtp_norm_b)

    # ---------------------------- forward ----------------------------
    def forward_probs(self, ids, compute_mtp=None):
        """Returns (final_prob, p_gen, mtp_logits_full_or_None).
        final_prob: (T, V) already-normalized probabilities (pointer-blended
        if use_pointer, else plain softmax). compute_mtp defaults to
        self.use_mtp; pass False during generation to skip the extra
        compute of a head you're not training right now."""
        if compute_mtp is None:
            compute_mtp = self.use_mtp

        T = len(ids)
        ids_arr = np.array(ids)
        x = self.token_emb.embedding_lookup(ids_arr)
        if not self.use_rope:
            x = x + self.pos_emb.embedding_lookup(np.arange(T))
        mask = self._causal_mask(T)
        rope = self._rope(T)

        for block in self.blocks:
            x = block(x, mask, rope)
        h = self._final_norm(x)

        gen_logits = h.matmul(self.token_emb.transpose((1, 0)))
        gen_prob = gen_logits.softmax(axis=-1)

        if self.use_pointer:
            q_ptr = h.matmul(self.Wq_ptr)
            k_ptr = h.matmul(self.Wk_ptr)
            scores_ptr = q_ptr.matmul(k_ptr.transpose((1, 0))) * (1.0 / np.sqrt(self.d_model))
            scores_ptr = scores_ptr + mask
            attn_ptr = scores_ptr.softmax(axis=-1)
            onehot = Tensor(np.eye(self.vocab_size)[ids_arr], requires_grad=False)
            copy_prob = attn_ptr.matmul(onehot)
            p_gen = (h.matmul(self.Wg) + self.bg).sigmoid()
            final_prob = gen_prob * p_gen + copy_prob * (p_gen * -1.0 + 1.0)
        else:
            final_prob, p_gen = gen_prob, None

        mtp_logits_full = None
        if compute_mtp and self.use_mtp:
            mtp_x = self.mtp_block(x, mask, rope)
            mtp_h = self._mtp_norm(mtp_x)
            mtp_logits_full = mtp_h.matmul(self.token_emb.transpose((1, 0)))

        return final_prob, p_gen, mtp_logits_full

    def generate(self, prompt_ids, max_new_tokens=40, temperature=0.9, top_k=20,
                 eos_id=None, on_token=None):
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = ids[-self.max_seq_len:]
            probs, _, _ = self.forward_probs(ctx, compute_mtp=False)
            last = probs.data[-1]
            if temperature is not None and temperature != 1.0:
                last = np.power(np.clip(last, 1e-12, None), 1.0 / max(temperature, 1e-6))
                last = last / last.sum()
            if top_k is not None and top_k < len(last):
                kth = np.partition(last, -top_k)[-top_k]
                last = np.where(last >= kth, last, 0.0)
                last = last / last.sum()
            next_id = int(np.random.choice(len(last), p=last))
            ids.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return ids

    def loss(self, inp, target):
        """inp, target: same-length lists of token ids (target[i] is the
        token that should follow inp[i]) -- exactly what encode_pattern()/
        chunk_corpus() already produce. Returns (total_loss, primary_loss,
        mtp_loss_or_None)."""
        T = len(inp)
        probs, p_gen, mtp_logits_full = self.forward_probs(inp)
        primary_loss = nll_from_probs(probs, target)

        if not self.use_mtp or T < 2:
            return primary_loss, primary_loss, None

        # target[i] is inp[i]'s next token; target[i+1] is therefore two
        # steps ahead of inp[i] -- exactly the MTP target.
        mtp_target = target[1:]
        n_valid = T - 1
        mtp_logits = Tensor(mtp_logits_full.data[:n_valid], requires_grad=True,
                             _children=(mtp_logits_full,), _op="slice")

        def _slice_backward():
            if mtp_logits_full.requires_grad:
                g = np.zeros_like(mtp_logits_full.data)
                g[:n_valid] = mtp_logits.grad
                mtp_logits_full.grad += g
        mtp_logits._backward = _slice_backward

        mtp_loss = softmax_cross_entropy(mtp_logits, np.array(mtp_target))
        total = primary_loss + mtp_loss * self.mtp_weight
        return total, primary_loss, mtp_loss


# ======================================================================
# ---- optimizer.py -- unchanged ----
# ======================================================================
class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self, clip=1.0, weight_decay=0.0):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if clip is not None:
                norm = np.linalg.norm(g)
                if norm > clip:
                    g = g * (clip / (norm + 1e-8))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if weight_decay > 0:
                p.data -= self.lr * weight_decay * p.data


# ======================================================================
# ---- main.py -- same CLI/workflow, model class swapped, CONFIG expanded
# ======================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(HERE, "model.npz")
META_PATH = os.path.join(HERE, "model.meta.json")
PATTERNS_PATH = os.path.join(HERE, "patterns.json")
CORPUS_PATH = os.path.join(HERE, "corpus.txt")
BPE_PATH = os.path.join(HERE, "bpe_tokenizer.json")

# ---- THE ARCHITECTURE CONFIG -- this is what you now get to tweak ----
CONFIG = dict(
    d_model=64, n_heads=4, n_layers=3, max_seq_len=256,
    d_latent=16,        # MLA bottleneck width -- only matters if use_mla=True
    d_ff_expert=96,      # width of each expert's hidden layer
    n_experts=4,          # 1 = MoE degenerates toward dense (see MoEFeedForward docstring)
    use_rope=True,          # rotary position embeddings vs. a learned pos_emb table
    use_rmsnorm=True,        # RMSNorm vs. LayerNorm
    use_mla=True,              # compressed-latent K/V vs. plain full-size K/V
    use_pointer=True,            # keep the original copy/pointer mechanism
    use_mtp=True,                  # auxiliary t+2 prediction head
    mtp_weight=0.3,
)
MIN_EPOCHS = 15
MAX_EPOCHS = 400
MAX_EPOCHS_MEMORIZE = 600
PATIENCE = 8
VAL_FRACTION = 0.15
MIN_EXAMPLES_FOR_VAL = 20
LR = 3e-3
BPE_TARGET_VOCAB_SIZE = 600
BPE_MIN_PAIR_FREQ = 2

TEMPLATES_PATH = os.path.join(HERE, "templates.json")
TEMPLATE_HOLDOUT_FRACTION = 0.25
TEMPLATE_CHECK_EVERY = 25
TEMPLATE_WEIGHT_DECAY = 0.1


def load_patterns():
    if not os.path.exists(PATTERNS_PATH):
        return []
    with open(PATTERNS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_patterns(patterns):
    with open(PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=1)


def import_patterns_from_file(path):
    if not os.path.exists(path):
        return None, f"file not found: {path}"
    added = []
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict) and "prompt" in item and "answer" in item:
                added.append({"prompt": item["prompt"], "answer": item["answer"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                added.append({"prompt": item[0], "answer": item[1]})
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or "||" not in line:
                    continue
                prompt, answer = line.split("||", 1)
                added.append({"prompt": prompt, "answer": answer})
    return added, None


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        return ""
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return f.read()


def append_corpus(new_text):
    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        f.write(new_text.strip() + "\n\n")


def read_pasted_text():
    print("paste your text below. finish with a line containing just: END")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_or_load_tokenizer():
    if os.path.exists(BPE_PATH):
        tok = BPETokenizer.load(BPE_PATH)
        print(f"[loaded bpe tokenizer: {tok.vocab_size} tokens "
              f"({len(tok.base_chars) + 4} base + {len(tok.merges)} merges)]")
        return tok
    print("[no bpe_tokenizer.json found -- training one from saved patterns/corpus]")
    return train_tokenizer()


def train_tokenizer(verbose=True):
    parts = [load_corpus()]
    for p in load_patterns():
        parts.append(p.get("prompt", "") + " " + p.get("answer", ""))
    synth_patterns, holdout_evals = expand_templates(load_templates())
    for p in synth_patterns:
        parts.append(p["prompt"])
    for item in holdout_evals:
        parts.append(item["prompt"] + " " + str(item["expected"]))
    text = "\n".join(t for t in parts if t)

    tok = BPETokenizer()
    if len(text) < 200:
        if verbose:
            print("[not enough text yet to learn meaningful subwords -- "
                  "using base characters only for now.]")
        tok.train(text, vocab_size=len(tok.stoi), min_pair_freq=BPE_MIN_PAIR_FREQ, verbose=verbose)
    else:
        tok.train(text, vocab_size=BPE_TARGET_VOCAB_SIZE, min_pair_freq=BPE_MIN_PAIR_FREQ, verbose=verbose)
    tok.save(BPE_PATH)
    return tok


def save_model(gpt, n_patterns, corpus_len):
    names, params = gpt.named_params()
    np.savez(SAVE_PATH, **{n: p.data for n, p in zip(names, params)})
    with open(META_PATH, "w") as f:
        json.dump({"n_patterns": n_patterns, "corpus_len": corpus_len, "config": gpt.cfg}, f, indent=1)


def load_model(gpt):
    if not os.path.exists(SAVE_PATH):
        return False, {"n_patterns": 0, "corpus_len": 0}
    data = np.load(SAVE_PATH)
    names, params = gpt.named_params()
    n_loaded = 0
    for n, p in zip(names, params):
        if n in data and data[n].shape == p.data.shape:
            p.data = data[n].copy()
            n_loaded += 1
    meta = {"n_patterns": 0, "corpus_len": 0}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    if n_loaded < len(names):
        print(f"[note: only {n_loaded}/{len(names)} saved tensors matched this CONFIG's "
              f"shapes -- the rest started from random init. This happens when you change "
              f"CONFIG after a previous :stabilize; run :stabilize again to retrain fully "
              f"under the new architecture.]")
    return True, meta


def encode_pattern(tok, prompt, answer, max_len):
    full = tok.encode(prompt, add_bos=True) + tok.encode(answer, add_bos=False, add_eos=True)
    full = full[:max_len + 1]
    return full[:-1], full[1:]


def chunk_corpus(tok, text, max_len):
    if not text:
        return []
    ids = tok.encode(text, add_bos=True, add_eos=True)
    chunks = []
    for i in range(0, max(0, len(ids) - 1), max_len):
        block = ids[i:i + max_len + 1]
        if len(block) >= 4:
            chunks.append((block[:-1], block[1:]))
    return chunks


def _split_train_val(examples, val_fraction, seed=0):
    examples = list(examples)
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    if n < 2:
        return examples, examples
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)
    val = examples[:n_val]
    train = examples[n_val:]
    return train, val


def _eval_loss(gpt, examples):
    if not examples:
        return None
    total = 0.0
    for inp, target in examples:
        total_loss, _, _ = gpt.loss(inp, target)
        total += float(total_loss.data)
    return total / len(examples)


def load_templates():
    if not os.path.exists(TEMPLATES_PATH):
        return []
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _cross_product_combos(var_groups):
    combos = [{}]
    for group_rows in var_groups:
        new_combos = []
        for combo in combos:
            for row in group_rows:
                merged = dict(combo)
                merged.update(row)
                new_combos.append(merged)
        combos = new_combos
    return combos


def expand_templates(templates, holdout_fraction=TEMPLATE_HOLDOUT_FRACTION, seed=0):
    train_patterns = []
    holdout_evals = []
    rng = random.Random(seed)
    for tdef in templates:
        template = tdef["template"]
        eval_template = tdef.get("eval_template", template)
        check_placeholder = tdef.get("check_placeholder")
        var_groups = []
        for group in tdef.get("vars", {}).values():
            placeholders = group["placeholders"]
            group_rows = [dict(zip(placeholders, values)) for values in group["values"]]
            var_groups.append(group_rows)
        combos = _cross_product_combos(var_groups)
        rng.shuffle(combos)
        n_holdout = int(round(len(combos) * holdout_fraction)) if check_placeholder else 0
        for i, flat_vars in enumerate(combos):
            if i < n_holdout:
                holdout_evals.append({"prompt": eval_template.format(**flat_vars),
                                       "expected": flat_vars[check_placeholder]})
            else:
                train_patterns.append({"prompt": template.format(**flat_vars), "answer": ""})
    return train_patterns, holdout_evals


def check_template_accuracy(gpt, tok, holdout_evals, max_new_tokens=15):
    if not holdout_evals:
        return None
    correct = 0
    for item in holdout_evals:
        ids = tok.encode(item["prompt"], add_bos=True)
        out = gpt.generate(ids, max_new_tokens=max_new_tokens, temperature=0.01,
                            top_k=1, eos_id=tok.eos_id)
        completion = tok.decode(out[len(ids):])
        if item["expected"] in completion:
            correct += 1
    return correct / len(holdout_evals)


def stabilize(tok, patterns, corpus_text):
    templates = load_templates()
    synth_patterns, holdout_evals = expand_templates(templates)
    if synth_patterns or holdout_evals:
        print(f"[templates.json: expanded into {len(synth_patterns)} synthetic training "
              f"examples + {len(holdout_evals)} held out to check generalization]")

    pattern_examples = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in patterns]
    synth_examples = [encode_pattern(tok, p["prompt"], p["answer"], CONFIG["max_seq_len"]) for p in synth_patterns]
    corpus_examples = chunk_corpus(tok, corpus_text, CONFIG["max_seq_len"])
    all_examples = pattern_examples + synth_examples + corpus_examples

    if not all_examples:
        print("[nothing saved yet -- teach it something or paste some text first]")
        return None, None

    use_val = len(all_examples) >= MIN_EXAMPLES_FOR_VAL
    if use_val:
        train_examples, val_examples = _split_train_val(all_examples, VAL_FRACTION)
    else:
        train_examples, val_examples = all_examples, []
        print(f"[only {len(all_examples)} examples -- skipping held-out validation, "
              f"targeting near-perfect memorization instead]")
    n_train = len(train_examples)
    epoch_cap = MAX_EPOCHS if use_val else MAX_EPOCHS_MEMORIZE

    print(f"[CONFIG: {CONFIG}]")
    print(f"[stabilizing: {len(pattern_examples)} patterns + {len(synth_examples)} synthetic + "
          f"{len(corpus_examples)} corpus blocks = {len(all_examples)} examples "
          f"({n_train} train / {len(val_examples)} val), up to {epoch_cap} epochs, "
          f"early stop after {PATIENCE} epochs with no improvement]")

    gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
    n_params = sum(p.data.size for p in gpt.params())
    print(f"[this configuration has {n_params:,} parameters]")
    opt = Adam(gpt.params(), lr=LR)

    best_monitor = float("inf")
    best_state = None
    epochs_since_improve = 0
    t0 = time.time()
    best_acc = -1.0
    best_acc_state = None
    MEMORIZE_THRESHOLD = 0.2

    for epoch in range(1, epoch_cap + 1):
        if not use_val:
            progress = epoch / epoch_cap
            opt.lr = LR * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * progress)))
        random.shuffle(train_examples)
        total = 0.0
        for inp, target in train_examples:
            gpt.zero_grad()
            total_loss, primary_loss, mtp_loss = gpt.loss(inp, target)
            total_loss.backward()
            opt.step(weight_decay=TEMPLATE_WEIGHT_DECAY if holdout_evals else 0.0)
            total += float(total_loss.data)
        train_avg = total / n_train
        val_avg = _eval_loss(gpt, val_examples) if use_val else None
        monitor = val_avg if use_val else train_avg

        improved = monitor is not None and monitor < best_monitor - 1e-4
        if improved:
            best_monitor = monitor
            names, params = gpt.named_params()
            best_state = {n: p.data.copy() for n, p in zip(names, params)}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        val_str = f"  val_loss={val_avg:.4f}" if use_val else ""
        print(f"  epoch {epoch:3d}  train_loss={train_avg:.4f}{val_str}"
              f"{'  (best)' if improved else ''}  elapsed={time.time()-t0:.1f}s")

        if holdout_evals and (epoch % TEMPLATE_CHECK_EVERY == 0 or epoch == 1):
            heldout_acc = check_template_accuracy(gpt, tok, holdout_evals)
            if heldout_acc > best_acc:
                best_acc = heldout_acc
                names, params = gpt.named_params()
                best_acc_state = {n: p.data.copy() for n, p in zip(names, params)}
                print(f"    [templates.json: held-out accuracy={heldout_acc:.2f} -- new best, checkpointed]")
            if best_acc >= 1.0 and epoch >= MIN_EPOCHS:
                print("[stopping: held-out accuracy reached 1.00]")
                break

        if epoch >= MIN_EPOCHS:
            if holdout_evals:
                pass
            elif not use_val:
                if train_avg < MEMORIZE_THRESHOLD:
                    print(f"[stopping: train_loss={train_avg:.4f} below memorization threshold]")
                    names, params = gpt.named_params()
                    best_state = {n: p.data.copy() for n, p in zip(names, params)}
                    break
            else:
                if epochs_since_improve >= PATIENCE:
                    print(f"[early stop: val_loss hasn't improved in {PATIENCE} epochs]")
                    break
    else:
        print(f"[hit epoch cap ({epoch_cap}) without early stopping -- restoring best checkpoint]")

    if holdout_evals and best_acc_state is not None:
        names, params = gpt.named_params()
        for n, p in zip(names, params):
            if n in best_acc_state:
                p.data = best_acc_state[n]
        save_model(gpt, len(patterns), len(corpus_text))
        print(f"[done -- best held-out accuracy={best_acc:.2f}, weights saved to {SAVE_PATH}]")
        return gpt, opt

    if best_state is not None:
        names, params = gpt.named_params()
        for n, p in zip(names, params):
            if n in best_state:
                p.data = best_state[n]

    save_model(gpt, len(patterns), len(corpus_text))
    print(f"[done -- best {'val' if use_val else 'train'}_loss={best_monitor:.4f}, "
          f"weights saved to {SAVE_PATH}]")
    return gpt, opt


def stream_generate(gpt, tok, prompt_text, max_new_tokens=80):
    prompt_ids = tok.encode(prompt_text, add_bos=True)
    print("assistant> ", end="", flush=True)

    def on_token(tid):
        piece = tok.itos.get(tid, "")
        if piece not in ("<EOS>", "<BOS>", "<PAD>"):
            print(piece, end="", flush=True)

    out_ids = gpt.generate(prompt_ids, max_new_tokens=max_new_tokens,
                            temperature=0.8, top_k=20, eos_id=tok.eos_id,
                            on_token=on_token)
    print()
    return out_ids[len(prompt_ids):]


def print_status(patterns, corpus_text, meta):
    n, clen = len(patterns), len(corpus_text)
    print(f"[{n} patterns saved, {clen} corpus characters saved]")
    if meta.get("n_patterns") == n and meta.get("corpus_len") == clen:
        print("[model is up to date with everything saved]")
    else:
        print(f"[model last stabilized at {meta.get('n_patterns',0)} patterns / "
              f"{meta.get('corpus_len',0)} corpus chars -- run :stabilize to catch it up]")
    if "config" in meta:
        print(f"[last stabilized CONFIG: {meta['config']}]")


def handle_command(cmd_line, tok, patterns, corpus_text, gpt, opt, meta):
    parts = cmd_line.split(maxsplit=1)
    cmd = parts[0]

    if cmd == ":stabilize":
        new_gpt, new_opt = stabilize(tok, patterns, corpus_text)
        if new_gpt is not None:
            gpt, opt = new_gpt, new_opt
            meta = {"n_patterns": len(patterns), "corpus_len": len(corpus_text), "config": CONFIG}
        return True, gpt, opt, meta

    if cmd == ":status":
        print_status(patterns, corpus_text, meta)
        return True, gpt, opt, meta

    if cmd == ":import":
        if len(parts) < 2:
            print("usage: :import <path to .json or .txt file>")
            return True, gpt, opt, meta
        added, err = import_patterns_from_file(parts[1].strip())
        if err:
            print(f"[import failed: {err}]")
        else:
            patterns.extend(added)
            save_patterns(patterns)
            print(f"[imported {len(added)} patterns ({len(patterns)} total now, "
                  f"not yet stabilized -- :stabilize when ready)]")
        return True, gpt, opt, meta

    return False, gpt, opt, meta


def main():
    print("=" * 60)
    print(" configurable pointerbot -- same commands, architecture is now CONFIG")
    print(" teach: type patterns | learn: paste text | :import: bulk-load")
    print(" :stabilize retrains from scratch on EVERYTHING under current CONFIG")
    print("=" * 60)

    tok = build_or_load_tokenizer()
    gpt = ConfigurableGPT(vocab_size=tok.vocab_size, **CONFIG)
    loaded, meta = load_model(gpt)
    opt = Adam(gpt.params(), lr=LR)
    patterns = load_patterns()
    corpus_text = load_corpus()

    print(f"[model {'loaded' if loaded else 'freshly initialized, no saved weights yet'}]")
    print_status(patterns, corpus_text, meta)
    print(f"[params: {sum(p.data.size for p in gpt.params()):,}]\n")

    mode = None
    try:
        while True:
            if mode is None:
                cmd = input("mode? [chat / teach / learn / template / quit] > ").strip()
                if cmd == ":quit":
                    break
                if cmd == ":mode":
                    continue
                if cmd.startswith(":"):
                    handled, gpt, opt, meta = handle_command(cmd, tok, patterns, corpus_text, gpt, opt, meta)
                    continue
                cmd = cmd.lower()
                if cmd in ("quit", "exit", "q"):
                    break
                if cmd not in ("chat", "teach", "learn", "template"):
                    print("please type: chat, teach, learn, template, :stabilize, :status, :import <path>, or quit")
                    continue
                mode = cmd
                print(f"-- {mode} mode -- (:mode to switch, :stabilize to retrain, :quit to exit)")
                continue

            if mode == "template":
                print("[template wizard omitted from this listing for brevity -- "
                      "identical to your original file's run_template_wizard(); "
                      "copy it over verbatim if you use templates.json]")
                mode = None
                continue

            if mode == "learn":
                text = read_pasted_text()
                if not text:
                    print("[nothing pasted, skipping]")
                else:
                    append_corpus(text)
                    corpus_text = load_corpus()
                    print(f"[saved ({len(corpus_text)} corpus characters total)]")
                mode = None
                continue

            user = input("you> ").strip()
            if user == ":quit":
                break
            if user == ":mode":
                mode = None
                continue
            if user.startswith(":"):
                handled, gpt, opt, meta = handle_command(user, tok, patterns, corpus_text, gpt, opt, meta)
                continue
            if user == "":
                continue

            if mode == "chat":
                stream_generate(gpt, tok, user)

            elif mode == "teach":
                reply_ids = stream_generate(gpt, tok, user)
                reply_text = tok.decode(reply_ids)
                fb = input("was that right? [y/n] > ").strip().lower()
                if fb == "y":
                    patterns.append({"prompt": user, "answer": reply_text})
                    save_patterns(patterns)
                    print(f"[pattern saved ({len(patterns)} total, not yet stabilized)]")
                elif fb == "n":
                    correct = input("what should it have said? > ")
                    patterns.append({"prompt": user, "answer": correct})
                    save_patterns(patterns)
                    print(f"[pattern saved ({len(patterns)} total, not yet stabilized)]")
                else:
                    print("[skipped -- not saved]")
    except (EOFError, KeyboardInterrupt):
        pass

    print("\nbye!")


if __name__ == "__main__":
    main()
