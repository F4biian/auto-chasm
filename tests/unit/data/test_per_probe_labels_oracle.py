"""Oracle tests for per-probe labels (independent targets for multiple heads).

These pin the correctness of the multi-head label routing that ``build_dataset``
(dict labels) → ``iterate_batches`` (per-probe ``[B, T]`` arrays) → ``JointLoss``
(per-probe targets) implements end to end.

The decisive property each test checks is *independence*: head ``a`` trains on
its own target and head ``b`` on a **different** one.  If labels bled together
(the old single-shared-array behavior), every one of these would fail — head
``b`` would be scored against head ``a``'s labels.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.data import build_dataset
from auto_chasm.trainers.data_utils import iterate_batches

# softplus(-10) = BCE-with-logits at logit=10 for the *matching* label; the
# *mismatched* label adds the logit itself (10).
_SOFTPLUS_NEG10 = math.log1p(math.exp(-10.0))


class _MockWordTokenizer:
    """One token id per whitespace word; offsets via the char-split fallback."""

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        """Return one id per whitespace-delimited word."""
        return [100 + i for i, _ in enumerate(text.split())]


class TestBuildDatasetEmitsPerProbeDict:
    """build_dataset emits independent per-probe label arrays for >= 2 heads."""

    def test_two_heads_get_their_own_spans(self) -> None:
        """Each head's positive span lands only in that head's array."""
        tok = _MockWordTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {
                        "hall": [{"start": 0, "end": 5, "label": 1}],
                        "qual": [{"start": 10, "end": 16, "label": 1}],
                    },
                }
            ]
        ]
        labels = build_dataset(conversations, tok, default_label=0)[0]["labels"]
        assert isinstance(labels, dict)
        # "alpha" is hall's positive; "gamma" is qual's positive; heads disjoint.
        assert labels["hall"] == [1, 0, 0]
        assert labels["qual"] == [0, 0, 1]


class TestIterateBatchesPerProbe:
    """iterate_batches turns per-sample dict labels into per-probe [B, T] arrays."""

    def test_emits_dict_of_matrices(self) -> None:
        """Two heads → a dict of [B, T] arrays; values land at the right cells."""
        data = [
            {"tokens": [1, 2, 3], "labels": {"a": [1, 0, 1], "b": [0, 1, 0]}},
            {"tokens": [4, 5, 6], "labels": {"a": [0, 0, 1], "b": [1, 1, 0]}},
        ]
        _tokens, labels, _lengths = next(iterate_batches(data, 2, 16, loop=False))
        assert isinstance(labels, dict)
        assert set(labels) == {"a", "b"}
        # Row order follows length-sort (equal lengths → stable); check both rows.
        assert sorted(labels["a"][:, :3].tolist()) == sorted([[1, 0, 1], [0, 0, 1]])
        assert sorted(labels["b"][:, :3].tolist()) == sorted([[0, 1, 0], [1, 1, 0]])

    def test_per_probe_dtype_is_independent(self) -> None:
        """A float (regression) head and an int (class) head keep their dtypes."""
        data = [{"tokens": [1, 2, 3], "labels": {"reg": [0.1, 0.5, 0.9], "cls": [0, 1, 0]}}]
        _tokens, labels, _lengths = next(iterate_batches(data, 1, 16, loop=False))
        assert labels["reg"].dtype.kind == "f"
        assert labels["cls"].dtype.kind in ("i", "u")

    def test_sample_missing_a_head_is_masked(self) -> None:
        """A sample that omits a head contributes an all-(-100) row for it."""
        data = [
            {"tokens": [1, 2, 3], "labels": {"a": [1, 1, 1]}},
            {"tokens": [4, 5, 6], "labels": {"a": [0, 0, 0], "b": [1, 1, 1]}},
        ]
        # Single batch keeps both samples; "b" is absent from the first sample.
        _tokens, labels, _lengths = next(iterate_batches(data, 2, 16, loop=False))
        # Exactly one row of "b" is all -100 (the sample that omitted it).
        b_rows = labels["b"][:, :3].tolist()
        assert [-100, -100, -100] in b_rows
        assert [1, 1, 1] in b_rows


class TestJointLossRoutesPerProbe:
    """JointLoss scores each head against ITS OWN labels (hand-computed BCE).

    The fake model emits a constant logit of +10 for *both* heads, so head a
    (labels 1) reads BCE ~ softplus(-10) ~ 4.5e-5 (matched) while head b
    (labels 0) reads ~ 10 + softplus(-10) ~ 10.0 (mismatched).
    """

    def test_matched_head_low_mismatched_head_high_mlx(self) -> None:
        """Head a (labels match logits) ~0; head b (labels oppose) ~10."""
        loss = JointLoss(weights={"lm_head": 0.0})
        batch = mx.array([[1, 2, 3, 4]])
        labels = {"a": mx.array([[1, 1, 1, 1]]), "b": mx.array([[0, 0, 0, 0]])}
        lengths = mx.array([[0, 4]])

        class _FixedTwoHead:
            def __call__(self, inputs: mx.array, mask: mx.array | None = None) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 8)), {
                    "a": mx.full((b, t), 10.0),
                    "b": mx.full((b, t), 10.0),
                }

        total, _ntoks, comp = loss(_FixedTwoHead(), batch, labels, lengths)
        # If routing were shared, head b would be scored against labels {1,1,1}
        # too and read ~0 — this asserts it is scored against ITS OWN {0,0,0}.
        assert float(comp["a"]) == pytest.approx(_SOFTPLUS_NEG10, abs=1e-5)
        assert float(comp["b"]) == pytest.approx(10.0 + _SOFTPLUS_NEG10, abs=1e-4)
        assert float(total) == pytest.approx(10.0 + 2 * _SOFTPLUS_NEG10, abs=1e-4)

    def test_mlx_torch_parity(self) -> None:
        """The per-head routed losses match across MLX and PyTorch."""
        pytest.importorskip("torch")
        import torch

        loss = JointLoss(weights={"lm_head": 0.0})

        class _MlxTwoHead:
            def __call__(self, inputs: mx.array, mask: mx.array | None = None) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 8)), {
                    "a": mx.full((b, t), 10.0),
                    "b": mx.full((b, t), 10.0),
                }

        class _TorchTwoHead:
            def __call__(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple:
                b, t = inputs.shape
                return torch.zeros((b, t, 8)), {
                    "a": torch.full((b, t), 10.0),
                    "b": torch.full((b, t), 10.0),
                }

        mlx_labels = {"a": mx.array([[1, 1, 1, 1]]), "b": mx.array([[0, 0, 0, 0]])}
        _t, _n, mc = loss(_MlxTwoHead(), mx.array([[1, 2, 3, 4]]), mlx_labels, mx.array([[0, 4]]))

        t_labels = {
            "a": torch.tensor([[1, 1, 1, 1]], dtype=torch.float32),
            "b": torch.tensor([[0, 0, 0, 0]], dtype=torch.float32),
        }
        _t2, _n2, tc = loss(
            _TorchTwoHead(), torch.tensor([[1, 2, 3, 4]]), t_labels, torch.tensor([[0, 4]])
        )
        assert float(mc["a"]) == pytest.approx(float(tc["a"]), abs=1e-4)
        assert float(mc["b"]) == pytest.approx(float(tc["b"]), abs=1e-4)


class _TinyMlp(nn.Module):
    """A minimal transformer-shaped MLX model for end-to-end training."""

    def __init__(self, h: int = 16, v: int = 32, layers: int = 2) -> None:
        """Build embedding → linear blocks → output projection."""
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        """Embed, pass through the blocks, project to the vocab."""
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    """Minimal config the probe machinery reads for hidden size / depth."""

    hidden_size = 16
    num_hidden_layers = 2


class TestTwoHeadsLearnIndependentTargets:
    """End-to-end: two heads trained on OPPOSITE targets diverge correctly."""

    def test_heads_diverge(self) -> None:
        """Head a (target 1) ends positive; head b (target 0) ends negative."""
        from auto_chasm import Trainer

        model = Model(_TinyMlp(), None, "mlx")
        model.model.config = _Cfg()
        model.attach_probe(ProbeConfig(name="a", layers=[0]))
        model.attach_probe(ProbeConfig(name="b", layers=[0]))

        # Opposite per-head targets over the whole sequence.
        data = [
            {"tokens": [1, 2, 3, 4, 5], "labels": {"a": [1, 1, 1, 1, 1], "b": [0, 0, 0, 0, 0]}}
            for _ in range(8)
        ]

        trainer = Trainer(
            model=model,
            loss_fn=JointLoss(weights={"lm_head": 0.0}),
            num_iters=60,
            batch_size=4,
            max_seq_length=16,
            learning_rate=5e-2,
            verbose=False,
        )
        trainer.train(data)

        # Read each head's logits after training.
        from auto_chasm.trainers.trainable import _TrainableModel

        tm = _TrainableModel(model.model, model._probes)
        _lm, probes = tm(mx.array([[1, 2, 3, 4, 5]]))
        mean_a = float(probes["a"].mean())
        mean_b = float(probes["b"].mean())
        # Each head learned ITS OWN target: a → 1 (positive logit), b → 0
        # (negative logit).  A shared-label bug would push both the same way.
        assert mean_a > 0.5, f"head a should predict 1 (logit>0), got {mean_a}"
        assert mean_b < -0.5, f"head b should predict 0 (logit<0), got {mean_b}"
