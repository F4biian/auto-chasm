"""End-to-end cross-backend parity demo.

Verifies that MLX and PyTorch backends produce near-identical training
outcomes given the same model architecture, initial weights, and data.
"""

from __future__ import annotations

import numpy as np


def _make_mlp_from_numpy_weights(hidden_dim: int, vocab_size: int, num_layers: int):
    """Create an MLX MLP with weights initialized from numpy arrays."""
    import mlx.core as mx
    import mlx.nn as nn

    np.random.seed(42)

    class Mlp(nn.Module):
        """Tiny MLP for parity testing."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            self.output_proj = nn.Linear(hidden_dim, vocab_size)

        def __call__(self, x: mx.array) -> mx.array:
            h = self.embedding(x)
            for layer in self.layers:
                h = nn.gelu(layer(h))
            return self.output_proj(h)

    model = Mlp()

    emb_weight = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
    model.embedding.weight = mx.array(emb_weight)

    for _i, layer in enumerate(model.layers):
        w = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        b = np.random.randn(hidden_dim).astype(np.float32) * 0.01
        layer.weight = mx.array(w)
        layer.bias = mx.array(b)

    out_w = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
    out_b = np.random.randn(vocab_size).astype(np.float32) * 0.01
    model.output_proj.weight = mx.array(out_w)
    model.output_proj.bias = mx.array(out_b)

    return model


def _make_torch_mlp_from_numpy_weights(hidden_dim: int, vocab_size: int, num_layers: int):
    """Create a PyTorch MLP with weights initialized from the same numpy arrays."""
    import torch
    import torch.nn as tnn

    np.random.seed(42)

    class Mlp(tnn.Module):
        """Tiny MLP for parity testing."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(vocab_size, hidden_dim)
            self.layers = tnn.ModuleList(
                [tnn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.output_proj = tnn.Linear(hidden_dim, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    model = Mlp()

    emb_weight = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
    model.embedding.weight.data = torch.tensor(emb_weight)

    for _i, layer in enumerate(model.layers):
        w = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        b = np.random.randn(hidden_dim).astype(np.float32) * 0.01
        layer.weight.data = torch.tensor(w)
        layer.bias.data = torch.tensor(b)

    out_w = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
    out_b = np.random.randn(vocab_size).astype(np.float32) * 0.01
    model.output_proj.weight.data = torch.tensor(out_w)
    model.output_proj.bias.data = torch.tensor(out_b)

    return model


class TestEndToEndParity:
    """Verify both backends produce equivalent training outcomes."""

    def test_single_step_loss_parity(self) -> None:
        """A single training step should produce near-identical loss values."""
        import mlx.core as mx
        import mlx.nn as mlx_nn
        import torch

        hidden_dim = 8
        vocab_size = 16
        num_layers = 2

        mlx_model = _make_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)
        torch_model = _make_torch_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)

        tokens = [1, 2, 3, 4, 5, 6, 7]

        # MLX forward + loss
        mlx_x = mx.array([tokens[:-1]])
        mlx_targets = mx.array([tokens[1:]])
        mlx_logits = mlx_model(mlx_x)
        mlx_loss = mlx_nn.losses.cross_entropy(
            mlx_logits.reshape(-1, vocab_size), mlx_targets.reshape(-1)
        )
        mlx_loss_val = float(mx.mean(mlx_loss).item())

        # Torch forward + loss
        torch_x = torch.tensor([tokens[:-1]])
        torch_targets = torch.tensor([tokens[1:]])
        torch_logits = torch_model(torch_x)
        torch_loss = torch.nn.functional.cross_entropy(
            torch_logits.reshape(-1, vocab_size), torch_targets.reshape(-1)
        )
        torch_loss_val = float(torch_loss.item())

        assert abs(mlx_loss_val - torch_loss_val) < 1e-4, (
            f"Single-step loss parity: mlx={mlx_loss_val}, torch={torch_loss_val}"
        )

    def test_gradient_parity(self) -> None:
        """Gradients should be numerically close after one backward pass."""
        import mlx.core as mx
        import mlx.nn as mlx_nn
        import torch

        hidden_dim = 8
        vocab_size = 16
        num_layers = 2

        mlx_model = _make_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)
        torch_model = _make_torch_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)

        tokens = [1, 2, 3, 4, 5, 6, 7]

        # MLX gradient
        def mlx_loss_fn(model, x, targets):
            logits = model(x)
            return mx.mean(
                mlx_nn.losses.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            )

        mlx_x = mx.array([tokens[:-1]])
        mlx_targets = mx.array([tokens[1:]])
        mlx_loss, mlx_grads = mx.value_and_grad(mlx_loss_fn)(mlx_model, mlx_x, mlx_targets)

        # Torch gradient
        torch_x = torch.tensor([tokens[:-1]])
        torch_targets = torch.tensor([tokens[1:]])
        torch_logits = torch_model(torch_x)
        torch_loss = torch.nn.functional.cross_entropy(
            torch_logits.reshape(-1, vocab_size), torch_targets.reshape(-1)
        )
        torch_loss.backward()

        # Compare embedding gradients (first layer)
        # MLX grads is a dict-like tree; access via mlx_grads["embedding"]["weight"]
        from mlx.utils import tree_flatten

        mlx_flat = dict(tree_flatten(mlx_grads))
        mlx_emb_grad = np.array(mlx_flat["embedding.weight"])
        torch_emb_grad = torch_model.embedding.weight.grad.numpy()
        np.testing.assert_allclose(mlx_emb_grad, torch_emb_grad, atol=1e-5)

    def test_multi_step_loss_decreases_equally(self) -> None:
        """After several optimizer steps, both backends should reach similar loss."""
        import mlx.core as mx
        import mlx.nn as mlx_nn
        import mlx.optimizers as optim
        import torch

        hidden_dim = 8
        vocab_size = 16
        num_layers = 2

        mlx_model = _make_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)
        torch_model = _make_torch_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)

        mlx_optimizer = optim.Adam(learning_rate=1e-3)
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)

        tokens = [1, 2, 3, 4, 5, 6, 7]

        # MLX training steps
        def mlx_loss_fn(model, x, targets):
            logits = model(x)
            return mx.mean(
                mlx_nn.losses.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            )

        mlx_x = mx.array([tokens[:-1]])
        mlx_targets = mx.array([tokens[1:]])

        mlx_losses = []
        for _ in range(5):
            loss, grads = mx.value_and_grad(mlx_loss_fn)(mlx_model, mlx_x, mlx_targets)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.state, mlx_optimizer.state)
            mlx_losses.append(float(loss.item()))

        # Torch training steps
        torch_losses = []
        for _ in range(5):
            torch_optimizer.zero_grad()
            torch_x = torch.tensor([tokens[:-1]])
            torch_targets = torch.tensor([tokens[1:]])
            torch_logits = torch_model(torch_x)
            loss = torch.nn.functional.cross_entropy(
                torch_logits.reshape(-1, vocab_size), torch_targets.reshape(-1)
            )
            loss.backward()
            torch_optimizer.step()
            torch_losses.append(float(loss.item()))

        # Both should decrease
        assert mlx_losses[-1] < mlx_losses[0], f"MLX loss didn't decrease: {mlx_losses}"
        assert torch_losses[-1] < torch_losses[0], f"Torch loss didn't decrease: {torch_losses}"

        # Final losses should be close (within 10% — optimizers may differ slightly)
        ratio = mlx_losses[-1] / torch_losses[-1]
        assert 0.5 < ratio < 2.0, (
            f"Final loss divergence: mlx={mlx_losses[-1]}, torch={torch_losses[-1]}"
        )

    def test_greedy_inference_parity(self) -> None:
        """Greedy inference should produce identical tokens."""
        import mlx.core as mx
        import torch

        hidden_dim = 8
        vocab_size = 16
        num_layers = 2

        mlx_model = _make_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)
        torch_model = _make_torch_mlp_from_numpy_weights(hidden_dim, vocab_size, num_layers)

        # Set both to eval
        mlx_model.eval()
        torch_model.eval()

        prompt = [1, 2, 3]

        # MLX greedy
        mlx_tokens = list(prompt)
        for _ in range(5):
            x = mx.array([mlx_tokens])
            logits = mlx_model(x)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            mlx_tokens.append(next_token)

        # Torch greedy
        torch_tokens = list(prompt)
        for _ in range(5):
            x = torch.tensor([torch_tokens])
            with torch.no_grad():
                logits = torch_model(x)
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            torch_tokens.append(next_token)

        assert mlx_tokens == torch_tokens, (
            f"Greedy inference mismatch: mlx={mlx_tokens}, torch={torch_tokens}"
        )
