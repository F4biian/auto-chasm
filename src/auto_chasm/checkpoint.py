"""Checkpoint save / load / export.

Saves probes, adapters, steering geometry, and training metadata
into a self-describing directory.  ``Model.from_checkpoint()``
restores everything in one call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from auto_chasm._checkpoint_weights import (
    load_probe_weights,
    safetensors_framework,
    save_probe_weights,
)
from auto_chasm.logger import get_logger

if TYPE_CHECKING:
    from auto_chasm.model import Model

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
ADAPTERS_NAME = "adapters.safetensors"
PROBES_DIR = "probes"
STEERING_DIR = "steering"

# Bump the major component when the on-disk format changes incompatibly.
CHECKPOINT_FORMAT_VERSION = "0.1.0"


def _check_checkpoint_version(manifest: dict[str, Any]) -> None:
    """Warn if a checkpoint was written by an incompatible format version.

    Args:
        manifest: The loaded manifest dict.
    """
    saved = str(manifest.get("auto_chasm_version", "unknown"))
    current_major = CHECKPOINT_FORMAT_VERSION.split(".")[0]
    saved_major = saved.split(".")[0]
    if saved == "unknown":
        logger.warning(
            "Checkpoint manifest has no version field; it may be from an older or "
            "hand-written checkpoint. Proceeding, but the format is not guaranteed."
        )
    elif saved_major != current_major:
        logger.warning(
            "Checkpoint format version %s differs in major version from the current "
            "%s; restoration may be incomplete.",
            saved,
            CHECKPOINT_FORMAT_VERSION,
        )


def _serialize_module_type(module_type: Any) -> Any:
    """Serialize a probe's ``module_type`` for the manifest.

    A string name passes through. A declarative :class:`ModuleSpec` (the library's
    own reconstructable head) with a string activation is stored structurally so it
    round-trips. Any other callable — an arbitrary lambda, or a ``ModuleSpec`` with a
    callable activation — becomes the ``"__callable__"`` sentinel (not reconstructable).

    Args:
        module_type: The probe config's ``module_type``.

    Returns:
        A JSON-serializable representation.
    """
    import dataclasses

    from auto_chasm.modules import ModuleSpec

    if isinstance(module_type, str):
        return module_type
    if isinstance(module_type, ModuleSpec) and isinstance(module_type.activation, str):
        return {"__module_spec__": dataclasses.asdict(module_type)}
    return "__callable__"


def _deserialize_module_type(value: Any) -> Any:
    """Inverse of :func:`_serialize_module_type` — rebuild a ``ModuleSpec`` from its dict.

    Args:
        value: The stored ``module_type`` (string, ``"__callable__"``, or a spec dict).

    Returns:
        A string, or a reconstructed :class:`ModuleSpec`.
    """
    if isinstance(value, dict) and "__module_spec__" in value:
        from auto_chasm.modules import ModuleSpec

        fields = dict(value["__module_spec__"])
        fields["hidden_dims"] = tuple(fields.get("hidden_dims", ()))
        return ModuleSpec(**fields)
    return value


def save_checkpoint(model: Model, path: str) -> None:
    """Save a model checkpoint to a self-describing directory.

    Saves probe weights, adapter weights (if applicable), steering
    geometry, and a manifest with full configuration.

    Args:
        model: The ``Model`` instance to save.
        path: Directory path to save to.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    if model.probes:
        (p / PROBES_DIR).mkdir(exist_ok=True)
    if model.steering_hooks:
        (p / STEERING_DIR).mkdir(exist_ok=True)

    # Re-saving into an existing checkpoint dir must not leave stale files behind:
    # a removed probe's weights, or an adapters file from a prior LoRA save that
    # would make a now-LoRA-free reload inject a phantom adapter.
    _prune_orphans(p / PROBES_DIR, {f"{n}.safetensors" for n in model.probes}, "*.safetensors")
    _prune_orphans(p / STEERING_DIR, {f"{n}.json" for n in model.steering_hooks}, "*.json")
    if model.lora_config is None and (p / ADAPTERS_NAME).exists():
        (p / ADAPTERS_NAME).unlink()
        logger.info("Removed orphaned adapters file (model has no LoRA).")

    manifest: dict[str, Any] = {
        "auto_chasm_version": "0.1.0",
        "backend": model.backend.name,
        "base_model": getattr(model, "_base_model_name", None),
        "probes": {},
        "steering": {},
    }

    # Store LoRA config if present
    lora_cfg = model.lora_config
    if lora_cfg is not None:
        manifest["lora"] = {
            "rank": lora_cfg.rank,
            "alpha": lora_cfg.alpha,
            "dropout": lora_cfg.dropout,
            "target_modules": lora_cfg.target_modules,
            "target_layers": lora_cfg.target_layers,
            "until_layer": lora_cfg.until_layer,
            "after_layer": lora_cfg.after_layer,
            "peft_method": lora_cfg.peft_method,
        }

    for name, probe in model.probes.items():
        save_probe_weights(probe, p / PROBES_DIR / f"{name}.safetensors", model.backend)
        manifest["probes"][name] = {
            "layers": probe.config.layers,
            "source": probe.config.source,
            "aggregation": probe.config.aggregation
            if isinstance(probe.config.aggregation, str)
            else "__callable__",
            "module_type": _serialize_module_type(probe.config.module_type),
            "module_config": probe.config.module_config,
            "granularity": probe.config.granularity,
            "layer_norm": probe.config.layer_norm,
        }

    for name, hook in model.steering_hooks.items():
        steering_path = p / STEERING_DIR / f"{name}.json"
        with open(steering_path, "w") as f:
            json.dump(hook.to_dict(), f, indent=2)
        manifest["steering"][name] = {
            "method": hook.config.method,
            "scale": hook.config.scale,
            "has_geometry": hook.has_geometry,
        }

    # Only write adapters when the model actually has LoRA. Otherwise the MLX
    # backend would dump the entire base model (it saves all trainable params),
    # creating a phantom adapters file that makes reload inject a default
    # LoraConfig the user never asked for.
    if model.lora_config is not None:
        _save_adapters(model, p / ADAPTERS_NAME)
        # Trainable base weights beyond the LoRA adapters (an *additionally* unfrozen
        # base) are still lost — the adapter file holds only the adapters.
        _warn_unpersisted_base(model, _count_tensor_file(p / ADAPTERS_NAME))
    else:
        _warn_unpersisted_base(model, 0)

    # Added special tokens grow the embedding/tokenizer, neither of which this
    # checkpoint persists — a reload restores the base vocab and any logits probe
    # sized to the grown table would then mismatch. Warn so they get re-added.
    n_added = getattr(model, "_n_added_special_tokens", 0)
    if n_added:
        logger.warning(
            "save_checkpoint does not persist added special tokens: this model added %d "
            "token(s) and grew its embedding, which are NOT in the checkpoint. Re-add them "
            "with add_special_tokens() after reload before using the model.",
            n_added,
        )

    manifest_path = p / MANIFEST_NAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Checkpoint saved to %s", p)


def _prune_orphans(directory: Path, keep: set[str], pattern: str) -> None:
    """Delete files in ``directory`` matching ``pattern`` whose name is not in ``keep``.

    Args:
        directory: Directory to scan (a no-op if it does not exist).
        keep: Filenames to preserve.
        pattern: Glob for candidate files (e.g. ``"*.safetensors"``).
    """
    if not directory.exists():
        return
    for f in directory.glob(pattern):
        if f.name not in keep:
            f.unlink()
            logger.info("Removed orphaned checkpoint file: %s", f)


def _count_tensor_file(path: Path) -> int:
    """Return the number of tensors in a safetensors/torch weight file (0 if missing).

    Args:
        path: The weight file to inspect.

    Returns:
        Tensor (key) count, without materialising the values where possible.
    """
    if not path.exists():
        return 0
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework=safetensors_framework()) as f:
            return len(f.keys())  # noqa: SIM118  # safe_open is not a dict
    except Exception:
        import torch

        return len(torch.load(str(path), map_location="cpu", weights_only=False))


def _warn_unpersisted_base(model: Model, n_saved_adapter_tensors: int) -> None:
    """Warn when trainable base weights would be silently dropped by the checkpoint.

    The checkpoint stores probes, LoRA adapters, and steering geometry — never the
    raw base weights. A model whose base is unfrozen (a full fine-tune, or LoRA plus
    a manually-unfrozen base) would therefore lose that training on reload, which
    restores the pristine pretrained base. The count is approximate for probes held
    as separate modules, but a real base always dwarfs the probe so the warning is
    reliable in practice.

    Args:
        model: The model being saved.
        n_saved_adapter_tensors: Number of tensors persisted to the adapter file
            (0 when there is no LoRA), excluded from the "unpersisted" tally.
    """
    n_total = len(model.backend.module.trainable_parameters(model.model))
    n_probe = sum(
        len(model.backend.module.trainable_parameters(pr.module)) for pr in model._probes.values()
    )
    n_unpersisted = n_total - n_probe - n_saved_adapter_tensors
    if n_unpersisted > 0:
        logger.warning(
            "save_checkpoint does not persist base-model weights: this model has ~%d "
            "trainable base parameter tensor(s) that are neither probe nor saved-adapter "
            "weights. They are NOT in the checkpoint; a reload restores the pretrained "
            "base. Freeze the base, attach a LoraConfig, or save the base separately.",
            n_unpersisted,
        )


def _save_adapters(model: Model, path: Path) -> None:
    """Save adapter weights if the model has them.

    Delegates to the backend's ``save_adapters`` for both MLX and
    PyTorch, producing a single file at ``path``.

    Args:
        model: The ``Model`` instance.
        path: File path for the adapters.
    """
    try:
        model.backend.wrapping.save_adapters(model.model, str(path))
        logger.info("Adapter weights saved.")
    except Exception as e:
        logger.warning("Could not save adapters: %s", e)


def load_checkpoint(
    path: str,
    base_model: str | None = None,
    load_steering: bool = True,
    backend_name: str | None = None,
) -> Model:
    """Load a model from a checkpoint directory.

    Restores probes, adapter weights, and optionally steering geometry.
    If a ``training_manifest.json`` exists (written by the trainer),
    automatically loads the best checkpoint and infers the base model.

    Args:
        path: Checkpoint directory path.
        base_model: Override base model name.
        load_steering: Whether to restore steering data.
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        A fully restored ``Model`` instance.
    """
    from auto_chasm.config import ProbeConfig, SteeringConfig
    from auto_chasm.model import Model

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    manifest_path = p / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Checkpoint at {p} has no {MANIFEST_NAME}; it is not a valid auto-chasm "
            f"checkpoint directory."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    _check_checkpoint_version(manifest)

    if backend_name is None:
        backend_name = manifest.get("backend")

    model_name = base_model or manifest.get("base_model")
    if model_name is None:
        raise ValueError("No base_model in manifest. Pass base_model='...' to load_checkpoint().")

    if base_model is not None and base_model != manifest.get("base_model"):
        logger.warning(
            "Loading checkpoint onto base_model=%r, which differs from the one it "
            "was saved with (%r). Adapter/probe shapes are not validated against the "
            "new architecture; mismatched weights will be reported as load warnings.",
            base_model,
            manifest.get("base_model"),
        )

    from auto_chasm.backends.loaders import resolve_backend_name

    model = Model.from_pretrained(model_name, backend_name=resolve_backend_name(backend_name))
    model._base_model_name = model_name  # type: ignore[attr-defined]

    # Apply LoRA only when the manifest recorded a real LoRA config. Keying on
    # the adapters file alone would fabricate a default LoraConfig for a
    # probe-only checkpoint (phantom LoRA), since the MLX backend may write an
    # adapters file even without LoRA.
    lora_cfg = manifest.get("lora")
    adapter_path = p / ADAPTERS_NAME
    if lora_cfg is not None:
        model.attach_lora(_lora_from_manifest(lora_cfg))

    # Attach probes BEFORE loading adapters (but after attach_lora — mirroring the
    # save-time order). A hidden/sub-block probe wraps its layer, so adapters
    # trained at that layer were SAVED under the wrapper's
    # ``model.layers.N.layer.self_attn.*`` keys. If adapters load while the model
    # is still unwrapped, those keys silently fail to match (load_weights
    # strict=False) and that layer's adapters stay at zero-init — corrupting the
    # reloaded model's logits and probe outputs.
    probe_names: list[str] = []
    for name, probe_cfg in manifest.get("probes", {}).items():
        _check_probe_reconstructable(name, probe_cfg)
        config = ProbeConfig(
            name=name,
            layers=probe_cfg["layers"],
            source=probe_cfg.get("source", "hidden"),
            aggregation=probe_cfg.get("aggregation", "concat"),
            module_type=_deserialize_module_type(probe_cfg.get("module_type", "linear")),
            module_config=probe_cfg.get("module_config", {}),
            granularity=probe_cfg.get("granularity", "token"),
            layer_norm=probe_cfg.get("layer_norm", False),
        )
        model.attach_probe(config)
        probe_names.append(name)

    if lora_cfg is not None and adapter_path.exists():
        _load_adapters(model, adapter_path)

    for name in probe_names:
        load_probe_weights(
            model.probes[name], p / PROBES_DIR / f"{name}.safetensors", model.backend
        )

    if load_steering:
        for name, steering_cfg in manifest.get("steering", {}).items():
            steering_path = p / STEERING_DIR / f"{name}.json"
            if steering_path.exists():
                with open(steering_path) as f:
                    data = json.load(f)
                from auto_chasm.steering import SteeringHook

                hook = SteeringHook.from_dict(data, backend=model.backend.name)
                # Keep the direction/layer overrides restored by from_dict;
                # only refresh method/scale from the manifest's top-level entry.
                steering_config = SteeringConfig(
                    method=steering_cfg.get("method", hook.config.method),
                    scale=steering_cfg.get("scale", hook.config.scale),
                    layer=hook.config.layer,
                    direction=hook.config.direction,
                )
                hook.config = steering_config
                model._steering_hooks[name] = hook
                _reactivate_steering(model, name, hook)
                logger.info("Steering data loaded for probe '%s'.", name)

    logger.info("Checkpoint loaded from %s", p)
    return model


def _reactivate_steering(model: Model, name: str, hook: Any) -> None:
    """Re-wire a restored steering hook into its probe's captures and enable it.

    ``load_checkpoint`` repopulates ``model._steering_hooks`` but, without this,
    never wires the hook into the probe's ``layer_captures`` nor calls
    ``hook.enable()`` — so a model saved with steering ENABLED reloads producing
    the UNSTEERED baseline (a silent save/load divergence). This mirrors the
    enable wiring in ``Model.enable_steering``.

    Args:
        model: The ``Model`` being restored.
        name: Probe name the hook belongs to.
        hook: The restored ``SteeringHook`` (geometry already populated by
            ``from_dict``).
    """
    from auto_chasm.steering import build_auto_steer_fn

    probe = model.probes.get(name)
    if probe is None:
        # Steering was saved for a probe that is not being restored (e.g. a
        # steering-only manifest entry). Nothing to wire — leave the hook
        # registered but inert rather than guessing at a missing probe.
        logger.warning("Steering data for '%s' has no matching probe; hook left inactive.", name)
        return

    built = build_auto_steer_fn(hook)
    if built is None:
        # No reconstructable geometry/custom fn; do not silently fake activation.
        logger.warning(
            "Steering hook for probe '%s' has no usable geometry after load; "
            "steering left disabled.",
            name,
        )
        return

    for capture in probe.layer_captures:
        capture.steer_fn = built
        capture.binary_head = probe.module

    hook.enable()


def _check_probe_reconstructable(name: str, probe_cfg: dict[str, Any]) -> None:
    """Raise if a probe's saved config cannot be rebuilt from JSON.

    A callable ``module_type`` / ``aggregation`` is serialized as the sentinel
    ``"__callable__"`` and a custom pooling (``granularity="custom"``) is not
    serialized at all — neither can be reconstructed from the manifest.  Rather
    than silently loading a degraded probe (custom pooling collapses to
    identity, callable heads/aggregations raise a confusing ``ValueError``),
    fail loudly so the user re-supplies the callable.

    Args:
        name: Probe name (for the error message).
        probe_cfg: The ``manifest["probes"][name]`` dict.

    Raises:
        ValueError: If the probe used a callable that cannot be serialized.
    """
    if probe_cfg.get("module_type") == "__callable__":
        raise ValueError(
            f"Probe '{name}' was saved with a callable module_type, which cannot "
            f"be reconstructed from a checkpoint. Re-supply the callable (e.g. "
            f"build the probe with ProbeConfig(module_type=your_callable) and load "
            f"weights via a hook) instead of loading this checkpoint directly."
        )
    if probe_cfg.get("aggregation") == "__callable__":
        raise ValueError(
            f"Probe '{name}' was saved with a callable aggregation, which cannot "
            f"be reconstructed from a checkpoint. Re-supply the callable aggregation "
            f"(ProbeConfig(aggregation=your_callable)) instead of loading directly."
        )
    if probe_cfg.get("granularity") == "custom":
        raise ValueError(
            f"Probe '{name}' was saved with granularity='custom', whose pooling "
            f"callable is not serializable. Re-supply the pooling callable "
            f"(ProbeConfig(granularity='custom', pooling=your_callable)) instead of "
            f"loading this checkpoint directly."
        )


def _lora_from_manifest(lora_cfg: dict[str, Any]) -> Any:
    """Reconstruct a full ``LoraConfig`` from its manifest dict.

    Restores every field — dropping ``target_layers``/``until_layer``/
    ``after_layer`` or ``peft_method`` would reload, e.g., a DoRA adapter on
    the second half of layers as plain LoRA on all layers, silently
    corrupting the restored adapter set.

    Args:
        lora_cfg: The ``manifest["lora"]`` dict.

    Returns:
        A reconstructed ``LoraConfig``.
    """
    from auto_chasm.config import LoraConfig

    return LoraConfig(
        rank=lora_cfg.get("rank", 8),
        alpha=lora_cfg.get("alpha", 16),
        dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get("target_modules"),
        peft_method=lora_cfg.get("peft_method", "lora"),
        target_layers=lora_cfg.get("target_layers"),
        until_layer=lora_cfg.get("until_layer"),
        after_layer=lora_cfg.get("after_layer"),
    )


def load_training_manifest(checkpoint_dir: str) -> dict[str, Any] | None:
    """Load the training manifest from a checkpoint directory.

    Searches both the directory itself and one level of subdirectories
    for ``training_manifest.json`` (the trainer writes it to its
    ``output_dir``, which may be a subdirectory of the checkpoint).

    Args:
        checkpoint_dir: The checkpoint directory path.

    Returns:
        The training manifest dict, or ``None`` if not found.
    """
    p = Path(checkpoint_dir)

    direct = p / "training_manifest.json"
    if direct.exists():
        with open(direct) as f:
            return json.load(f)  # type: ignore[no-any-return]

    for sub in p.iterdir():
        if sub.is_dir():
            candidate = sub / "training_manifest.json"
            if candidate.exists():
                with open(candidate) as f:
                    return json.load(f)  # type: ignore[no-any-return]

    return None


def _load_adapters(model: Model, path: Path) -> None:
    """Load adapter weights if the file exists.

    Delegates to the backend's ``load_adapters`` for both MLX and
    PyTorch.

    Args:
        model: The ``Model`` instance.
        path: File path for the adapters.
    """
    if not path.exists():
        logger.debug("No adapter weights found at %s", path)
        return

    try:
        model.backend.wrapping.load_adapters(model.model, str(path))
        logger.info("Adapter weights loaded.")
    except Exception as e:
        logger.warning("Could not load adapters: %s", e)


def export_checkpoint(checkpoint_dir: str, output_path: str) -> None:
    """Export a checkpoint directory to a single file.

    Args:
        checkpoint_dir: Checkpoint directory path.
        output_path: Output file path (``.auto_chasm``).

    Raises:
        ValueError: If the checkpoint directory does not contain a manifest.
    """
    import tarfile

    p = Path(checkpoint_dir)
    if not (p / MANIFEST_NAME).exists():
        raise ValueError(f"Directory {p} does not contain a {MANIFEST_NAME}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(p, arcname=p.name)
    logger.info("Checkpoint exported to %s", output_path)


def import_checkpoint(archive_path: str, output_dir: str) -> str:
    """Import a checkpoint from a single file.

    Args:
        archive_path: Path to the ``.auto_chasm`` archive.
        output_dir: Directory to extract to.

    Returns:
        Path to the extracted checkpoint directory.
    """
    import tarfile

    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=str(p), filter="data")
        # Locate the checkpoint by the archive's stored top-level directory
        # rather than by counting ``iterdir()`` entries: any pre-existing file
        # in a non-empty ``output_dir`` would fool the count heuristic into
        # returning the parent dir, which has no manifest and cannot be loaded.
        top_levels = {Path(m.name).parts[0] for m in tar.getmembers() if Path(m.name).parts}

    candidates = [p / top for top in sorted(top_levels)]
    result_dir = next(
        (c for c in candidates if (c / MANIFEST_NAME).exists()),
        None,
    )
    if result_dir is None:
        # Fall back to scanning everything we extracted (e.g. a flat archive
        # whose manifest sits directly under ``output_dir``).
        if (p / MANIFEST_NAME).exists():
            result_dir = p
        else:
            result_dir = next(
                (d for d in p.iterdir() if d.is_dir() and (d / MANIFEST_NAME).exists()),
                None,
            )
    if result_dir is None:
        raise ValueError(
            f"Imported archive {archive_path} does not contain a {MANIFEST_NAME}; "
            f"it is not a valid auto-chasm checkpoint."
        )

    result = str(result_dir)
    logger.info("Checkpoint imported to %s", result)
    return result
