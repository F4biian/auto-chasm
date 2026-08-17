"""Shared adapter-key validation for :meth:`Backend.load_adapters`.

A LoRA adapter file is a flat ``{parameter path: tensor}`` mapping, and both
backends load it non-strictly so that a file holding only the LoRA parameters
can be applied to a full model. Non-strict loading also means a key that matches
nothing is silently dropped -- and a *partial* mismatch is far more dangerous
than a total one, because the model still runs and still generates fluent text
from weights that are only partly what was trained.

That is not hypothetical. Attaching a probe REPLACES the block it hooks with a
capture wrapper holding the original as ``.layer``, which moves that block's
parameters from ``model.layers.13.*`` to ``model.layers.13.layer.*``. An adapter
saved from a model with the probe attached therefore carries ``.layer.`` in the
keys for exactly that one block. Loading it into a model whose probe is not yet
attached matched 322 of 336 tensors; the 14 that silently vanished were the
probe layer's, and generation drifted on 61 of 100 prompts with no error
anywhere.

So the rule is: every key in the file must have a home in the model. Extra
parameters in the MODEL are expected and fine (the file is LoRA-only); extra
keys in the FILE are a bug in the caller and are reported as one.
"""

from __future__ import annotations


def check_adapter_keys(
    path: str, file_keys: set[str], model_keys: set[str], strict: bool = True
) -> None:
    """Raise unless every adapter key has a matching model parameter.

    Args:
        path: Adapter file path, for the error message.
        file_keys: Parameter paths present in the adapter file.
        model_keys: Parameter paths present in the model.
        strict: When ``False``, an unmatched key is a warning rather than an
            error. Use only when a partial load is genuinely intended.

    Raises:
        ValueError: If any adapter key is missing from the model (or, when the
            overlap is empty, that the file does not belong to this model at
            all).
    """
    from auto_chasm.logger import get_logger

    if not file_keys:
        return
    missing = file_keys - model_keys
    if not missing:
        return

    sample = sorted(missing)[:3]
    if not (file_keys & model_keys):
        raise ValueError(
            f"Adapter file {path!r} has no parameters matching the model "
            f"(e.g. {sample[0]!r}). The LoRA config or base model does not match "
            "what produced these adapters."
        )

    detail = (
        f"Adapter file {path!r}: {len(missing)} of {len(file_keys)} parameters have no "
        f"match in the model and would be silently dropped, e.g. {sample}.\n"
        "The usual cause is CALL ORDER: attaching a probe replaces the block it hooks "
        "with a capture wrapper, which moves that block's parameters under an extra "
        "'.layer.' path component. Attach every probe BEFORE calling load_adapters, "
        "exactly as the adapters were saved.\n"
        "Pass strict=False to load anyway."
    )
    if strict:
        raise ValueError(detail)
    get_logger(__name__).warning(detail)
