# auto-chasm guides

Start with the first guide. It goes from a model to a trained, saved probe in six
steps. The other guides each cover one topic and can be read in any order. Every
code block runs as is: each guide starts with a short setup block.

| Guide | You learn how to |
|---|---|
| [1. Getting started](01-getting-started.md) | go from a model to a trained, saved probe |
| [2. Data and labels](02-data.md) | label texts, chat conversations and single tokens |
| [3. LM token weights](03-lm-token-weights.md) | train the model only on some tokens, or make it unlearn them |
| [4. Probes](04-probes.md) | read other layers, use several probes, build your own head |
| [5. Training](05-training.md) | choose losses, handle class imbalance, stop early, save memory |
| [6. Custom losses](06-custom-losses.md) | write your own loss |
| [7. Generation and steering](07-generation-and-steering.md) | generate text, read probes per token, steer the model |
| [8. LoRA, checkpoints, layer sweeps](08-lora-checkpoints-sweeps.md) | fine-tune with adapters, save runs, find the best layer |
| [9. Mass-mean probes and evaluation](09-mass-mean-and-evaluation.md) | fit probes without training, report AUROC with error bars |
| [10. API reference](10-api-reference.md) | look up any public name |

Runnable scripts are in [`demo/`](../demo/). Back to the [project README](../README.md).
