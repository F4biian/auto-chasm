# 1. Getting started

This guide takes you once through the whole workflow, in six steps. Run the code blocks
in order, in one Python session.

[All guides](README.md) · Next: [2. Data and labels](02-data.md)

## Step 1: Load a model

```python
from auto_chasm import Model

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(model.generate("The capital of France is", max_tokens=5))
```

Any standard Hugging Face language model works, for example Llama, Qwen or Gemma.
The backend (MLX or PyTorch) is picked automatically. To choose one yourself, pass
`backend_name="mlx"` or `backend_name="torch"`.

## Step 2: Label some data

A probe learns from examples. Here, each text gets one label: `1` for a question,
`0` for a statement.

```python
from auto_chasm import Dataset

texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]

data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)
```

Keep `append_eos=True` whenever you have one label per text. It makes sure the probe
learns from a state that has seen the whole text.

## Step 3: Attach a probe

```python
from auto_chasm import ProbeConfig

model.attach_probe(ProbeConfig(name="question", layers=[-1]))
model.prepare_for_joint_training()
```

The probe is called `"question"`, like the labels, and reads the last layer (`-1`).
`prepare_for_joint_training` freezes the base model so that only the probe (and LoRA
adapters, if you use them) will train.

## Step 4: Train

```python
from auto_chasm import JointLoss, Trainer

loss = JointLoss(weights={"lm_head": 0.0, "question": 1.0})
Trainer(model=model, loss_fn=loss, num_iters=40, batch_size=4).train(data)
```

`JointLoss` has one weight per part: `"lm_head"` is the language-model loss, and
every probe has its own entry. With `"lm_head": 0.0`, only the probe learns.

## Step 5: Read the probe

On a fixed text:

```python
ids = model.tokenizer.encode("Do you like music?")
print(model.forward([ids]).probes["question"].logits[0, -1].item())
```

Or token by token while the model generates:

```python
for step in model.generate_with_probes("Tell me, do you", max_tokens=5):
    score = step.probes["question"].logits[0, -1].item()
    print(f"{step.token_str!r:12} {score:+.2f}")
```

## Step 6: Steer and save

Steering pushes the model's activations along the direction the probe has learned.
It needs the average activation of each class, which `compute_class_means` collects:

```python
from auto_chasm import SteeringConfig

class_means = model.compute_class_means(data)
model.enable_steering("question", config=SteeringConfig(method="nullify"), class_means=class_means)
print(model.generate("Tell me, do you", max_tokens=8))
model.disable_steering("question")
```

Finally, save everything and load it again:

```python
model.save_checkpoint("runs/question-probe")
restored = Model.from_checkpoint("runs/question-probe")
```

## Next steps

- Label chat data or single tokens: [2. Data and labels](02-data.md)
- Use several probes, other layers or your own head: [4. Probes](04-probes.md)
- Train the model too, with LoRA: [8. LoRA, checkpoints, layer sweeps](08-lora-checkpoints-sweeps.md)
- Write your own loss: [6. Custom losses](06-custom-losses.md)
