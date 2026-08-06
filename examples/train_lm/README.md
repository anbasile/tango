# Fine-tuning a language model

<!-- start overview -->

This Tango example showcases how you could train or fine-tune a causal language model like [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
or [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj) from [transformers](https://github.com/huggingface/transformers) on WikiText2 or a similar dataset.
It's best that you run this experiment on a machine with a GPU and PyTorch [properly installed](https://pytorch.org/get-started/locally/#start-locally), otherwise Tango will fall back to CPU-only and it will be extremely slow.

<!-- end overview -->

To getting started, just run

```
tango run config.jsonnet -i tokenize_step.py
```
