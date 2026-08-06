# Evaluating T0

This example uses the `transformers::run_generation_dataset` step to run the
[T0 model](https://api.semanticscholar.org/CorpusID:239009562). It runs the
[XSum summarization data](https://github.com/EdinburghNLP/XSum), prompted in 10 different ways, and computes
ROUGE scores for all variants. Finally, it computes an overall ROUGE score.

This example uses mostly built-in Tango steps. You will need the `datasets` and `transformers` integrations.
The only custom step in this example is the `RougeScoreStep`, which computes ROUGE scores from the
generated text.

> **Known limitation.** `bigscience/P3` is a loading-script dataset, and 🤗 Datasets dropped support
> for loading scripts in v3. This example therefore does not run against the currently pinned
> `datasets` version. The step graph and the `RougeScoreStep` are still valid as a reference; to run
> it you would need to point `raw_data` at a parquet export of the P3 subsets you want.