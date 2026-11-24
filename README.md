# Do Vision-Language Models Follow Written Rules?

**Testing whether vision-language models can apply written rules to measurements shown in charts and tables.**

![Status: In Development](https://img.shields.io/badge/Status-In%20Development-orange)
![Target: ACL or EMNLP](https://img.shields.io/badge/Target-ACL%20%2F%20EMNLP-blue)

---

## Project objective

This project investigates whether vision-language models can combine a written rule with visual evidence to make the correct decision. We vary chart measurements and rules independently to test when a model should change its answer. Irrelevant chart edits and equivalent rule paraphrases test when the answer should stay the same. We also supply verified measurements or rules to help diagnose errors and compare direct answering with a method that extracts both inputs before executing the rule.

## Datasets

- **Controlled charts and tables** contain 32 pilot scenarios and 384 held-out test scenarios. Each scenario has six conditions and an English and Bangla version of the written rule. The chart and question are held fixed across languages.
- **ChartQA** is a candidate source for replication on unmodified public charts. The replication subset will be audited before evaluation.

## Models

Planned main checkpoints, subject to pilot validation:

- **Qwen2.5-VL-3B-Instruct**
- **Gemma 3 4B IT** for cross-family evaluation in English and Bangla. Repository integration is pending.
- **SmolVLM2-2.2B-Instruct** as an optional English baseline.

## Status

The project is under active development and targets a future ACL or EMNLP submission. 
