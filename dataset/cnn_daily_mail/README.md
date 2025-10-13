---
dataset_info:
  features:
    - name: instruction
      dtype: string
    - name: input
      dtype: string
    - name: output
      dtype: string
license: apache-2.0
task_categories:
- summarization
- text-generation
language:
- en
size_categories:
- 1K<n<10K
---

This dataset is a subset of https://huggingface.co/datasets/cnn_dailymail.

The training set is composed of 2,000 examples of the original training set and the test set is composed of 1,000 examples of the original validation set.

We use the version 1.0.0 of the CNN/DailyMail dataset.
