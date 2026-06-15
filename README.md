- [TODO](#todo)
- [Version 2](#version-2)
- [Benchmark](#benchmark)
- [Install](#install)
- [Usage](#usage)

# TODO

- 在repo的README.md加入模型改进，benchmark以及pCBS结合数量

- upload COP to hf after training

- model card

- README

# Version 2

In version 1, we sample 3000 ChIP-seq occupancy sites for each protein to mitigate the training pressure. To balance the positive and negative samples, we use the dynamic hard negative sampling. That is, we dynamically select different negative samples for each epoch and tend to select the negative samples with bad model predictions (hard negative samples). However, we find that the dynamic hard negative sampling has the following defects besides that it increases the implementation complexity. Firstly, sampling 3000 ChIP-seq occupancy sites decreases the diversity of positive samples. Secondly, changing the negative samples for each epoch make the training process unstable and hard to converge. Last but not least, the benchmark becomes unfair. Several machine learning frameworks (like SVM, logistic regression, perceptron, passive aggressive classifier) are trained in single epochs. For these models, the negative samples are selected randomly. As a result, these models observe much fewer negative samples compared to models trained in multiple epochs (like COP).

In version 2, we reflect our strategy of negative sampling. We replace the dynamic hard negative sampling by the static random negative sampling. We try to include all ChIP-seq peaks for each protein. If a protein has more than 200,000 peaks, we randomly select 200,000 peaks to control the size of the final training data. For each peak, we randomly select a protein not occupying near the peak to form a negative sample. Besides the simplicity of implementation, this strategy keeps the diversity of the positive samples and stabilizes the training proces. Moreover, it exposes the same negative samples to models trained in single epochs and multiple epochs.

As a tiny point, we replace Adaboost by XGBoost because the latter is the industry standard for performance and is the go-to algorithm for many machine learning competitions. We also use the decision tree and random forest based on XGBoost instead of the implementation from scikit learn. We also include the logistic regression classifier in the new training round.

# Benchmark

![accuracy](paper/benchmark/default_mouse_C2H2_AccuracyMetric.pdf)


# Install

```shell
$ git clone https://github.com/ljw20180420/COP.git
$ cd COP
$ conda env create -p ./.conda -f environment.yml
```

# Usage

```shell
$ conda activate ./.conda
$ ./run.py --help
```
