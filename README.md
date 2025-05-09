TODO
```list
zero-shot learning
contrastive learning
阴性对照（随机蛋白序列, 打乱蛋白顺序, 突变锌指蛋白, 未训练的锌指蛋白, 非锌指蛋白）
baseline需要加上24年的briefings bioinformatics和李天杰说的review,以及DeepDF的引用
Model interpretability. (attention)
Baseline model. (DummyClassifier, scikit-learn, lightGBM, DeepZF)
Auto fine-tuning.
Add comment.
Write paper.
use all data
remove duplicate DNA sequences
Select DNA based on hyper-sensitive sites
Use diff peak rather than random shuffle as negative samples.
```

### 运行流程
```bash
preprocess/run.sh
AI_models/run_bind_transformer.py --command download
AI_models/run_bind_transformer.py --command test
```

### Install
```bash
conda create --name ENVIRONMENT --file conda.yaml
conda create -prefix PATH --file conda.yaml
```

### Train
```python

```
### Test and save pipeline
```python
from AI_models.bind_transformer.test import test
test(data_files="test/data.csv")
```
### Inference
```python
from AI_models.bind_transformer.inference import inference
for output in inference(data_files="test/inference.csv"):
    pass
```
### App
```python
from AI_models.bind_transformer.app import app
app()
```
