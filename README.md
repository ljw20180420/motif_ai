TODO
```list
remove duplicate DNA sequences
latent attention
multi-scale attention
logit和概率的AUC可能不一样
baseline需要加上24年的briefings bioinformatics和李天杰说的review
Use better negative data than random shuffle. Maybe filter by motif?
Model interpretability.
Baseline model.
Use proteinBERT.
Auto fine-tuning.
Add comment.
Write paper.
```

### 运行流程
```bash
preprocess/run.sh
AI_models/run_DeepZF.sh
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
