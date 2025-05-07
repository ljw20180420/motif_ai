TODO
```list
use (File-based) Journal Storage instead of RDB
remove duplicate DNA sequences
baseline需要加上24年的briefings bioinformatics和李天杰说的review,以及DeepDF的引用
Use better negative data than random shuffle. Maybe filter by motif?
Model interpretability. (attention)
Baseline model. (DummyClassifier, scikit-learn, lightGBM, DeepZF)
Auto fine-tuning.
Add comment.
Write paper.
```

### 运行流程
```bash
preprocess/run.sh
AI_models/run_bind_transformer.py --command download
AI_models/run_bind_transformer.py --command train
# AI_models/run_DeepZF.sh
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
