TODO
```list
加入所有数据
可变PR, F1的threshold步长
use (File-based) Journal Storage instead of RDB
write model in tensordict
compile tensordict modules
psycopg.errors.UndefinedFunction: operator does not exist: trialvaluetype = character varying,
LINE 3: ...$3::INTEGER ORDER BY CASE trial_values.value_type WHEN $4::V...
HINT:  No operator matches the given name and argument types. You might need to add explicit type casts.
remove duplicate DNA sequences
latent attention
multi-scale attention
logit和概率的AUC可能不一样
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
