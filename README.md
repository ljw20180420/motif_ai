### Install
```bash
pip install -r requirements.txt -r requirements_pytorch.txt
```

### Train
```python
from AI_models.bind_transformer.train import train
train(data_files="test/data.csv")
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
