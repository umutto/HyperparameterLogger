# HyperparameterLogger  
Simple logging wrapper for model hyperparameters from gensim.d2v, sklearn and keras.  
  
---  
### Description  
Hyper-parameter logging library I've created for storing models during training/model tuning. Currently it supports sklearn models, keras models and gensim's doc2vec implementation. Model coverage can be easily extended by adding a new class method.  
Logged parameters and models are saved on a given path as json, yaml or xml (yaml and xml uses [PyYaml](https://pyyaml.org/) and [xmltodict](https://github.com/martinblech/xmltodict) libraries.)

### Setup
```bash
python setup.py sdist
pip install ./dist/HyperparameterLogger-0.2.dev0.tar.gz
```
### Use
```python
... 
# model creation, hyperparameter settings etc...
...
model.fit(x, y)
loss, acc = model.evaluate(test_x, test_y) # optional

from HyperparameterLogger import ModelTracker
tracker = ModelTracker.load_from_keras(model, '/saved_models',
                                       evaluation={'loss': loss, 'accuracy': acc})
# save weights, log parameters and plot the neural network graph.
tracker.log('json').plot().save()

```
  
Saved models in a directory can be pretty printed using Pandas Dataframe.
```python
ModelTracker.get_logs_from_dir('/saved_models', log_suffix='json')
```

---  
**Disclaimer:** *This is a very very simple library I've created to fit my needs. Don't rely on it for your own projects.*
