import logging
log = logging.getLogger(__name__)

import pathlib
import collections
import pickle
from datetime import datetime
import json
try:
    import yaml
except ImportError:
    print('Could not detect PyYaml library. Install PyYaml ' +
          'if you want to be able to export configurations as .yaml.')
try:
    import xmltodict
except ImportError:
    print('Could not detect xmltodict library. Install xmltodict ' +
          'if you want to be able to export configurations as .xml.')


class ModelTracker(object):
    def __init__(self, directory, name, model, save_func,
                 model_type, config, optimizer, history, **kwargs):
        self._directory = directory
        self._name = name
        self._model = model
        self._model_type = model_type
        self._config = config
        self._optimizer = optimizer
        self._history = history
        self._save_func = save_func
        self._kwargs = kwargs
        self._kwargs['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._set_path()

    def __str__(self):
        return f"Keys that are logged: {', '.join(self.get_dict().keys())}"

    def __repr__(self):
        return json.dumps(self.get_dict(), indent=4)

    def get_dict(self):
        stats = {
            'name': self._name,
            'type': self._model_type,
            'config': self._config,
            'optimizer': self._optimizer,
            'history': self._history
        }
        stats.update(self._kwargs)
        return stats

    def _set_path(self, custom_path=None):
        self._file_path = custom_path

        if self._file_path:
            return self._file_path

        i = 0
        file_name = self._name
        while list(pathlib.Path(self._directory).glob(f'{file_name}.*')):
            i += 1
            file_name = self._name + f'_{i}'

        if i != 0:
            log.warning(
                f'Filename already exists, adding postfix {i} to the given filename')

        self._file_path = self._directory + file_name
        self._kwargs['physical_path'] = self._file_path
        return self._file_path

    def save_model(self):
        file_path = self._file_path + '.' + self._model_type.split('.')[0]
        if self._save_func:
            self._save_func(self._model, file_path)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self._model, f)

        return self

    def save_helper(self, helper, name='helper_lib'):
        file_path = f"{self._file_path}_{name}.pickle"
        with open(file_path, 'wb') as f:
            pickle.dump(helper, f)

        return self

    def plot_keras_graph(self, orientation='LR'):
        file_path = self._file_path + '.png'

        from keras.utils import plot_model
        try:
            plot_model(self._model, file_path,
                       show_shapes=True, rankdir=orientation)
        except ImportError as e:
            log.warning('Error when trying to plot model graph, skipping. ' +
                        'Please install graphviz to export a graph. ' +
                        'Error: ' + e.msg)

        return self

    def log(self, output_format='json'):
        output_format = output_format.lower()
        file_path = self._file_path + '.' + output_format
        with open(file_path, 'w', encoding='utf-8') as f:
            d = self.get_dict()
            if output_format == 'yaml':
                yaml.dump(d, f, indent=4)
            elif output_format == 'json':
                json.dump(d, f, indent=4)
            elif output_format == 'xml':
                f.write(xmltodict.unparse({'root': d}, pretty=True))
            else:
                raise ValueError(f'{output_format} is not a valid choice.')

        return self

    @staticmethod
    def get_logs_from_dir(file_path, name_prefix='', log_suffix='json'):
        log_suffix = log_suffix.lower()
        import pandas as pd

        def get_all_kv_pairs(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    yield from get_all_kv_pairs(v)
                else:
                    yield (k, v)

        log_files = list(pathlib.Path(file_path).glob(
            f'{name_prefix}*.{log_suffix}'))
        param_logs = []
        for log_doc in log_files:
            if log_suffix == 'json':
                l = json.load(open(log_doc))
            elif log_suffix == 'xml':
                l = xmltodict.parse(open(log_doc).read())
            elif log_suffix == 'yaml':
                l = yaml.load(open(log_doc))
            else:
                raise ValueError(f'"{log_suffix}" is not a supported format.')
            param_logs.append(collections.OrderedDict(get_all_kv_pairs(l)))

        return pd.DataFrame(param_logs, index=[l.stem for l in log_files])

    @classmethod
    def load_from_sklearn(cls, model, path='', name='sklearn_model',
                          seed=None, train_time=None, evaluation=None):
        model_type = model.__module__ + '.' + model.__class__.__name__

        config = model.get_params()
        history = {}
        kwargs = {
            'class_name': model.__class__.__name__,
            'seed': seed,
            'train_time': train_time,
            'evaluation': evaluation
        }
        optimizer = {}

        save_func = None

        return cls(path, name, model, save_func, model_type, config,
                   optimizer, history, **kwargs)

    @classmethod
    def load_from_gensim_d2v(cls, model, path='', name='gensim_model',
                             seed=None, evaluation=None):
        model_type = model.__module__ + '.' + model.__class__.__name__

        config = {
            'window': model.window,
            'vector_size': model.vector_size,
            'min_count': model.min_count,
            'max_vocab_size': model.vocabulary.max_vocab_size,
            'sample': model.sample,
            'dm': model.dm,
            'dm_concat': model.dm_concat,
            'dm_tag_count': model.dm_tag_count,
            'hs': model.hs,
            'dbow': model.dbow,
            'dbow_words': model.dbow_words,
            'negative': model.negative,
            'sg': model.sg,
            'cbow_mean': model.cbow_mean,
            'memory': model.estimate_memory(),
            'null_word': model.vocabulary.null_word
        }

        optimizer = {
            'alpha': model.alpha,
            'min_alpha': model.min_alpha,
        }

        training_loss = None
        try:
            training_loss = model.get_latest_training_loss()
        except AttributeError:
            log.warning("Can't find latest training loss.")

        history = {
            'epoch': model.iter,
            'samples': model.corpus_count,
            'batch_words': model.batch_words,
            'train_count': model.train_count,
            'workers': model.workers,
            'training_loss': training_loss
        }

        kwargs = {
            'class_name': model.__class__.__name__,
            'seed': seed,
            'train_time': model.total_train_time,
            'evaluation': evaluation,
            'comment': model.comment
        }

        def save_func(m, p): return m.save(p)

        return cls(path, name, model, save_func, model_type, config,
                   optimizer, history, **kwargs)

    @classmethod
    def load_from_keras(cls, model, path='', name=None,
                        seed=None, train_time=None, evaluation=None):
        name = name or model.name or 'keras_model'
        model_type = model.__module__ + '.' + model.__class__.__name__

        stats = json.loads(model.to_json())

        kwargs = {
            'class_name': stats['class_name'],
            'keras_version': stats['keras_version'],
            'backend': stats['backend'],
            'seed': seed,
            'train_time': train_time,
            'evaluation': evaluation
        }

        stats['config']['input_shape'] = model.input_shape
        stats['config']['output_shape'] = model.output_shape
        config = stats['config']

        optimizer = {'name': model.optimizer.__class__.__name__,
                     'config': model.optimizer.get_config()}

        history_params, history_hist = None, None
        if hasattr(model, 'history'):
            history_params = model.history.params
            history_hist = model.history.history

        history = {'params': history_params,
                   'metric_history': history_hist}

        def save_func(m, p): return m.save(p)

        return cls(path, name, model, save_func, model_type, config,
                   optimizer, history, **kwargs)
