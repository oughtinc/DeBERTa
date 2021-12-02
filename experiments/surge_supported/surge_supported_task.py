#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from glob import glob
from collections import OrderedDict,defaultdict,Sequence
import copy
import math
from scipy.special import softmax
import numpy as np
import pdb
import os
import sys
import csv

import random
import torch
import re
import ujson as json
from DeBERTa.apps.tasks.metrics import *
from DeBERTa.apps.tasks import EvalData, Task,register_task
from DeBERTa.utils import xtqdm as tqdm
from DeBERTa.data import ExampleInstance, ExampleSet, DynamicDataset,example_to_feature
from DeBERTa.data.example import *
from DeBERTa.utils import get_logger
from DeBERTa.data.example import _truncate_segments
from DeBERTa.apps.models.multi_choice import MultiChoiceModel

logger=get_logger()

__all__ = ["SurgeSupportedTask"]

@register_task(name="SurgeSupported", desc="A dataset of title/abstract pairs and claims, task is to classify as supported or unsupported")
class SurgeSupportedTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'surge_dataset_train.csv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_csv(input_src)
    examples = [ExampleInstance((f'{l[2]}\n{l[3]}', l[4]), self.label2id(l[5])) for l in data]

    examples = ExampleSet(examples)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, extra_data=None, **kwargs):
    ds = [
        self._data('dev', "surge_dataset_val.csv", 'dev'),
        ]

    if extra_data is not None:
      extra_data = extra_data.split(',')
      for d in extra_data:
        n,path=d.split(':')
        ds.append(self._data(n, path, 'dev+'))

    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_csv(input_src)
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((f'{l[2]}\n{l[3]}', l[4])) for l in data])
    else:
      examples = ExampleSet([ExampleInstance((f'{l[2]}\n{l[3]}', l[4]), self.label2id(l[5])) for l in data])

    predict_fn = self.get_predict_fn(examples)
    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy=metric_accuracy(logits, labels))
    return metrics_fn

  def get_predict_fn(self, data):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output = os.path.join(output_dir, 'pred-probs-{}-{}.tsv'.format(name, prefix))
      probs = softmax(logits, axis=-1)
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('sentence1\tsentence2\tnot_entailment\tentailment\n')
        for d,probs in zip(data, probs):
          fs.write(f'{d.segments[0]}\t{d.segments[1]}\t{probs[0]}\t{probs[1]}\n')
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, labels[p]))
    return predict_fn

  @classmethod
  def _read_csv(cls, input_file, quotechar='"'):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
      reader = csv.reader(f, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def get_labels(self):
    """See base class."""
    return ["neutral", "entails"]