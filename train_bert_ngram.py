# %%
import pandas as pd
import numpy as np
import os
import random
import datetime
import string
import re
import tempfile
import sys

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext tensorboard')
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    METHOD_NAME = 'dev/ngram_bert'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
else:
    METHOD_NAME = sys.argv[1]
    LOG_DIR = "logs/ngram/" + METHOD_NAME

from utils import *
from models import *

tf.config.list_physical_devices(device_type='GPU')

data_dir = '/home/burtenshaw/now/spans_toxic/data/'

MAX_LEN = 200

train = pd.read_pickle(data_dir + "train.bin")
val = pd.read_pickle(data_dir + "val.bin")
test = pd.read_pickle(data_dir + "test.bin")

#%%
train['input_ids'], train['token_type_ids'], train['attn_mask'] = [x.tolist() for x in bert_prep(train.text.to_list(), max_len = MAX_LEN)]
val['input_ids'], val['token_type_ids'], val['attn_mask'] = [x.tolist() for x in bert_prep(val.text.to_list(), max_len = MAX_LEN)]
test['input_ids'], test['token_type_ids'], test['attn_mask'] = [x.tolist() for x in bert_prep(test.text.to_list(), max_len = MAX_LEN)]
#%%

NGRAM_SMALLEST = 50
NGRAM_LARGEST = 100

HPARAMS = [
          hp.HParam('activation', hp.Discrete(['relu'])),
          hp.HParam('batch_size', hp.Discrete([16, 32])),
          hp.HParam('lr', hp.Discrete([2e-5, 5e-5])),
          hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
          hp.HParam('n_layers', hp.Discrete([1,2,3])),
          hp.HParam('model_scale',hp.Discrete([1,2,3])),
          hp.HParam('pre', hp.Discrete(range(NGRAM_SMALLEST, NGRAM_LARGEST))),
          hp.HParam('post', hp.Discrete(range(NGRAM_SMALLEST, NGRAM_LARGEST))),
          hp.HParam('word', hp.Discrete([0])),
          hp.HParam('epochs', hp.Discrete([2]))
          ]

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    AUC(name='auc')
]


with tf.summary.create_file_writer(LOG_DIR).as_default():
    hp.hparams_config(
        hparams=HPARAMS,
        metrics=[hp.Metric(m.name, display_name=m.name) for m in METRICS],
    )

print('logging at :', LOG_DIR)
now = datetime.datetime.now()
tomorrow = now + datetime.timedelta(days=0.5)
runs = 0
#%%

while now < tomorrow:

    hparams = {hp.name : hp.domain.sample_uniform() for hp in HPARAMS}
    
    pre, post, word = hparams['pre'], hparams['post'], hparams['word']

    X_train, y_train = make_BERT_context_data(train, pre = pre, post = post)
    X_val, y_val = make_BERT_context_data(val, pre = pre, post = post)
    X_test, y_test = make_BERT_context_data(test, pre = pre, post = post)

    train_samples = {'X_train' : X_train, 
                     'y_train' : y_train, 
                     'X_val' : X_val, 
                     'y_val' : y_val, 
                     'X_test' : X_test, 
                     'y_test' : y_test}

    # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)
    # param_str = '_'.join(['%s_%s' % (k,v) for k,v in hparams.items()]).replace('.', '')
    param_str = '%s_run_%s' % (METHOD_NAME, runs)

    run_dir = LOG_DIR + '/' + param_str

    callbacks = [hp.KerasCallback(run_dir, hparams),
                TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        
        results = ngram_dual_bert(data = train_samples,
                                    pre_length = pre + 1,
                                    post_length = post + 1,
                                    hparams = hparams, 
                                    callbacks = callbacks, 
                                    metrics = METRICS)

        print('_' * 80)
        # print(ngram_str)
        for k, v in hparams.items():
            print('\t|%s = %s' % (k, v))

        print( ' = ' )

        for metric, score in results.items():
            print('\t|%s : %s' % (metric , score))
            tf.summary.scalar(metric, score, step=1)
        
        print('_' * 80)
    
    print('\n accuracy : %s' % results)

    now = datetime.datetime.now()
    runs += 1

# %%
