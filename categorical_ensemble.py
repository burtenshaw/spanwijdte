#%%
import pandas as pd
import numpy as np
import os
import random
import datetime
import string
import re
import tempfile
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from tensorboard.plugins.hparams import api as hp

from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # ipython convenience
    mgc = get_ipython().magic
    mgc(u'%load_ext tensorboard')
    mgc(u'%load_ext autoreload')
    mgc(u'%autoreload 2')
    METHOD_NAME = 'dev/categorical_ensemble'
    LOG_DIR = "logs/" + METHOD_NAME
    os.chdir('/home/burtenshaw/now/spans_toxic')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    METHOD_NAME = sys.argv[1]
    LOG_DIR = "logs/categorical/" + METHOD_NAME

os.chdir('/home/burtenshaw/now/spans_toxic')

%load_ext autoreload
%autoreload 2

from results import EvalResults
from utils import *
from models import *

tf.config.list_physical_devices(device_type='GPU')

# %%
''' 

predict word label based on : 

- model prediction for each word in sentence
- model prediction for current word
- lexical toxicity for full sentence
- lexical toxicity for word 

'''

MAX_LEN = 128

# embeddings
ibm = pd.read_pickle('data/predictions/TRAIN_IBM_TOX.bin')
glove = pd.read_pickle('data/predictions/glove.bin')
sentence_bert = pd.read_pickle('data/predictions/sentence_bert.bin')
spacy_vectors = pd.read_pickle('data/predictions/spacy_vectors.bin')
spacy_sentiment = pd.read_pickle('data/predictions/spacy_sentiment.bin')

# baselines
lex = pd.read_pickle('data/predictions/lexical_pred.bin')
spacy_baseline = pd.read_pickle('data/predictions/spacy_baseline.bin')

# categorical
start = pd.read_pickle('data/predictions/start.bin') 
end = pd.read_pickle('data/predictions/end.bin') 
n_span = pd.read_pickle('data/predictions/n_span.bin') 
len_ = pd.read_pickle('data/predictions/len_.bin') 

# word models
ngram_bert = pd.read_pickle('data/predictions/ngram_bert.bin')
ngram_glove_lstm = pd.read_pickle('data/predictions/ngram_glove_lstm.bin')

# span models
span_bert = pd.read_pickle('data/predictions/span_bert.bin') 

#%% GROUP word level PREDICTIONS to sentence level models
ngram_bert = ngram_bert.groupby(level=0).pred.apply(np.array).apply(pad_mask, max_len =MAX_LEN)
ngram_glove_lstm = ngram_glove_lstm.groupby(level=0).pred.apply(np.array).apply(pad_mask, max_len =MAX_LEN)

#%% align all indicies
lex = lex[['sentence_pred','lexical','label']].loc[muse.index.drop_duplicates()].rename(columns={'lexical':'pred'})
# Horizontally 
# df = pd.concat([muse,lex], axis= 1, keys = ['muse', 'lex'])

word_level = [
    glove,
    sentence_bert,
    spacy_vectors,
    spacy_sentiment,
    lex,
    spacy_baseline,
    start,
    end,
    ngram_bert,
    ngram_glove_lstm,
    span_bert,
]

sentence_level = [
    ibm,
    len_,
    n_span,
]


#%%

X_word = np.hstack([np.vstack(word_predictions) for word_predictions in word_level])
X_sentence = np.hstack(sentence_level)

X = np.hstack(X_word, X_sentence)
y = pd.read_pickle('data/data.bin')['word_mask']

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)

# %% RANDO FOREST
clf = RandomForestClassifier(max_depth=40, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
ensemble_y_pred = clf.predict_proba(X_test)
#%%

def ngram_glove_lstm(data, word_seq_len, sentence_seq_len, hparams, callbacks, metrics):  

    word = tf.keras.Input(shape=(word_seq_len,), dtype="int64")
    sentence = tf.keras.Input(shape=(sentence_seq_len,), dtype="int64")
    
    word_embedding = layers.Embedding(MAX_LEN, 
                        len(word_level), 
                        input_length = word_seq_len, trainable=True)

    sentence_embedding = layers.Embedding(len(sentence_level), 
                        1,
                        input_length = sentence_seq_len, trainable=True)
    
    word_embedded = word_embedding(word)
    sentence_embedded = sentence_embedding(sentence)

    merged = tf.keras.layers.concatenate([word_embedded, sentence_embedded], axis=1)
    input_length = word_seq_len + sentence_seq_len

    layer =  layers.Bidirectional(layers.LSTM(input_length*hparams['model_scale']))(merged)

    model_scale = hparams['model_scale']

    for _ in range(hparams['n_layers']):
        layer = layers.Dense(input_length*hparams['model_scale'], activation=hparams['activation'])(layer)
        layer = tf.keras.layers.Dropout(hparams['dropout'])(layer)
        model_scale = model_scale / 2

    output = layers.Dense(MAX_LEN, activation='softmax')(layer)

    model = tf.keras.Model(
        inputs=[word, sentence],
        outputs=[output],
    )

    opt = Adam(lr = hparams['lr'])

    model.compile(optimizer = opt, 
                  loss = 'categorical_crossentropy', 
                  metrics = metrics)

    model.fit(  data['X_train'] , 
                data['y_train'],
                batch_size=hparams['batch_size'],
                validation_split=0.2,
                epochs=hparams['epochs'],
                verbose = 1,
                callbacks= callbacks)

    scores = model.evaluate(data['X_test'], data['y_test'], return_dict = True)

    return scores


#%%

HPARAMS = [
          hp.HParam('activation', hp.Discrete(['relu'])),
          hp.HParam('batch_size', hp.Discrete([8,16,32])),
          hp.HParam('lr', hp.Discrete([2e-5, 5e-5, 7e-5])),
          hp.HParam('dropout',hp.RealInterval(0.1, 0.4)),
          hp.HParam('n_layers', hp.Discrete([1,2])),
          hp.HParam('model_scale',hp.Discrete([1,2])),
          hp.HParam('epochs', hp.Discrete([10])),
          ]

METRICS = [
          tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
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
    
    train_samples = {'X_train' : X_train, 
                     'y_train' : y_train, 
                     'X_test' : X_test, 
                     'y_test' : y_test}

    # ngram_str = '/pr_%s_w_%s_po_%s_' % (pre, word, post)
    # param_str = '_'.join(['%s_%s' % (k,v) for k,v in hparams.items()]).replace('.', '')
    param_str = '%s_run_%s' % (METHOD_NAME, runs)
    run_dir = LOG_DIR + '/' + param_str

    callbacks = [hp.KerasCallback(run_dir, hparams),
                TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
                EarlyStopping(patience=2)]

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        
        results = categorical_bert(data = train_samples,
                               input_length = MAX_LEN,
                               output_length = MAX_LEN,
                               hparams = hparams, 
                               callbacks = callbacks, 
                               metrics = METRICS,
                               loss = 'categorical_crossentropy')

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



results_df = df.loc[test_index]
results_df['ensemble', 'pred'] = np.amax(ensemble_y_pred, -1)
_data = pd.read_pickle("data/train.bin").loc[test_index]

component_models = [('muse', 0.5), ('lex', 0), ('ensemble', 0.01)]

r = EvalResults(component_models, results_df , params = {'muse' : {'lr' : 0, 'activation' : 'relu'}})
r.rdf# %%

# %%
