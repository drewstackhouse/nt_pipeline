import re
import pathlib
import argparse

import tensorflow_text as text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

parser = argparse.ArgumentParser()
parser.add_argument('--input','-i', help='input file path', required=True)
args = parser.parse_args()

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

train_src = []
train_tgt = []

with open(f'{args.input}','r') as infile:
  for line in infile:
    src_line, tgt_line = line.split('\t')
    train_src.append(src_line)
    train_tgt.append(tgt_line)

train_src = tf.data.Dataset.from_tensor_slices(train_src)
train_tgt = tf.data.Dataset.from_tensor_slices(train_tgt)

bert_tokenizer_params=dict(lower_case=False)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    vocab_size = 16000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

src_vocab = bert_vocab.bert_vocab_from_dataset(
    train_src.batch(1000).prefetch(2),
    **bert_vocab_args
)

write_vocab_file('src_vocab.txt', src_vocab)

tgt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_tgt.batch(1000).prefetch(2),
    **bert_vocab_args
)

write_vocab_file('tgt_vocab.txt', tgt_vocab)
