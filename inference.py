import sys
import os
import time
import pickle
import argparse
import tensorflow as tf
from glob import glob

from model_v1 import Model
from util import *

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='test_seq_data.txt')
parser.add_argument('--train_dir', default='./model')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--user_hidden_units', default=64, type=int)
parser.add_argument('--item_hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=2001, type=int)
parser.add_argument('--num_heads', default=4, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--threshold_user', default=0.08, type=float)
parser.add_argument('--threshold_item', default=0.9, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gpu', default=0, type=int)
# parser.add_argument('--eval_freq', default=10, type=int)
# parser.add_argument('--k', default=3, type=int)

parser.add_argument('--k1', default=10, type=int)
parser.add_argument('--num_cands', default=-1, type=int, help="`-1` to include all items as prediction candidates")
parser.add_argument('--early_stop_epochs', default=50, type=int)
# parser.add_argument('--min_len_for_eval', default=5, type=int)

# for inference
parser.add_argument('--usernum', default=65427, type=int)
parser.add_argument('--itemnum', default=21077, type=int)
parser.add_argument('--fast_infer', action="store_true", help="faster infer computation")


args = parser.parse_args()

if __name__ == "__main__":
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    params = '\n'.join([str(k) + ',' + str(v) 
      for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
    print(params)
    f.write(params)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  data_dir = os.environ['SM_CHANNEL_EVAL']
  output_dir = os.environ['SM_OUTPUT_DATA_DIR']

  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  data_path = os.path.join(data_dir, "test_seq_data.txt")
  output_path = os.path.join(output_dir, "output.csv")

  dataset = prepare_test_data(data_path)
  usernum, itemnum = args.usernum, args.itemnum

  cc = 0.0
  max_len = 0
  for u in dataset:
    cc += len(dataset[u])
    max_len = max(max_len, len(dataset[u]))
  print("Average sequence length: {0}\n".format(cc / len(dataset)))
  print("Maximum length of sequence: {0}\n".format(max_len))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  model = Model(usernum, itemnum, args)
  graph_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  load_vars = []
  for var in graph_vars:
    if "global_step" in var.name: continue
    else: load_vars.append(var)
  saver = tf.train.Saver(var_list=load_vars, max_to_keep=5)
  global_step = tf.train.get_or_create_global_step()

  with tf.Session(config=config) as sess:
    save_path = tf.train.latest_checkpoint(args.train_dir)
    if save_path:
      saver.restore(sess, save_path)
      print("[Restored] {}".format(save_path))
    else:
      raise FileNotFoundError("Checkpoint not found.")

    t0 = time.time()
    predict(model, dataset, args, sess, output_path)

    print("[Done] time = {}".format(time.time() - t0))
    sys.stdout.flush()
