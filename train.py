'''
modified from https://github.com/wuliwei9278/SSE-PT
leon updated:
  1. early stopping, training efficiency, code style
  2. impl inference.py
  3. mixture of experts
  4. improve negative sampling / data augmentation (simulate test setting)
'''

import sys
import os
import time
import pickle
import argparse
import tensorflow as tf
# from tqdm import tqdm
from glob import glob

from sampler import WarpSampler
from model_v1 import Model, Coordinator
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='train_data')
parser.add_argument('--train_dir', default='./models/test')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--user_hidden_units', default=64, type=int)
parser.add_argument('--item_hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=2001, type=int)
parser.add_argument('--num_heads', default=4, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--threshold_user', default=0.9, type=float)
parser.add_argument('--threshold_item', default=0.9, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--eval_freq', default=10, type=int)
parser.add_argument('--k', default=3, type=int)

# leon: updated as follows
parser.add_argument('--k1', default=10, type=int)
parser.add_argument('--num_cands', default=-1, type=int, help="`-1` to include all items as prediction candidates")
parser.add_argument('--early_stop_epochs', default=50, type=int)
parser.add_argument('--min_len_for_eval', default=5, type=int)
parser.add_argument('--best_res_log', default='best_result')
parser.add_argument('--max_users_to_eval', default=10000, type=int)

parser.add_argument('--fast_infer', action="store_true", help="faster infer computation")
parser.add_argument('--std_test', action="store_true", help="train & test over std tasks")
parser.add_argument('--with_test', action="store_true", help="prepare test set")
parser.add_argument('--bidi_attn', action="store_true", help="bidirectional attn")
parser.add_argument('--start_delay', default=0, type=int, help="start training after n seconds")

parser.add_argument('--expert_paths', default="", help="comma splitted")
parser.add_argument('--eval_tgt_idx', default=0, type=int, help="which metric as eval target")

parser.add_argument('--random_aug_long', action="store_true", help="augment data from long seqs")
parser.add_argument('--harder_neg_samp', action="store_true", help="negative samples from history")

args = parser.parse_args()


def prepare_env():
  for i in range(args.start_delay, 0, -1):
    time.sleep(1)
    if i % 1800 == 0:
      print("{} seconds to start training".format(i))
      sys.stdout.flush()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    params = '\n'.join([str(k) + ',' + str(v) 
      for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
    print(params)
    f.write(params)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def load_dataset():
  if args.std_test:
    dataset = data_partition_ust(args.dataset, args.min_len_for_eval, args.with_test)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
  else:
    dataset = data_partition_ust(args.dataset, args.min_len_for_eval)
    [user_train, user_valid, usernum, itemnum] = dataset

  num_batch = len(user_train) // args.batch_size
  cc = 0.0
  max_len = 0
  for u in user_train:
    cc += len(user_train[u])
    max_len = max(max_len, len(user_train[u]))
  print("\nThere are {0} users {1} items \n".format(usernum, itemnum))
  print("Average sequence length: {0}\n".format(cc / len(user_train)))
  print("Maximum length of sequence: {0}\n".format(max_len))

  return dataset, user_train, usernum, itemnum, num_batch


def create_model(usernum, itemnum, args):
  model_paths = args.expert_paths.split(',')
  num_experts = len(model_paths)
  print("[num_experts] {}".format(num_experts))
  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.train.get_or_create_global_step()
    if num_experts > 1:
      model = Coordinator(usernum, itemnum, args, num_experts)
    else:
      model = Model(usernum, itemnum, args)

    if num_experts > 1:
      coord_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "coordinator")
      for var in coord_vars:
        print("[coord_var] {} = {}".format(var.name, var.shape))
      saver = tf.train.Saver(var_list=coord_vars, max_to_keep=5)
    else:
      saver = tf.train.Saver(max_to_keep=5)

    # if num_experts > 1:
    #   for i in range(num_experts): # restore experts' variables
    #     scope = "expert_{}".format(i)
    #     variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)}
    #     print("[created graph variables]")
    #     for vname in variables:
    #       print("  {}".format(vname))
    #       sys.stdout.flush()
  return graph, model, num_experts, model_paths, global_step, saver


def restore_collection(path, scope, sess, graph):
  '''
  args:
    path: checkpoint file
  '''
  with graph.as_default():
    print("[restoring experts' params]")
    # variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)}
    variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)}
    # print("[graph variables]")
    # for vname in variables:
    #   print("  {}".format(vname))
    #   sys.stdout.flush()
    for var_name, _ in tf.contrib.framework.list_variables(path):
      if "Adam" in var_name or "beta1_power" in var_name or "beta2_power" in var_name or "global_step" in var_name: continue
      var_value = tf.contrib.framework.load_variable(path, var_name)
      target_var_name = '%s/%s:0' % (scope, var_name)
      target_variable = variables[target_var_name]
      sess.run(target_variable.assign(var_value))
      print("  {} -> {}".format(var_name, target_var_name))
      sys.stdout.flush()


def main():

  prepare_env()

  dataset, user_train, usernum, itemnum, num_batch = load_dataset()

  sampler = WarpSampler(user_train, usernum, itemnum, 
        args=args,
        batch_size=args.batch_size, maxlen=args.maxlen,
        threshold_user=args.threshold_user, 
        threshold_item=args.threshold_item,
        n_workers=3,
        )

  graph, model, num_experts, expert_paths, global_step, saver = create_model(usernum, itemnum, args)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.Session(config=config, graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    if num_experts > 1:
      for i, path in enumerate(expert_paths): # restore experts' variables
        restore_collection(path, "expert_{}".format(i), sess, graph)

    best_result = 0.0
    best_res_path = os.path.join(args.train_dir, args.best_res_log)
    if os.path.isfile(best_res_path):
      with open(best_res_path, 'r') as inf:
        best_result = float(inf.readline().strip())
    best_step = 0
    no_improve = 0
    save_path = tf.train.latest_checkpoint(args.train_dir)
    if save_path:
      saver.restore(sess, save_path)
      print("[restored] {}".format(save_path))
    else:
      save_path = saver.save(sess, os.path.join(args.train_dir, "model.ckpt"), global_step)
      print("[saved] {}".format(save_path))

    T = 0.0
    t0 = time.time()
    t_valid = evaluate_valid(model, dataset, args, sess)
    print("[init] time = {}, best = {}, eval HR@{} = {}, HR@{} = {}],".format(time.time() - t0, best_result, args.k, t_valid[0], args.k1, t_valid[1]))
    if args.std_test:
      t0 = time.time()
      t_test = evaluate(model, dataset, args, sess)
      print("[init] time = {}, test NDCG{} = {}, NDCG{} = {}, HR{} = {}, HR{} = {}]".format(time.time() - t0, args.k,
        t_test[0], args.k1, t_test[1], args.k, t_test[2], args.k1, t_test[3]))

    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
      # for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
      total_loss = 0.0
      for step in range(num_batch):
        u, seq, pos, neg = sampler.next_batch()
        if num_experts > 1:
          log_freq = 1000
          loss, _, global_step_val = sess.run(
              [model.loss, model.train_op, global_step],
              {model.u: u, model.input_seq: seq, model.pos: pos, model.is_training: True})
          if step % log_freq == 0:
            print("[step-{}] {}/{}, avg_loss = {}".format(global_step_val, step+1, num_batch, total_loss/log_freq))
            total_loss = 0.0
          else:
            total_loss += loss
        else:
          user_emb_table, item_emb_table, attention, auc, loss, _, global_step_val = sess.run(
              [model.user_emb_table, model.item_emb_table, model.attention, model.auc, model.loss, model.train_op, global_step],
              {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg, model.is_training: True})
          print("[step-{}] {}/{}, auc = {}, loss = {}".format(global_step_val, step+1, num_batch, auc, loss))
        sys.stdout.flush()

      if epoch % args.eval_freq == 0:
        t1 = time.time()
        T += t1 - t0
        # t_test = evaluate(model, dataset, args, sess)
        t_valid = evaluate_valid(model, dataset, args, sess)
        t2 = time.time()
        # print("[{0}, {1}, {2}, {3}, {4}, {5}],".format(epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
        print("[epoch = {}, time = {} (train/eval = {}/{}), HR@{} = {}, HR@{} = {}],".format(epoch, T, t1-t0, t2-t1, args.k, t_valid[0], args.k1, t_valid[1]))
        t0 = t2

        # early stopping
        if t_valid[args.eval_tgt_idx] > best_result:
          print("[best_result] {} (step-{}) < {} (step-{})".format(best_result, best_step, t_valid[args.eval_tgt_idx], global_step_val))
          best_result = t_valid[args.eval_tgt_idx]
          best_step = global_step_val
          # ckpt_paths = glob(os.path.join(args.train_dir, "model.ckpt*"))
          # for path in ckpt_paths:
          #   os.remove(path)
          #   print("[removed] {}".format(path))
          with open(best_res_path, 'w') as outf:
            outf.write("{}".format(best_result))
          save_path = saver.save(sess, os.path.join(args.train_dir, "model.ckpt"), global_step_val)
          print("[saved] {}".format(save_path))
          no_improve = 0
        else:
          print("[best_result] {} (step-{}) > {} (step-{})".format(best_result, best_step, t_valid[args.eval_tgt_idx], global_step_val))
          no_improve += args.eval_freq
          if no_improve >= args.early_stop_epochs:
            print("[stop training] no improvement for {} epochs".format(no_improve))
            break
        sys.stdout.flush()

    if args.std_test:
      t_test = evaluate(model, dataset, args, sess)
      print("[final] time = {}, test NDCG{} = {}, NDCG{} = {}, HR{} = {}, HR{} = {}]".format(time.time() - t0, args.k,
        t_test[0], args.k1, t_test[1], args.k, t_test[2], args.k1, t_test[3]))

  sampler.close()
  print("[Done]")

if __name__ == "__main__":
  main()
