import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition_ust(fname, min_len_for_eval):
  '''leon: adapt to ust's data format, remove test set
  '''
  # NOTE: usernum/itemnum might be sparse? NO
  usernum = 0
  itemnum = 0
  User = {}
  user_train = {}
  user_valid = {}
  with open('data/%s.txt' % fname, 'r') as inf:
    for line in inf:
      uid, iids = line.rstrip().split(':')
      uid = int(uid)
      iids = iids.split(',')
      iids = [int(iid) for iid in iids]
      User[uid] = iids
      usernum = max(uid, usernum)
      itemnum = max(iids + [itemnum])

  for uid, items in User.items():
    nfeedback = len(items)
    if nfeedback < min_len_for_eval: # NOTE: to keep enough training data
      user_train[uid] = items
      user_valid[uid] = []
    else:
      user_train[uid] = items[:-1]
      user_valid[uid] = [items[-1]]
  return [user_train, user_valid, usernum, itemnum]


def prepare_test_data(path):
  '''leon: prepare testing data
  '''
  user_test = {}
  with open(path, 'r') as inf:
    for line in inf:
      uid, iids = line.rstrip().split(':')
      uid = int(uid)
      iids = iids.split(',')
      iids = [int(iid) for iid in iids]
      user_test[uid] = iids

  print("[prepare_test_data] loaded {} users from {}".format(len(user_test), path))
  return user_test


def evaluate(model, dataset, args, sess):
  '''final testing
  '''
  [train, valid, usernum, itemnum] = copy.deepcopy(dataset)

  NDCG = 0.0
  HT = 0.0
  valid_user = 0.0

  if usernum > 10000: # TODO: configurable
    users = random.sample(range(1, usernum + 1), 10000)
  else:
    users = range(1, usernum + 1)
  for u in users:

    if len(train[u]) < 1 or len(valid[u]) < 1: continue # TODO: set dataset

    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    seq[idx] = valid[u][0]
    idx -= 1
    for i in reversed(train[u]):
      seq[idx] = i
      idx -= 1
      if idx == -1: break
    rated = set(train[u])
    rated.add(0)
    item_idx = [valid[u][0]] # TODO: set dataset
    for _ in range(100):
      t = np.random.randint(1, itemnum + 1)
      while t in rated: t = np.random.randint(1, itemnum + 1)
      item_idx.append(t)

    predictions = -model.predict(sess, [u], [seq], item_idx)
    predictions = predictions[0]
    #print(predictions)
    rank = predictions.argsort().argsort()[0]

    valid_user += 1

    if rank < args.k:
      NDCG += 1 / np.log2(rank + 2)
      HT += 1
    if valid_user % 1000 == 0:
      #print '.',
      sys.stdout.flush()

  return NDCG / valid_user, HT / valid_user


def predict(model, dataset, args, sess, outpath):
  print("[pred] started")
  users = copy.deepcopy(dataset)
  itemnum = args.itemnum

  with open(outpath, 'w') as outf:
    for num_tested, (uid, items) in enumerate(users.items()):
      seq = np.zeros([args.maxlen], dtype=np.int32)
      idx = args.maxlen - 1
      for i in reversed(items): # based on the last `maxlen` items
        seq[idx] = i
        idx -= 1
        if idx == -1: break

      item_idx = [i for i in range(1, itemnum + 1)]

      predictions = -model.predict(sess, [uid], [seq], item_idx) # smaller -> more likely
      predictions = predictions[0] # [itemnum] NOTE: items-indices = [1, itemnum]-[0, itemnum-1]

      top_items = predictions.argsort()[:args.k1] + 1 # top-k indices. `+1` correct index offset

      outf.write("{}\n".format(top_items.tolist()))

      if num_tested % 100 == 0:
        print("[pred] {}/{}".format(num_tested+1, len(users)))
        sys.stdout.flush()

  return num_tested+1


def evaluate_valid(model, dataset, args, sess):
  print("[eval] started")
  [train, valid, usernum, itemnum] = copy.deepcopy(dataset)

  # NDCG = 0.0
  valid_user = 0.0
  HT = 0.0
  HT1 = 0.0
  max_users_to_eval = 10000 # TODO: configurable
  if usernum > max_users_to_eval:
    users = random.sample(range(1, usernum + 1), max_users_to_eval)
  else:
    users = range(1, usernum + 1)
  for u in users:
    if len(train[u]) < 1 or len(valid[u]) < 1: continue # NOTE: around half of users has no valid data (short item list)

    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    for i in reversed(train[u]): # based on the last `maxlen` items
      seq[idx] = i
      idx -= 1
      if idx == -1: break

    item_idx = [valid[u][0]] # downsampled candidate item set, to accelerate eval
    item_cands = set([i for i in range(1, itemnum + 1)])
    if args.num_cands < 0: # predict over all items
      rated = set([item_idx[0]])
      item_cands -= rated
      # args.num_cands = itemnum
      item_idx.extend(item_cands)
    else:
      rated = set(train[u])
      rated.add(0)
      '''
      for _ in range(args.num_cands):
        t = np.random.randint(1, itemnum + 1)
        while t in rated:
          t = np.random.randint(1, itemnum + 1)
        rated.add(t)
        item_idx.append(t) # add random candidate items which have yet appeared for this user
      '''
      item_cands -= rated
      item_idx.extend(random.sample(item_cands, args.num_cands))

    predictions = -model.predict(sess, [u], [seq], item_idx) # `-`: max -> min
    predictions = predictions[0] # [args.num_cands]

    rank = predictions.argsort().argsort()[0] # rank of the ground truth item

    valid_user += 1

    if rank < args.k:
      # NDCG += 1 / np.log2(rank + 2)
      HT += 1
    if rank < args.k1:
      HT1 += 1

    if valid_user % 1000 == 0:
      print("[eval] {}/{}".format(int(valid_user), len(users)))
      sys.stdout.flush()

  print("[eval] {}/{}".format(int(valid_user), len(users)))
  sys.stdout.flush()

  return HT / valid_user, HT1 / valid_user
