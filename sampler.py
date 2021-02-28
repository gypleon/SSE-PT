import random
import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
  '''
  '''
  t = np.random.randint(l, r)
  while t in s:
    t = np.random.randint(l, r)
  return t


def random_neighbor(cands, truth, rand_false):
  t = np.random.choice(cands)
  n_fails = 0
  while t == truth:
    t = np.random.choice(cands)
    n_fails += 1
    if n_fails > 9:
      t = rand_false
      break
  return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen,  
          threshold_user, threshold_item,
          result_queue, SEED,
          random_aug_long, harder_neg_samp):
  def sample():
    ''' ret: (uid, iids, iid_pos_labels, iid_neg_labels)
    '''
    user = np.random.randint(1, usernum + 1)
    while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

    # NOTE: `0` as heading padding
    seq = np.zeros([maxlen], dtype=np.int32) # perturbed item seq
    pos = np.zeros([maxlen], dtype=np.int32) # perturbed positive labels (the next item for autoregression)
    neg = np.zeros([maxlen], dtype=np.int32) # random negative labels

    input_seq = user_train[user]
    neighbors = set(input_seq)
    neg_cands = list(neighbors)
    rand_neg = random.sample(set([i for i in range(1, itemnum+1)])-neighbors, 1)[0] # guarantee at least one differfing from the truth
    if random_aug_long:
      len_input_seq = len(input_seq)
      if len_input_seq > maxlen:
        start_idx = np.random.randint(0, len_input_seq-maxlen+1)
        input_seq = input_seq[start_idx:start_idx+maxlen]

    nxt = input_seq[-1]
    idx = maxlen - 1

    ts = set(input_seq)

    for i in reversed(input_seq[:-1]):
      # SSE: stochastic shared embeddings -> regularization
      # SSE for user side (2 lines)
      if random.random() > threshold_item: # apply SSE to a user's item seq
        i = np.random.randint(1, itemnum + 1)
        # nxt = np.random.randint(1, itemnum + 1) # leon: keep targets always true
      seq[idx] = i
      pos[idx] = nxt
      if harder_neg_samp:
        if nxt != 0: neg[idx] = random_neighbor(neg_cands, nxt, rand_neg)
      else:
        if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
      nxt = i
      idx -= 1
      if idx == -1: break
    
    # SSE for item side (2 lines)
    # equivalent to hard parameter sharing
    if random.random() > threshold_user:
      user = np.random.randint(1, usernum + 1)
   
    return (user, seq, pos, neg)

  np.random.seed(SEED)
  while True:
    one_batch = []
    for i in range(batch_size):
      one_batch.append(sample())

    result_queue.put(zip(*one_batch)) # zip along timesteps


class WarpSampler(object):
  def __init__(self, User, usernum, itemnum, args, batch_size=64, maxlen=10, 
         threshold_user=1.0, threshold_item=1.0, n_workers=1):
    self.result_queue = Queue(maxsize=n_workers * 10)
    self.processors = []
    for i in range(n_workers):
      self.processors.append(
        Process(target=sample_function, args=(User,
                            usernum,
                            itemnum,
                            batch_size,
                            maxlen,
                            threshold_user,
                            threshold_item,
                            self.result_queue,
                            np.random.randint(2e9),
                            args.random_aug_long,
                            args.harder_neg_samp,
                            )))
      self.processors[-1].daemon = True
      self.processors[-1].start()

  def next_batch(self):
    return self.result_queue.get()

  def close(self):
    for p in self.processors:
      p.terminate()
      p.join()
