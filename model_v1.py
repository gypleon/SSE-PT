from modules import *


class Model():
  def __init__(self, usernum, itemnum, args, reuse=tf.AUTO_REUSE,
      is_training = None, u = None, input_seq = None, test_item = None,
      ):
    if args.num_cands < 0:
      num_cands = itemnum
    else:
      num_cands = args.num_cands

    if len(args.expert_paths.split(',')) > 1:
      self.is_training = is_training
      self.u = u
      self.input_seq = input_seq
      self.pos = tf.zeros([args.batch_size, args.maxlen], dtype=tf.int32)
      self.neg = tf.zeros([args.batch_size, args.maxlen], dtype=tf.int32)
      self.test_item = test_item
    else:
      self.is_training = tf.placeholder(tf.bool, shape=())
      self.u = tf.placeholder(tf.int32, shape=(None))
      self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
      self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
      self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
      self.test_item = tf.placeholder(tf.int32, shape=(num_cands))

    pos = self.pos
    neg = self.neg
    mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1) # mask padding positions

    with tf.variable_scope("SASRec", reuse=reuse):
      # sequence embedding, item embedding table
      self.seq, item_emb_table = embedding(self.input_seq,
                         vocab_size=itemnum + 1,
                         num_units=args.item_hidden_units,
                         zero_pad=True,
                         scale=True,
                         l2_reg=args.l2_emb,
                         scope="input_embeddings",
                         with_t=True,
                         reuse=reuse
                         )
      self.item_emb_table = item_emb_table
      # Positional Encoding
      t, pos_emb_table = embedding(
        tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
        vocab_size=args.maxlen,
        num_units=args.item_hidden_units + args.user_hidden_units,
        zero_pad=False,
        scale=False,
        l2_reg=args.l2_emb,
        scope="dec_pos",
        reuse=reuse,
        with_t=True
      )
      #self.seq += t

      # User Encoding. for inference
      u0_latent, user_emb_table = embedding(self.u[0],
                         vocab_size=usernum + 1,
                         num_units=args.user_hidden_units,
                         zero_pad=False,
                         scale=True,
                         l2_reg=args.l2_emb,
                         scope="user_embeddings",
                         with_t=True,
                         reuse=reuse
                         ) # [H_u]
      self.user_emb_table = user_emb_table

      u_latent = embedding(self.u,
                 vocab_size=usernum + 1,
                 num_units=args.user_hidden_units,
                 zero_pad=False,
                 scale=True,
                 l2_reg=args.l2_emb,
                 scope="user_embeddings",
                 with_t=False,
                 reuse=reuse
                 ) # [B, H]
      self.u_latent = tf.tile(tf.expand_dims(u_latent, 1), [1, tf.shape(self.input_seq)[1], 1]) # [B, T, H]

      # Concat item embedding with user embedding
      self.hidden_units = args.item_hidden_units + args.user_hidden_units
      self.seq = tf.reshape(tf.concat([self.seq, self.u_latent], 2),
                  [tf.shape(self.input_seq)[0], -1, self.hidden_units])
      self.seq += t
      # Dropout
      self.seq = tf.layers.dropout(self.seq,
                     rate=args.dropout_rate,
                     training=tf.convert_to_tensor(self.is_training))
      self.seq *= mask

      # Build blocks
      self.attention = []
      for i in range(args.num_blocks):
        with tf.variable_scope("num_blocks_%d" % i):

          # Self-attention
          self.seq, attention = multihead_attention(queries=normalize(self.seq),
                           keys=self.seq,
                           num_units=self.hidden_units,
                           num_heads=args.num_heads,
                           dropout_rate=args.dropout_rate,
                           is_training=self.is_training,
                           causality=not args.bidi_attn,
                           scope="self_attention")
          self.attention.append(attention)
          # Feed forward
          self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                       dropout_rate=args.dropout_rate, is_training=self.is_training)
          self.seq *= mask

      self.seq = normalize(self.seq)
    
    user_emb = tf.reshape(self.u_latent, [tf.shape(self.input_seq)[0] * args.maxlen, 
                        args.user_hidden_units])

    pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen]) # [B * T]
    neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
    pos_emb = tf.nn.embedding_lookup(item_emb_table, pos) # [B * T, H_i]
    neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

    pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units]) # [B * T, H_i + H_u == H]
    neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])

    seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, self.hidden_units]) # [B * T, H]

    # for inference
    test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item) # [I, H_i]
    
    test_user_emb = tf.tile(tf.expand_dims(u0_latent, 0), [num_cands, 1]) # [I, H_u]
    # combine item and user emb
    test_item_emb = tf.reshape(tf.concat([test_item_emb, test_user_emb], 1), [-1, self.hidden_units]) # [num_cands, H]

    if args.fast_infer:
      seq_emb_3d = tf.reshape(seq_emb, [tf.shape(self.input_seq)[0], args.maxlen, self.hidden_units])
      self.test_logits = tf.matmul(seq_emb_3d[:, -1, :], tf.transpose(test_item_emb)) # [B, num_cands]
    else: # leon: NOTE: no need to calculate over the whole length
      self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb)) # [B * T, num_cands], B = 1 
      self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, num_cands]) # [B, T, num_cands]
      self.test_logits = self.test_logits[:, -1, :]

    if len(args.expert_paths.split(',')) > 1:
      self.probs = tf.nn.softmax(self.test_logits) # [B, num_cands]

    # prediction layer
    self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1) # [B * T, H] -> [B * T], NOTE: similar to cos-sim
    self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

    # ignore padding items (0)
    istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
    self.loss = tf.reduce_sum(
      - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
      tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
    ) / tf.reduce_sum(istarget)
    
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss += sum(reg_losses)

    tf.summary.scalar('loss', self.loss)
    self.auc = tf.reduce_sum(
      ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
    ) / tf.reduce_sum(istarget)

    tf.summary.scalar('auc', self.auc)
    # self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.global_step = tf.train.get_global_step()
    self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
    self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    self.merged = tf.summary.merge_all()

  def predict(self, sess, u, seq, item_idx):
    return sess.run(self.test_logits,
            {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})

  def get_probs(self):
    if len(args.expert_paths.split(',')) > 1: return self.probs
    else: return None


class Coordinator():
  def __init__(self, usernum, itemnum, args, num_experts, reuse=tf.AUTO_REUSE):
    self.is_training = tf.placeholder(tf.bool, shape=())
    self.u = tf.placeholder(tf.int32, shape=(None))
    self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
    self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen)) # iid labels
    # self.test_item = tf.placeholder(tf.int32, shape=(itemnum))

    mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1) # mask padding positions

    input_shape = tf.shape(self.input_seq)
    batch_size = input_shape[0]
    outputs = []
    for i in range(num_experts):
      print("[coord] building expert_{}".format(i))
      with tf.variable_scope("expert_{}".format(i), reuse=reuse):
        expert = Model(usernum, itemnum, args,
            is_training = tf.constant(False), 
            u = self.u, input_seq = self.input_seq, test_item = tf.range(1, itemnum+1, dtype=tf.int32),
            )
        # TODO: parallelly decode w/ `batch_size`
        outputs.append(expert.get_probs())

        # outputs.append(tf.get_variable("test", shape=[128, itemnum]))
    # output = tf.stack(outputs, axis=-1) # [B, I, E]

    with tf.variable_scope("coordinator"):
      # self.logits = tf.layers.dense(output, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)) # [B, 1, I]
      for i in range(num_experts):
        expert_weights = tf.get_variable("ex_weights_{}".format(i), shape=[itemnum],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        expert_bias = tf.get_variable("ex_bias_{}".format(i), shape=[itemnum], initializer=tf.zeros_initializer())
        outputs[i] = outputs[i] * expert_weights + expert_bias
      output = tf.stack(outputs, axis=-1) # [B, I, E]
      self.logits = tf.reduce_mean(output, axis=-1) # [B, I]
      self.predictions = tf.nn.softmax(self.logits) # [B, I]

    one_hot_labels = tf.one_hot(self.pos[:, -1] - 1, depth=itemnum, dtype=tf.float32) # [B, I], `pos-1` maps iid to vocab
    self.loss = tf.reduce_sum(-one_hot_labels * tf.log(self.predictions))
    coord_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "coordinator")
    self.global_step = tf.train.get_global_step()
    self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
    self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=coord_vars)

  def predict(self, sess, u, seq, item_idx): # `item_idx` for compatible api
    return sess.run(self.predictions,
            {self.u: u, self.input_seq: seq, self.is_training: False})
