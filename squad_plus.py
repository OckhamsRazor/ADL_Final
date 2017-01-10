import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf

import squad_input

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 32
    test_batch_size = 32
    #embed_size = 80
    #embed_size = 80
    embed_size = 100
    hidden_size = 100
    choice_num = 4

    max_epochs = 256
    early_stopping = 20

    dropout = 0.55
    lr = 0.01
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    #word2vec_init = False
    word2vec_init = True
    embedding_init = 1.7320508 # root 3

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    drop_grus = False

    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 2
    num_attention_features = 4

    max_allowed_inputs = 100
    num_train = 650
    #num_train = 28000

    floatX = np.float32

    babi_id = "6"
    babi_test_id = ""

    test_file = "data/test.json"

    train_mode = True

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

    # TODO fix positional encoding so that it varies according to sentence lengths

def _xavier_weight_init():
    """Xavier initializer for all variables except embeddings as desribed in [1]"""
    def _xavier_initializer(shape, **kwargs):
        eps = np.sqrt(6) / np.sqrt(np.sum(shape))
        out = tf.random_uniform(shape, minval=-eps, maxval=eps)
        return out
    return _xavier_initializer

# from https://danijar.com/variable-sequence-lengths-in-tensorflow/
# used only for custom attention GRU as TF handles this with the sequence length param for normal RNNs
def _last_relevant(output, length):
    """Finds the output at the end of each input"""
    batch_size = int(output.get_shape()[0])
    max_length = int(output.get_shape()[1])
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
    

class DMN_PLUS(object):

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            #self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size, self.max_z_len = squad_input.load_babi(self.config, split_sentences=True)
        else:
            #self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size,self.max_z_len = squad_input.load_babi(self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        #self.answer_placeholder = tf.placeholder(tf.float32, shape=(self.config.batch_size,))
        self.answer_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

        ##
        self.choice_placeholder = tf.placeholder(tf.int32, shape=(4,self.config.batch_size, self.max_z_len))
        self.choice_len_placeholder = tf.placeholder(tf.int32, shape=(4,self.config.batch_size,))
        ##

    def add_reused_variables(self):
        """Adds trainable variables which are later reused""" 
        gru_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)

        # apply droput to grus if flag set
        if self.config.drop_grus:
            self.gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
        else:
            self.gru_cell = gru_cell

        with tf.variable_scope("memory/attention", initializer=_xavier_weight_init()):
            b_1 = tf.get_variable("bias_1", (self.config.embed_size,))
            W_1 = tf.get_variable("W_1", (self.config.embed_size*self.config.num_attention_features, self.config.embed_size))

            W_2 = tf.get_variable("W_2", (self.config.embed_size, 1))
            b_2 = tf.get_variable("bias_2", 1)

        with tf.variable_scope("memory/attention_gru", initializer=_xavier_weight_init()):
            Wr = tf.get_variable("Wr", (self.config.embed_size, self.config.hidden_size))
            Ur = tf.get_variable("Ur", (self.config.hidden_size, self.config.hidden_size))
            br = tf.get_variable("bias_r", (1, self.config.hidden_size))

            W = tf.get_variable("W", (self.config.embed_size, self.config.hidden_size))
            U = tf.get_variable("U", (self.config.hidden_size, self.config.hidden_size))
            bh = tf.get_variable("bias_h", (1, self.config.hidden_size))

    def get_predictions(self, output):
        """Get answer predictions from output"""
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):
        """Calculate loss"""
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(att, labels))
        loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.answer_placeholder)) + gate_loss
        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(v)
        tf.scalar_summary('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op
  

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        questions = tf.split(1, self.max_q_len, questions)
        questions = [tf.squeeze(q, squeeze_dims=[1]) for q in questions]

        _, q_vec = tf.nn.rnn(self.gru_cell, questions, dtype=np.float32, sequence_length=self.question_len_placeholder)
        
        return q_vec

    def get_choice_representation(self, embeddings, choices, choice_length):
        #self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        #self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        #self.choice_placeholder = tf.placeholder(tf.int32, shape=(4,self.config.batch_size, self.max_z_len))
        #self.choice_len_placeholder = tf.placeholder(tf.int32, shape=(4,self.config.batch_size,))
        """Get choice vectors via embedding and GRU"""
        #z_vec = [[],[],[],[]]
        #for i in range(4):
        #    choices = tf.slice( self.choice_placeholder, [i,0,0], [1, self.config.batch_size, self.max_z_len] )
        #    choices = tf.reshape( choices, [self.config.batch_size, self.max_z_len] )
        #    choices = tf.nn.embedding_lookup(embeddings, choices )
        #    choices = tf.split(1, self.max_z_len, choices)
        #    choices = [tf.squeeze(z, squeeze_dims=[1]) for z in choices]
            #choice_length = tf.slice ( self.choice_len_placeholder, [i,0,0], [1, self.config.batch_size, :] )
        #    choice_length = tf.gather ( self.choice_len_placeholder, i )
        #    _, _vec = tf.nn.rnn( self.gru_cell, choices, dtype=np.float32, sequence_length=choice_length )
            #tf.get_variable_scope().reuse_variables()
        #    z_vec[i] = _vec

        choices = tf.nn.embedding_lookup(embeddings, choices)
        choices = tf.split(1, self.max_z_len, choices)
        choices = [tf.squeeze(z, squeeze_dims=[1]) for z in choices]
        _, z_vec = tf.nn.rnn( self.gru_cell, choices, dtype=np.float32, sequence_length=choice_length )
        #tf.get_variable_scope().reuse_variables()

        return z_vec


    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        inputs = tf.split(1, self.max_input_len, inputs)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

        outputs, _, _ = tf.nn.bidirectional_rnn(self.gru_cell, self.gru_cell, inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)

        # f<-> = f-> + f<-
        fact_vecs = [tf.reduce_sum(tf.pack(tf.split(1, 2, out)), 0) for out in outputs]

        # apply dropout
        fact_vecs = [tf.nn.dropout(fv, self.dropout_placeholder) for fv in fact_vecs]

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=True, initializer=_xavier_weight_init()):

            W_1 = tf.get_variable("W_1")
            b_1 = tf.get_variable("bias_1")

            W_2 = tf.get_variable("W_2")
            b_2 = tf.get_variable("bias_2")

            #print fact_vec.get_shape()
            #print q_vec.get_shape()
            features = [fact_vec*q_vec, fact_vec*prev_memory, tf.abs(fact_vec - q_vec), tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(1, features)
            #print feature_vec.get_shape()
            #print W_2.get_shape()
            #print b_2.get_shape()
            attention = tf.matmul(tf.tanh(tf.matmul(feature_vec, W_1) + b_1), W_2) + b_2
            
        return attention

    def _attention_GRU_step(self, rnn_input, h, g):
        """Implement attention GRU as described by https://arxiv.org/abs/1603.01417"""
        with tf.variable_scope("attention_gru", reuse=True, initializer=_xavier_weight_init()):

            Wr = tf.get_variable("Wr")
            Ur = tf.get_variable("Ur")
            br = tf.get_variable("bias_r")

            W = tf.get_variable("W")
            U = tf.get_variable("U")
            bh = tf.get_variable("bias_h")

            r = tf.sigmoid(tf.matmul(rnn_input, Wr) + tf.matmul(h, Ur) + br)
            h_hat = tf.tanh(tf.matmul(rnn_input, W) + r*tf.matmul(h, U) + bh)
            rnn_output = g*h_hat + (1-g)*h

            return rnn_output

    def generate_episode(self, memory, q_vec, fact_vecs):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(self.get_attention(q_vec, memory, fv), squeeze_dims=[1]) for fv in fact_vecs]

        attentions = tf.transpose(tf.pack(attentions))

        self.attentions.append(attentions)

        softs = tf.nn.softmax(attentions)
        softs = tf.split(1, self.max_input_len, softs)
        
        gru_outputs = []

        # set initial state to zero
        h = tf.zeros((self.config.batch_size, self.config.hidden_size))

        # use attention gru
        for i, fv in enumerate(fact_vecs):
            h = self._attention_GRU_step(fv, h, softs[i])
            gru_outputs.append(h)

        # extract gru outputs at proper index according to input_lens
        gru_outputs = tf.pack(gru_outputs)
        gru_outputs = tf.transpose(gru_outputs, perm=[1,0,2])
        episode = _last_relevant(gru_outputs, self.input_len_placeholder)

        return episode

    def add_answer_module(self, rnn_output, q_vec, z_vec):
        #z_vec = tf.stack( z_vec)
        z_vec = tf.pack( z_vec)
        z_vec = tf.transpose( z_vec, [1,0, 2])
	with tf.variable_scope("answer"):
            rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)
            #U = tf.get_variable("U", (2*self.config.embed_size, self.vocab_size))
            U = tf.get_variable("U", (2*self.config.embed_size, self.config.embed_size))
            #b_p = tf.get_variable("bias_p", (self.vocab_size,))
            b_p = tf.get_variable("bias_p", (self.config.embed_size,))

            final_story_encoding = tf.matmul(tf.concat(1, [rnn_output, q_vec]), U) + b_p # [batch_size, embed_size]

            final_story_encoding = tf.reshape(final_story_encoding, [ self.config.batch_size, 1 , self.config.embed_size ] )
            story_list = []
            for i in range(self.config.batch_size):
                single_story = tf.slice( final_story_encoding, [i, 0, 0] , [1, 1, self.config.embed_size] )
                single_story = tf.reshape(single_story, [1, self.config.embed_size])
                story_list.append(single_story)
            choice_list = []
	    for i in range(self.config.batch_size):
                choices_single_story = tf.slice( z_vec, [i, 0, 0] , [1, self.config.choice_num , self.config.embed_size] )
                choices_single_story = tf.reshape( choices_single_story, [self.config.choice_num, self.config.embed_size] )
                choice_list.append( choices_single_story )

            assert len(story_list) == len(choice_list), "stroy_size is not equal to choice size, exiting !!!"
            answer_list = []
            for i in range(len(story_list)):
                single_story = story_list[i]  #single story : [1, self.config.embed_size]
                choices_single_story = choice_list[i] # [self.config.choice_num, self.config.embed_size]
                four_answer_list = []
                for j in range(self.config.choice_num): # compute the cosine similarity between the context and the four choices
                    single_choice = tf.slice( choices_single_story, [j,0], [1,self.config.embed_size] )
                    choice_norm = tf.sqrt( tf.reduce_sum( tf.square(single_choice) ) )  # [1, self.config.embed_size]
                    story_norm = tf.sqrt( tf.reduce_sum( tf.square(single_story)) )
                    norm = tf.mul( story_norm , choice_norm ) # (story_norm float32, choice_norm int64)

                    #single_answer = tf.divide( tf.matmul( single_story , tf.transpose(single_choice,perm=[1,0]) ) , norm )
                    single_answer = tf.div( tf.matmul( single_story , tf.transpose(single_choice,perm=[1,0]) ) , norm )
                    four_answer_list.append(single_answer)

                #answer = tf.stack(four_answer_list) # [ 1, self.config.choice_num]
                answer = tf.pack(four_answer_list) # [ 1, self.config.choice_num]
                answer_list.append(answer)
            #answer = tf.stack(answer_list) # [ batch_size, 1, self.config.choice_num]
            answer = tf.pack(answer_list) # [ batch_size, 1, self.config.choice_num]
            answer = tf.reshape(answer, [self.config.batch_size, self.config.choice_num] )
        return answer

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
         
        # input fusion module
        with tf.variable_scope("question", initializer=_xavier_weight_init()):
            print '==> get question representation'
            q_vec = self.get_question_representation(embeddings)

        ##
        z_vec=[] 
        with tf.variable_scope("choice", initializer=_xavier_weight_init()):#, reuse=True):
            print '==> get choice representation'
            choices = tf.slice( self.choice_placeholder, [0,0,0], [1, self.config.batch_size, self.max_z_len] )
            choices = tf.reshape( choices, [self.config.batch_size, self.max_z_len] )
            choice_length = tf.gather ( self.choice_len_placeholder, 0 )
            z_vec.append(self.get_choice_representation(embeddings, choices, choice_length))

        with tf.variable_scope("choice", initializer=_xavier_weight_init(), reuse=True):
            print '==> get choice representation'
            choices = tf.slice( self.choice_placeholder, [1,0,0], [1, self.config.batch_size, self.max_z_len] )
            choices = tf.reshape( choices, [self.config.batch_size, self.max_z_len] )
            choice_length = tf.gather ( self.choice_len_placeholder, 1 )
            z_vec.append(self.get_choice_representation(embeddings, choices, choice_length))

        with tf.variable_scope("choice", initializer=_xavier_weight_init(), reuse=True):
            print '==> get choice representation'
            choices = tf.slice( self.choice_placeholder, [2,0,0], [1, self.config.batch_size, self.max_z_len] )
            choices = tf.reshape( choices, [self.config.batch_size, self.max_z_len] )
            choice_length = tf.gather ( self.choice_len_placeholder, 2 )
            z_vec.append(self.get_choice_representation(embeddings, choices, choice_length))

        with tf.variable_scope("choice", initializer=_xavier_weight_init(), reuse=True):
            print '==> get choice representation'
            choices = tf.slice( self.choice_placeholder, [3,0,0], [1, self.config.batch_size, self.max_z_len] )
            choices = tf.reshape( choices, [self.config.batch_size, self.max_z_len] )
            choice_length = tf.gather ( self.choice_len_placeholder, 3 )
            z_vec.append(self.get_choice_representation(embeddings, choices, choice_length))

        #sys.exit()

        with tf.variable_scope("input", initializer=_xavier_weight_init()):
            print '==> get input representation'
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=_xavier_weight_init()):
            print '==> build episodic memory'

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print '==> generating episode', i
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs)

                # untied weights for memory update
                Wt = tf.get_variable("W_t"+ str(i), (2*self.config.hidden_size+self.config.embed_size, self.config.hidden_size))
                bt = tf.get_variable("bias_t"+ str(i), (self.config.hidden_size,))

                # update memory with Relu
                prev_memory = tf.nn.relu(tf.matmul(tf.concat(1, [prev_memory, episode, q_vec]), Wt) + bt)

            output = prev_memory

        # pass memory module output through linear answer module
        output = self.add_answer_module(output, q_vec, z_vec)

        return output


    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) / config.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        p = np.random.permutation(len(data[0]))
        #qp, ip, ql, il, im, a, r = data
        #qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p]

        qp, ip, ql, il, im, a, r, zp, zl = data
        qp, ip, ql, il, im, a, r, zp, zl = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p], [zp[0][p], zp[1][p], zp[2][p], zp[3][p]], [zl[0][p], zl[1][p], zl[2][p], zl[3][p]]

        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.rel_label_placeholder: r[index],
                  self.dropout_placeholder: dp,
                  ##
                  self.choice_placeholder: [zp[0][index], zp[1][index], zp[2][index], zp[3][index]],
                  self.choice_len_placeholder: [zl[0][index], zl[1][index], zl[2][index], zl[3][index]]}
                  ##
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            answers = a[step*config.batch_size:(step+1)*config.batch_size]
            accuracy += np.sum(pred == answers)/float(len(answers))

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.mean(total_loss), accuracy/float(total_steps)

    def run_test_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) / config.test_batch_size
        #total_loss = []
        #accuracy = 0

 
        # shuffle data
        #p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a, r = data   # WE need a new data parser when testing , because there will be no a !
        #qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p] 

        pred_list = []
        for step in range(total_steps):
            index = range(step*config.test_batch_size,(step+1)*config.test_batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  #self.answer_placeholder: a[index],
                  self.rel_label_placeholder: r[index],
                  self.dropout_placeholder: dp}
            #loss, pred, summary, _ = session.run(
            #  [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)
            pred  ,_ = session.run(
              [self.pred , train_op], feed_dict=feed)

            pred_list.append(pred)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            #answers = a[step*config.batch_size:(step+1)*config.batch_size]
            #accuracy += np.sum(pred == answers)/float(len(answers))


            #total_loss.append(loss)
            #if verbose and step % verbose == 0:
            #    sys.stdout.write('\r{} / {} : loss = {}'.format(
            #      step, total_steps, np.mean(total_loss)))
             #   sys.stdout.flush()


        #if verbose:
        #    sys.stdout.write('\r')

        #print
        #print "********total_steps=",total_steps
        #return np.mean(total_loss), accuracy/float(total_steps)
        return pred_list

    def __init__(self, config):

        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=True)
        self.add_placeholders()
        self.add_reused_variables()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.merge_all_summaries()

