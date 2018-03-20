# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score, normalize_answer
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, CNNEmbedding
from modules import HighWayNetwork, BiDaff, BiLSTM

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    #TODO generalize to hande preconfigured char_emb_matrix
    def __init__(self, FLAGS, id2word, word2id, emb_matrix, id2char = None, char2id = None, char_emb_matrix=None):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.id2char = id2char
        self.char2id = char2id

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #opt = tf.train.AdadeltaOptimizer(learning_rate = FLAGS.learning_rate, rho = FLAGS.lr_decay)
        #Baseline optimizer
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.char_context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len+2])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.char_qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_len+2])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Create character level embeddings
        #Context
        # raise vocab size, 64 to accessible varibale
        if self.FLAGS.char_embedding:
            Q = tf.get_variable('Q', shape=[87+1+1+1+1, self.FLAGS.char_embedding_size],
                                initializer = tf.contrib.layers.xavier_initializer())
            char_embedder = CNNEmbedding(self.FLAGS.char_window_size,self.FLAGS.char_filter_num,
                                         self.FLAGS.char_embedding_size, self.FLAGS.word_len, self.keep_prob,1)
            char_context_emb = char_embedder.build_graph(self.char_context_ids,self.context_mask,self.FLAGS.context_len, Q,
                                                         scope = "Context_Emb")
            self.full_context_embs = tf.concat([self.context_embs,char_context_emb],axis=2)
            # Highway Network
            highway_network = HighWayNetwork()
            self.full_context_embs = highway_network.build_graph(self.full_context_embs,2*self.FLAGS.embedding_size)
            #question
            char_qn_emb = char_embedder.build_graph(self.char_qn_ids, self.qn_mask, self.FLAGS.question_len, Q,
                                                    scope="Question_Emb")
            self.full_qn_embs = tf.concat([self.qn_embs, char_qn_emb], axis=2)
            # Highway network
            self.full_qn_embs = highway_network.build_graph(self.full_qn_embs, 2*self.FLAGS.embedding_size)
        else:
            self.full_context_embs =self.context_embs
            self.full_qn_embs = self.qn_embs

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.full_context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.full_qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        context_hiddens_exp = tf.expand_dims(context_hiddens, 2) # (batch_size, context_len, 1, hidden_size*2)
        #context_hiddens_exp = tf.tile(context_hiddens_exp, [1, 1, self.FLAGS.question_len, 1]) #(batch_size, context_len
                                                                                        # ,question_len , hidden_size*2)
        question_hiddens_exp = tf.expand_dims(question_hiddens, 1) #(batch_size, 1, question_len, hidden_size*2)
        #question_hiddens_exp = tf.tile(question_hiddens_exp, [1, self.FLAGS.context_len, 1, 1]) #(batch_size, context_len,
                                                                                                # question_len, hidden_size*2)
        #print question_hiddens_exp.get_shape(), context_hiddens_exp.get_shape()
        e_multiplied = tf.multiply(context_hiddens_exp, question_hiddens_exp)
       # S_intermediate = tf.concat([context_hiddens_exp, question_hiddens_exp, e_multiplied], 3)
        #print e_multiplied.get_shape(), S_intermediate.get_shape()
        W_A = tf.get_variable("W_BiDaff", shape=[1,1,1,self.FLAGS.hidden_size * 6],
                              initializer=tf.contrib.layers.xavier_initializer())
        #S = tf.multiply(S_intermediate, W_A)
        q1 = tf.reduce_sum(tf.multiply(context_hiddens_exp,W_A[0,0,0,0:self.FLAGS.hidden_size * 2]),axis=3)
        c1 = tf.reduce_sum(tf.multiply(question_hiddens_exp, W_A[0, 0, 0, self.FLAGS.hidden_size * 2:
                                                                          self.FLAGS.hidden_size * 4]), axis=3)
        q1_c1 = tf.reduce_sum(tf.multiply(e_multiplied, W_A[0, 0, 0, self.FLAGS.hidden_size * 4:
                                                                          self.FLAGS.hidden_size * 6]), axis=3)
        #S = tf.reduce_sum(S,axis = 3)
        #print "q shape",q1.get_shape(), "c shape", c1.get_shape(), "e", q1_c1.get_shape()
        S = q1+c1+q1_c1
        #print "S", S.get_shape()
        biAttn_layer = BiDaff(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        c2q,q2c = biAttn_layer.build_graph(question_hiddens,self.qn_mask,context_hiddens,self.context_mask,S)
        #print c2q.get_shape(), q2c.get_shape()
        #q2c = tf.expand_dims(q2c, 1) # shape(batch_size,1,context_vec_size)
        q2c = tf.tile(q2c, [1,self.FLAGS.context_len,1]) # shape(batch_size,context_vec_size,context_vec_size)
        #print q2c.get_shape()
        c_c2q = tf.multiply(context_hiddens,c2q) #shape(batch_size,num_question,hidden_size*2)
        c_q2c = tf.multiply(context_hiddens, q2c) #shape(batch_size,num_question,hidden_size*2)
        #print c_c2q.get_shape(), c_q2c.get_shape()
        blended_reps_bi = tf.concat([context_hiddens, c2q, c_c2q,c_q2c],axis = 2)  #shape(batch_size,num_context,hidden_size*8)


        # ########################
        # attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        # _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)
        #
        # # Concat attn_output to context_hiddens to get blended_reps
        # blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)
        # ########################
        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default

        modelling = BiLSTM(self.FLAGS.hidden_size,self.keep_prob)
        modelling2 = BiLSTM(self.FLAGS.hidden_size,self.keep_prob)
        modelling3 = BiLSTM(self.FLAGS.hidden_size,self.keep_prob)
        blended_reps_int = modelling.build_graph(blended_reps_bi,self.context_mask,"LSTM1") #shape(batch_size,context_len,2*hidden_size)
       #print "blended vector after LSTM", blended_reps_int.get_shape(), "Blended vector after BiDaff", blended_reps_bi.get_shape()
        blended_reps_final = modelling2.build_graph(blended_reps_int,self.context_mask,"LSTM2") #shape(batch_size,context_len,2*hidden_size)
        blended_reps_start = tf.concat([blended_reps_bi,blended_reps_final],axis = 2)
        blended_reps_final2 = modelling3.build_graph(blended_reps_final,self.context_mask,"LSTM3")
        # shape(batch_size,context_len,2*hidden_size)

        # for input to softmax of end distribution. This is how it is defined in the paper.
        blended_reps_end = tf.concat([blended_reps_bi, blended_reps_final2], axis=2)
        #blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_
        # reps_final is shape (batch_size, context_len, hidden_size)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            #self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_start,
                                                                                     self.context_mask)
        pos_start = tf.argmax(self.probdist_start,axis=1,output_type=tf.int32)-1 
	# Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
	    st = tf.sequence_mask(pos_start,self.FLAGS.context_len)
	    st = tf.cast(st,dtype=tf.int32)
	    st = tf.negative(st)
	    st = st +1 
            st = tf.multiply(st,self.context_mask)
            softmax_layer_end = SimpleSoftmaxLayer()
            #self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_end, st)


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            a = tf.greater(loss_end,20)
	    a = tf.cast(a,dtype=tf.float32)
	    loss_end = tf.multiply(loss_start,a)+tf.multiply((tf.negative(a)+1.0),loss_end) 
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        if self.FLAGS.char_embedding:
            input_feed[self.char_context_ids] = batch.char_context_ids
            input_feed[self.char_qn_ids] = batch.char_qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        if self.FLAGS.char_embedding:
            input_feed[self.char_context_ids] = batch.char_context_ids
            input_feed[self.char_qn_ids] = batch.char_qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        if self.FLAGS.char_embedding:
            input_feed[self.char_context_ids] = batch.char_context_ids
            input_feed[self.char_qn_ids] = batch.char_qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.

        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path,
                                         self.FLAGS.batch_size, context_len=self.FLAGS.context_len,
                                         question_len=self.FLAGS.question_len, discard_long=True,
                                         word_length=self.FLAGS.word_len, char2id=self.char2id):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path,
                                         self.FLAGS.batch_size, context_len=self.FLAGS.context_len,
                                         question_len=self.FLAGS.question_len, discard_long=False,
                                         word_length=self.FLAGS.word_len, char2id=self.char2id):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total

    def gen_stats(self, session, context_path, qn_path, ans_path):
        """
        Generate a variety of statistics to get a more holistic view of how the model is performing
        :param session:
        :param context_path:
        :param qn_path:
        :param ans_path:
        :param dataset:
        :param num_samples:
        :param print_to_screen:
        :return:
        """
        list_of_data_tuples=[]
        count =0
        totally_wrong = 0
        # For every batch
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path,
                                         self.FLAGS.batch_size, context_len=self.FLAGS.context_len,
                                         question_len=self.FLAGS.question_len, discard_long=True,
                                         word_length=self.FLAGS.word_len, char2id=self.char2id):
            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist()  # list length batch_size
            pred_end_pos = pred_end_pos.tolist()  # list length batch_size

            # For every question in a batch
            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(
                zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start: pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)
                if pred_ans_start > batch.ans_span[1] or pred_ans_end < batch.ans_span[0]:
                    totally_wrong +=1
                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)

                # Calc interesting things
                true_answer_length = len(true_answer)
                pred_answer_length = len(normalize_answer(pred_answer))
                if "where" in batch.qn_tokens[ex_idx] or "Where" in batch.qn_tokens[ex_idx]:
                    question_type = "where"
                elif "when" in batch.qn_tokens[ex_idx] or "When" in batch.qn_tokens[ex_idx]:
                    question_type = "when"
                elif "how" in batch.qn_tokens[ex_idx] or "how" in batch.qn_tokens[ex_idx]:
                    question_type = "how"
                elif "why" in batch.qn_tokens[ex_idx] or "Why" in batch.qn_tokens[ex_idx]:
                    question_type = "why"
                elif "what" in batch.qn_tokens[ex_idx] or "What" in batch.qn_tokens[ex_idx]:
                    question_type = "what"
                elif "who" in batch.qn_tokens[ex_idx] or "Who" in batch.qn_tokens[ex_idx]:
                    question_type = "who"
                elif "Do" in batch.qn_tokens[ex_idx] or "do" in batch.qn_tokens[ex_idx]:
                    question_type = "do"
                elif "which" in batch.qn_tokens[ex_idx] or "Which" in batch.qn_tokens[ex_idx]:
                    question_type = "which"
                else:
                    question_type = "other"
                list_of_data_tuples.append((f1,em,question_type,true_answer_length,pred_answer_length,totally_wrong))
            count += 1
            if count%20==0:
                print("Batch #: "+str(count))
        return list_of_data_tuples

    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path,
                                             self.FLAGS.batch_size, context_len=self.FLAGS.context_len,
                                             question_len=self.FLAGS.question_len, discard_long=True,
                                             word_length = self.FLAGS.word_len, char2id = self.char2id):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
