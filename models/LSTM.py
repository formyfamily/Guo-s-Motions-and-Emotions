import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
import pdb

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']

FLAGS = tf.app.flags.FLAGS

class LSTM(object):
    def __init__(self,
            num_units,
            num_layers,
            feature_size,
            label_type,
            learning_rate=0.5,
            max_gradient_norm=5.0,
            learning_rate_decay_factor=0.98):

        self.keep_prob = tf.placeholder(tf.float32)
        self.input = tf.placeholder(shape=(None,10,feature_size), dtype=tf.float32, name='features')  # shape: batch*len(10)*feature_size
        self.input =  tf.nn.dropout(self.input, keep_prob=self.keep_prob) #batch*len(10)*feature_size
        self.input_len = tf.placeholder(shape=(None), dtype=tf.float32, name='input_len') #batch
        self.labels = tf.placeholder(shape=(None,None) , dtype=np.float32, name='labels')  # shape: batch

        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        
        cell = tf.contrib.rnn.BasicLSTMCell(num_units) 
        
        outputs, states = dynamic_rnn(cell, self.input, self.input_len, dtype=tf.float32, scope="rnn")
        #pdb.set_trace() 
        # note that no matter the label type there're always two outputs.
        self.logits = tf.nn.dropout(tf.layers.dense(inputs=states[1], units=2, activation=None), keep_prob=self.keep_prob, name='answer')

        if(label_type == 'f'):
            self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits), name='loss')
            predict_labels = tf.argmax(self.logits, 1, 'predict_labels')
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.labels, predict_labels), tf.int32), name='accuracy')
        else:
            self.loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.labels, predictions=self.logits), name='loss')
            self.accuracy = tf.Variable(0, trainable=False, name='accuracy')
        mean_loss = self.loss / tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
            
        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        tf.summary.scalar('loss/step', self.loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=5, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def train_step(self, session, data, summary=False, keep_prob=0.5):
        input_feed = {self.keep_prob: keep_prob, self.input: data['features'], self.input_len: data['input_len'], self.labels: data['labels']}
        output_feed = [self.loss, self.accuracy, self.logits, self.gradient_norm, self.update]
        if summary:
            output_feed.append(self.merged_summary_op)
        return session.run(output_feed, input_feed)
    
