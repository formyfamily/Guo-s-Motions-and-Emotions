import os
import sys
import pdb
import array
import json
import random
import zipfile
import time
import numpy as np
import tensorflow as tf
from models import LSTM
from argparse import ArgumentParser
from utils import ScreenPrinter, DataSet
from tensorflow.python.framework import constant_op

random.seed(1229)

dataArg = {	
	'movies' : None, # None means all movies
	'visual_features' : {
		"acc" : False,
		"cedd" : False,
		"cl" : False,
		"eh" : False,
		"fc6" : True,
		"fcth" : False,
		"gabor" : False,
		"jcd" : False,
		"lbp" : False,
		"sc" : False,
		"tamura" : False
	},
	'feature_dir' : "MEDIAEVAL17-DevSet-Visual_features/features",
	'valence_arousal_annotation_dir' : "MEDIAEVAL17-DevSet-Valence_Arousal-annotations/annotations",
	'fear_annotation_dir' : "MEDIAEVAL17-DevSet-Fear-annotations/annotations",
	'uploaded_dir' : "data/uploaded_data",
	'label_name' : 'f',
	'need_upload' : True 
}

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean("read_graph", False, "Set to False to build graph.")
tf.app.flags.DEFINE_integer("labels", 5, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 20, "Number of epoch.")
tf.app.flags.DEFINE_integer("units", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 24, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("keep_prob", 0.6, "Possibility of not dropout.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("train_dir", 'train', "Dir to save trained model")
tf.app.flags.DEFINE_string("label_name", "f", "Determine which label to solve")
tf.app.flags.DEFINE_string("learning_rate", 0.01, "Determine which label to solve")
FLAGS = tf.app.flags.FLAGS

def train(model, sess, dataset):
	st, ed, loss, accuracy = 0, 0, .0, .0
	gen_summary = True
	while ed < dataset.len:
		st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < dataset.len else dataset.len
		batch_data = dataset.gen_batch(st, ed, labelName=FLAGS.label_name)
		outputs = model.train_step(sess, batch_data, summary=gen_summary, keep_prob=FLAGS.keep_prob)
		if gen_summary: 
			summary = outputs[-1]
			gen_summary = False
		loss += outputs[0]
		accuracy += outputs[1]
	sess.run(model.epoch_add_op)
	return loss / dataset.len, accuracy / dataset.len, summary

def evaluate(model, sess, dataset):
	loss, accuracy, correlations = 0, 0, .0
	batch_data = dataset.gen_batch(0, dataset.len, labelName=FLAGS.label_name)
	if dataArg['label_name'] == 'va':
		outputs = sess.run([model.loss2, 'accuracy:0', model.correlation_coe], {model.keep_prob:1.0, 'features:0':batch_data['features'], 'input_len:0':batch_data['input_len'], 'labels:0':batch_data['labels']})
		loss = outputs[0]
		accuracy = outputs[1]
		correlations = outputs[2]
		print(outputs[0])
		print(correlations[1])
	else:	
		outputs = sess.run(['loss:0', 'accuracy:0', model.predict_labels], {model.keep_prob:1.0, 'features:0':batch_data['features'], 'input_len:0':batch_data['input_len'], 'labels:0':batch_data['labels']})
		TP, FP, TN, FN = 0, 0, 0, 0
		loss = outputs[0]
		accuracy = outputs[1]
		for i in range(0, batch_data['labels'].shape[0]):
			if batch_data['labels'][i] == 1 and outputs[2][i] == 1:
				TP += 1 
			if batch_data['labels'][i] == 1 and outputs[2][i] == 0:
				FN += 1
			if batch_data['labels'][i] == 0 and outputs[2][i] == 1:
				FP += 1
			if batch_data['labels'][i] == 0 and outputs[2][i] == 0:
				TN += 1
		precision = 0 if TP == 0 else TP/(TP+FP)
		recall = 0 if TP == 0 else TP/(TP+FN)
		F1 = 0 if TP == 0 else 2*TP/(2*TP+FP+FN)
		print("accuracy = %f"%(accuracy/dataset.len)) ;
		print("precision = %f"%precision) ;
		print("recall = %f"%recall) ;
		print("F1 = %f"%F1) ;
	#pdb.set_trace()		
	loss = outputs[0]
	accuracy = outputs[1]
	return loss / dataset.len, accuracy / dataset.len

def inference(model, sess, dataset):
	st, ed, loss, accuracy = 0, 0, .0, .0
	result = []
	while ed < dataset.len:
		st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < dataset.len else dataset.len
		batch_data = dataset.gen_batch(st, ed, labelName=FLAGS.label_name)
		outputs = sess.run(['predict_labels:0'], {model.keep_prob:1.0, 'features:0':batch_data['features'], 'input_len:0':batch_data['input_len'], 'labels:0':batch_data['labels']})
		result += outputs[0].tolist()

	with open('result.txt', 'w') as f:
		for label in result:
			f.write('%d\n' % label)

def get_args():
	parser = ArgumentParser(description="Guo's Emotions!")
	parser.add_argument('-m','--mode', default ="data",type=str, help='mode:data,train,test',required=False)
	args = parser.parse_args()
	return args

def main():
	args = get_args()
	if(args.mode == "data"):
		dataset = DataSet.DataSet(dataArg)
		pdb.set_trace()
	elif(args.mode == "train"):
		dataArg['uploaded_dir'] = "data/dev_uploaded_data"
		dataArg['need_upload'] = False 
		devDataset = DataSet.DataSet(dataArg) 
		print(devDataset.featureData.shape)
		dataArg['uploaded_dir'] = "data/test_uploaded_data"
		dataArg['need_upload'] = False 
		testDataset = DataSet.DataSet(dataArg) 
		print(testDataset.featureData.shape)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			if FLAGS.is_train:
				print(FLAGS.__flags)
				model = LSTM.LSTM(
						FLAGS.units, 
						FLAGS.layers,
						devDataset.feature_size,
						dataArg['label_name'],
						FLAGS.learning_rate)
				if FLAGS.log_parameters:
					model.print_parameters()
				
				if tf.train.get_checkpoint_state(FLAGS.train_dir):
					print("Reading model parameters from %s" % FLAGS.train_dir)
					model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
				else:
					print("Created model with fresh parameters.")
					tf.global_variables_initializer().run()
					tf.local_variables_initializer().run()

				summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
				while model.epoch.eval() < FLAGS.epoch:
					epoch = model.epoch.eval()
					devDataset.random_shuffle() 
					start_time = time.time()
					loss, accuracy, summary = train(model, sess, devDataset)
					#while True:
					#	pass
					sess.run(model.learning_rate_decay_op)
					summary_writer.add_summary(summary, epoch)
					summary = tf.Summary()
					summary.value.add(tag='loss/train', simple_value=loss)
					summary.value.add(tag='accuracy/train', simple_value=accuracy)
					#summary.value.add(tag='logits/train', simple_value=logits)
					summary_writer.add_summary(summary, epoch)
					#model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
					print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (epoch, model.learning_rate.eval(), time.time()-start_time, loss, accuracy))
					loss, accuracy = evaluate(model, sess, devDataset)
					summary = tf.Summary()
					summary.value.add(tag='loss/dev', simple_value=loss)
					summary.value.add(tag='accuracy/dev', simple_value=accuracy)
					summary_writer.add_summary(summary, epoch)
					print("		dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
					loss, accuracy = evaluate(model, sess, testDataset)
					summary = tf.Summary()
					summary.value.add(tag='loss/test', simple_value=loss)
					summary.value.add(tag='accuracy/test', simple_value=accuracy)
					summary_writer.add_summary(summary, epoch)
					#print("		test_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))

if __name__ == "__main__":
	main()