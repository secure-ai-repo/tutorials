"""
file: tfnet.py
includes: definition of class TFNet
this class initializes by building the forward pass
its methods include train, predict and savepb - saving
the current model to a protobuf file (no variable included)
"""

import sys
from yolo.drawer import *
from darknet import *
from tfop import *
from data import *

class TFNet(object):

	def __init__(self, darknet, FLAGS):
		self.meta = yolo_metaprocess(darknet.meta)
		self.darknet = darknet
		self.FLAGS = FLAGS
		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			self.forward()
			self.setup_meta_ops()

	def forward(self):
		self.ckpt = self.FLAGS.savepb is None
		verbalise = not self.ckpt

		# Placeholder
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')
		self.feed = dict() # for any other placeholders, e.g. dropout

		# Iterate through darknet layers
		now = self.inp
		num = len(self.darknet.layers)
		for i, layer in enumerate(self.darknet.layers):
			name = '{}-{}'.format(str(i),layer.type)
			if i+1 == num: name += '-tfnetoutput'
			args = [layer, now, name]
			if not self.ckpt: args += [self.feed]
			now = op_create(*args)(verbalise)

		# Attach the output to this tfnet
		self.out = now

	def setup_meta_ops(self):
		cfg = {
			'allow_soft_placement': False,
			'log_device_placement': False}
		if self.FLAGS.gpu > 0: 
			utility = min(FLAGS.gpu, 1.)
			print 'GPU mode with {} usage'.format(utility)
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True

		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		if self.FLAGS.train: yolo_loss(self)
		self.sess.run(tf.initialize_all_variables())
		
		if not self.ckpt:
			self.saver = tf.train.Saver(tf.all_variables(), 
				max_to_keep = self.FLAGS.keep)

			self.step = int()
			if self.FLAGS.load > 0:
				self.step = self.FLAGS.load
				load_point = 'backup/model-{}'.format(self.FLAGS.load)
				print 'Loading from {}'.format(load_point)
				self.saver.restore(self.sess, load_point)

	def savepb(self):
		"""
		Create a standalone const graph def
		So that C++ can load and run it.
		What's good abt it?
		1. Don't double the necessary size
		2. Convert on the fly - at any point you want
		"""
		darknet_ckpt = self.darknet
		flags_ckpt = self.FLAGS
		flags_ckpt.savepb = None # signal
		
		with self.graph.as_default() as g:
			for var in tf.trainable_variables():
				print var.name
				name = ':'.join(var.name.split(':')[:-1])
				var_name = name.split('-')
				val = var.eval(self.sess)
				l_idx = int(var_name[0])
				w_sig = var_name[-1]
				trained = var.eval(self.sess)
				darknet_ckpt.layers[l_idx].w[w_sig] = trained

		for layer in darknet_ckpt.layers:
			for ph in layer.h: # Set all placeholders to dfault val
				layer.h[ph] = self.feed[layer.h[ph]]['dfault']

		tfnet_ckpt = TFNet(darknet_ckpt, self.FLAGS)		
		tfnet_ckpt.sess = tf.Session(graph = tfnet_ckpt.graph)
		#tfnet_ckpt.predict() # uncomment for unit testing

		name = 'graph-{}.pb'.format(self.meta['model'])
		print('Saving const graph def to {}'.format(name))
		graph_def = tfnet_ckpt.sess.graph_def
		tf.train.write_graph(graph_def,'./',name,False)
	
	def train(self, train_set, parsed_annota, batch, epoch):
		batches = shuffle(train_set, parsed_annota, batch, epoch, self.meta)

		print('Training statistics:')
		print('\tLearning rate : {}'.format(self.FLAGS.lr))
		print('\tBatch size    : {}'.format(batch))
		print('\tEpoch number  : {}'.format(epoch))
		print('\tBackup every  : {}'.format(self.FLAGS.save))

		total = int() # total number of batches
		for i, packet in enumerate(batches):
			if i == 0: total = packet; continue
			x_batch, datum = packet
			feed_dict = yolo_feed_dict(self, x_batch, datum)
			feed_dict[self.inp] = x_batch
			for k in self.feed: feed_dict[k] = self.feed[k]['feed']

			_, loss = self.sess.run([self.train_op, self.loss], feed_dict)
			print('step {} - batch {} - loss {}'.format(i+self.step, i, loss))
			if i % (self.FLAGS.save/batch) == 0 or i == total:
				print 'save checkpoint and binaries at step {}'.format(self.step+i)
				self.saver.save(self.sess, 'backup/model-{}'.format(self.step+i))

	def predict(self):
		inp_path = self.FLAGS.testset
		all_inp_ = os.listdir(inp_path)
		all_inp_ = [i for i in all_inp_ if is_yolo_inp(i)]
		batch = min(self.FLAGS.batch, len(all_inp_))

		for j in range(len(all_inp_)/batch):
			inp_feed = list()
			all_inp = all_inp_[j*batch: (j*batch+batch)]
			new_all = list()
			for inp in all_inp:
				new_all += [inp]
				this_inp = '{}/{}'.format(inp_path, inp)
				this_inp = yolo_preprocess(this_inp)
				inp_feed.append(this_inp)
			all_inp = new_all

			feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
			for k in self.feed: feed_dict[k] = self.feed[k]['dfault']
		
			print ('Forwarding {} inputs ...'.format(len(inp_feed)))
			start = time.time()
			out = self.sess.run([self.out], feed_dict)
			stop = time.time()
			last = stop - start
			print ('Total time = {}s / {} inps = {} ips'.format(
				last, len(inp_feed), len(inp_feed) / last))

			for i, prediction in enumerate(out[0]):
				yolo_postprocess(prediction, '{}/{}'.format(inp_path, all_inp[i]), self.FLAGS, self.meta)