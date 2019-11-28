#! /usr/bin/env python

from darknet import *
from tfnet import *
from tensorflow import flags

flags.DEFINE_string("testset", "test", "path to testing directory")
flags.DEFINE_string("dataset", "../pascal/VOCdevkit/IMG/", "path to dataset directory")
flags.DEFINE_string("annotation", "../pascal/VOCdevkit/ANN/", "path to annotation directory")
flags.DEFINE_float("threshold", 0.1, "detection threshold")
flags.DEFINE_string("model", "", "configuration of choice")
flags.DEFINE_boolean("train", False, "training mode or not?")
flags.DEFINE_integer("load", 0, "load a saved backup/checkpoint, -1 for newest")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_float("gpu", 0.0, "How much gpu (from 0.0 to 1.0)")
flags.DEFINE_float("lr", 1e-5, "Learning rate")
flags.DEFINE_integer("keep",20,"Number of most recent training results to save")
flags.DEFINE_integer("batch", 12, "Batch size")
flags.DEFINE_integer("epoch", 1000, "Number of epoch")
flags.DEFINE_integer("save", 2000, "Save checkpoint every ? training examples")

FLAGS = flags.FLAGS
image = FLAGS.dataset
annot = FLAGS.annotation + 'parsed.bin'

def get_dir(dirs):
	for d in dirs:
		this = os.path.abspath(os.path.join(os.path.curdir, d))
		if not os.path.exists(this): os.makedirs(this)
get_dir([FLAGS.testset, 'results', 'backup'])

checkpoint = 'backup/checkpoint'
recent = os.path.isfile(checkpoint)
last = int()

if recent:
	with open(checkpoint,'r') as f:
		lines = f.readlines()
		name = lines[-1].split(' ')
		name = name[1].split('"')[1]
		last = int(name.split('-')[1])

if FLAGS.load < 0: FLAGS.load = last
darknet = Darknet(FLAGS.model)

print ('\nCompiling net & fill in parameters...')
start = time.time()
if FLAGS.gpu <= 0.: 
	with tf.device('cpu:0'): tfnet = TFNet(darknet, FLAGS)
else: tfnet = TFNet(darknet, FLAGS)
print ('Finished in {}s\n'.format(time.time() - start))

if FLAGS.train:
	print('Enter training ...')
	tfnet.train(image, annot, FLAGS.batch, FLAGS.epoch)
	if not FLAGS.savepb: exit('Training finished')

if FLAGS.savepb:
	print('Rebuild a constant version ...')
	tfnet.savepb(); exit('Done')

tfnet.predict()