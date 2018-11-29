
# coding: utf-8

## Notebook for network training

# In[ ]:

import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import logging
from sklearn.cross_validation import StratifiedKFold
import lasagne
import theano
from lasagne.layers import InputLayer
#from lasagne.layers.dnn import Conv3DDNNLayer
#from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.layers.conv import Conv3DLayer as Conv3DDNNLayer
from lasagne.layers.pool import Pool3DLayer as Pool3DDNNLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import identity, softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.layers import InverseLayer
import theano.tensor as T
import time
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import sys
import os
from dotlayer import DotLayer
from skimage import data as datta
from skimage import color, io, img_as_float


from mayavi import mlab

import matplotlib.pyplot as plt

import gzip
import shutil


import ConfigParser
from cnn_cort.load_options import *

def make_image(data, outputname, cmap, size=(1, 1), dpi=110, vmin=0, vmax=0):
	fig = plt.figure()
	fig.set_size_inches(size)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.set_cmap(cmap)
	if vmax - vmin > 0:
		ax.imshow(data, aspect='equal', vmin=vmin, vmax=vmax)
	else:
		ax.imshow(data, aspect='equal')
	plt.savefig(outputname, dpi=dpi)


def make_masked_image(data, data2, outputname, cmap, size=(1, 1), dpi=110, vmin=0, vmax=0):
	fig = plt.figure()
	fig.set_size_inches(size)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.set_cmap('bone')
	ax.imshow(data2)
	plt.set_cmap(cmap)

	if vmax - vmin > 0:
		ax.imshow(data, aspect='equal', vmin=vmin, vmax=vmax)
	else:
		ax.imshow(data, aspect='equal')
	plt.savefig(outputname, dpi=dpi)
	

name = sys.argv[1]

print(name)

shutil.copyfile(name, 'data/test1/T1.nii')


with open('data/test1/T1.nii', 'rb') as f_in:
    with gzip.open('data/test1/T1.nii.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


CURRENT_PATH = os.getcwd()
# --------------------------------------------------
# 1. load options from config file. Options are set
#    the configuration.cfg file 
# --------------------------------------------------


user_config = ConfigParser.RawConfigParser()

user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)

# --------------------------------------------------
# load data 
# --------------------------------------------------

from cnn_cort.base import load_data, generate_training_set, load_test_names, test_scan
from cnn_cort.nets import build_model

'''
# get data patches from all orthogonal views 
x_axial, x_cor, x_sag, y, x_atlas, names = load_data(options)

# build the training dataset
x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train = generate_training_set(x_axial,
                                                                                        x_cor,
                                                                                        x_sag,
                                                                                        x_atlas,
                                                                                        y,
                                                                                        options)

# --------------------------------------------------
# build the net model
# --------------------------------------------------
weights_path = os.path.join(CURRENT_PATH, 'nets')
net = build_model(weights_path, options)


# --------------------------------------------------
# train the net
# --------------------------------------------------
net.fit({'in1': x_train_axial,
         'in2': x_train_cor,
         'in3': x_train_sag,
         'in4': x_train_atlas}, y_train)

'''
# --------------------------------------------------
# test the model (for each scan)
# --------------------------------------------------

# get the testing image paths
t1_test_paths, folder_names  = load_test_names(options)

# reload the network weights and build it 
weights_path = os.path.join(CURRENT_PATH, 'nets')
options['net_verbose'] = 0
net = build_model(weights_path, options)

# iterate through all test scans
for t1, current_scan in zip(t1_test_paths, folder_names):
    t = test_scan(net, t1, options)
    print "    -->  tested subject :", current_scan, "(elapsed time:", t, "min.)"

#### Batch iteration functions

# In[ ]:

from utils import iterate_minibatches, iterate_minibatches_train







# In[ ]:

input_var = T.tensor5(name='input', dtype='float32')
target_var = T.ivector()
input_var2 = T.matrix(name='input2', dtype='float32')
#input_var2 = T.vector2(name='input2', dtype='float32')


#### Network definition

# In[ ]:

def build_net():
	"""Method for VoxResNet Building.

	Returns
	-------
	dictionary
		Network dictionary.
	"""
	net = {}
	net['input'] = InputLayer((None, 1, 110, 110, 110), input_var=input_var)
	net['conv1a'] = Conv3DDNNLayer(net['input'], 32, 3, pad='same',
								   nonlinearity=identity)
	net['bn1a'] = BatchNormLayer(net['conv1a'])
	net['relu1a'] = NonlinearityLayer(net['bn1a'])
	net['conv1b'] = Conv3DDNNLayer(net['relu1a'], 32, 3, pad='same',
								   nonlinearity=identity)
	net['bn1b'] = BatchNormLayer(net['conv1b'])
	net['relu1b'] = NonlinearityLayer(net['bn1b'])
	net['conv1c'] = Conv3DDNNLayer(net['relu1b'], 64, 3, stride=(2, 2, 2),
								   pad='same', nonlinearity=identity)
	# VoxRes block 2
	net['voxres2_bn1'] = BatchNormLayer(net['conv1c'])
	net['voxres2_relu1'] = NonlinearityLayer(net['voxres2_bn1'])
	net['voxres2_conv1'] = Conv3DDNNLayer(net['voxres2_relu1'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres2_bn2'] = BatchNormLayer(net['voxres2_conv1'])
	net['voxres2_relu2'] = NonlinearityLayer(net['voxres2_bn2'])
	net['voxres2_conv2'] = Conv3DDNNLayer(net['voxres2_relu2'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres2_out'] = ElemwiseSumLayer([net['conv1c'],
										   net['voxres2_conv2']])
	# VoxRes block 3
	net['voxres3_bn1'] = BatchNormLayer(net['voxres2_out'])
	net['voxres3_relu1'] = NonlinearityLayer(net['voxres3_bn1'])
	net['voxres3_conv1'] = Conv3DDNNLayer(net['voxres3_relu1'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres3_bn2'] = BatchNormLayer(net['voxres3_conv1'])
	net['voxres3_relu2'] = NonlinearityLayer(net['voxres3_bn2'])
	net['voxres3_conv2'] = Conv3DDNNLayer(net['voxres3_relu2'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres3_out'] = ElemwiseSumLayer([net['voxres2_out'],
										   net['voxres3_conv2']])

	net['bn4'] = BatchNormLayer(net['voxres3_out'])
	net['relu4'] = NonlinearityLayer(net['bn4'])
	net['conv4'] = Conv3DDNNLayer(net['relu4'], 64, 3, stride=(2, 2, 2),
								  pad='same', nonlinearity=identity)
	# VoxRes block 5
	net['voxres5_bn1'] = BatchNormLayer(net['conv4'])
	net['voxres5_relu1'] = NonlinearityLayer(net['voxres5_bn1'])
	net['voxres5_conv1'] = Conv3DDNNLayer(net['voxres5_relu1'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres5_bn2'] = BatchNormLayer(net['voxres5_conv1'])
	net['voxres5_relu2'] = NonlinearityLayer(net['voxres5_bn2'])
	net['voxres5_conv2'] = Conv3DDNNLayer(net['voxres5_relu2'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres5_out'] = ElemwiseSumLayer([net['conv4'], net['voxres5_conv2']])
	# VoxRes block 6
	net['voxres6_bn1'] = BatchNormLayer(net['voxres5_out'])
	net['voxres6_relu1'] = NonlinearityLayer(net['voxres6_bn1'])
	net['voxres6_conv1'] = Conv3DDNNLayer(net['voxres6_relu1'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres6_bn2'] = BatchNormLayer(net['voxres6_conv1'])
	net['voxres6_relu2'] = NonlinearityLayer(net['voxres6_bn2'])
	net['voxres6_conv2'] = Conv3DDNNLayer(net['voxres6_relu2'], 64, 3,
										  pad='same', nonlinearity=identity)
	net['voxres6_out'] = ElemwiseSumLayer([net['voxres5_out'],
										   net['voxres6_conv2']])

	net['bn7'] = BatchNormLayer(net['voxres6_out'])
	net['relu7'] = NonlinearityLayer(net['bn7'])
	net['conv7'] = Conv3DDNNLayer(net['relu7'], 128, 3, stride=(2, 2, 2),
								  pad='same', nonlinearity=identity)

	# VoxRes block 8
	net['voxres8_bn1'] = BatchNormLayer(net['conv7'])
	net['voxres8_relu1'] = NonlinearityLayer(net['voxres8_bn1'])
	net['voxres8_conv1'] = Conv3DDNNLayer(net['voxres8_relu1'], 128, 3,
										  pad='same', nonlinearity=identity)
	net['voxres8_bn2'] = BatchNormLayer(net['voxres8_conv1'])
	net['voxres8_relu2'] = NonlinearityLayer(net['voxres8_bn2'])
	net['voxres8_conv2'] = Conv3DDNNLayer(net['voxres8_relu2'], 128, 3,
										  pad='same', nonlinearity=identity)
	net['voxres8_out'] = ElemwiseSumLayer([net['conv7'], net['voxres8_conv2']])
	# VoxRes block 9
	net['voxres9_bn1'] = BatchNormLayer(net['voxres8_out'])
	net['voxres9_relu1'] = NonlinearityLayer(net['voxres9_bn1'])
	net['voxres9_conv1'] = Conv3DDNNLayer(net['voxres9_relu1'], 128, 3,
										  pad='same', nonlinearity=identity)
	net['voxres9_bn2'] = BatchNormLayer(net['voxres9_conv1'])
	net['voxres9_relu2'] = NonlinearityLayer(net['voxres9_bn2'])
	net['voxres9_conv2'] = Conv3DDNNLayer(net['voxres9_relu2'], 128, 3,
										  pad='same', nonlinearity=identity)
	net['voxres9_out'] = ElemwiseSumLayer([net['voxres8_out'],
										   net['voxres9_conv2']])

	net['pool10'] = Pool3DDNNLayer(net['voxres9_out'], 7)
	net['fc11'] = DenseLayer(net['pool10'], 128)
	net['prob'] = DenseLayer(net['fc11'], 2, nonlinearity=softmax)

	# not in original code
	net['rsfc11'] = lasagne.layers.ReshapeLayer(net['fc11'], [1, 1, 128])
	net['maxfc'] = lasagne.layers.MaxPool1DLayer(net['rsfc11'], 128)


	return net

def build_invnet(net):


	invnet = {}
	invnet['input'] = InputLayer((None, 128), input_var=input_var2)
	#invnet['prob'] = InverseLayer(net['prob'], net['prob'])
	#invnet['maxfc'] = InverseLayer(net['maxfc'], net['maxfc'])
	#invnet['rsfc11'] = InverseLayer(invnet['maxfc'] net['rsfc11'])

	invnet['prob'] = InverseLayer(net['prob'], net['prob'])
	#print(lasagne.layers.get_output_shape(invnet['prob']))
	#invnet['mult'] = DotLayer(invnet['input'], invnet['prob'])
	invnet['mult'] = lasagne.layers.ElemwiseMergeLayer([invnet['input'], invnet['prob']], merge_function=theano.tensor.mul)
	invnet['fc11'] = InverseLayer(invnet['mult'], net['fc11'])
	invnet['pool10'] = InverseLayer(invnet['fc11'], net['pool10'])

	invnet['voxres9_conv2'] = InverseLayer(invnet['pool10'], net['voxres9_conv2'])
	invnet['voxres9_relu2'] = InverseLayer(invnet['voxres9_conv2'], net['voxres9_relu2'])
	invnet['voxres9_bn1'] = InverseLayer(invnet['voxres9_relu2'], net['voxres9_bn1'])

	invnet['voxres8_conv2'] = InverseLayer(invnet['voxres9_bn1'], net['voxres8_conv2'])
	invnet['voxres8_relu2'] = InverseLayer(invnet['voxres8_conv2'], net['voxres8_relu2'])
	invnet['voxres8_bn2'] = InverseLayer(invnet['voxres8_relu2'], net['voxres8_bn2'])

	invnet['voxres8_conv1'] = InverseLayer(invnet['voxres8_bn2'], net['voxres8_conv1'])
	invnet['voxres8_relu1'] = InverseLayer(invnet['voxres8_conv1'], net['voxres8_relu1'])
	invnet['voxres8_bn1'] = InverseLayer(invnet['voxres8_relu1'], net['voxres8_bn1'])

	invnet['conv7'] = InverseLayer(invnet['voxres8_bn1'], net['conv7'])
	invnet['relu7'] = InverseLayer(invnet['conv7'], net['relu7'])
	invnet['bn7'] = InverseLayer(invnet['relu7'], net['bn7'])

	invnet['voxres6_conv2'] = InverseLayer(invnet['bn7'], net['voxres6_conv2'])
	invnet['voxres6_relu2'] = InverseLayer(invnet['voxres6_conv2'], net['voxres6_relu2'])
	invnet['voxres6_bn2'] = InverseLayer(invnet['voxres6_relu2'], net['voxres6_bn2'])
	invnet['voxres6_conv1'] = InverseLayer(invnet['voxres6_bn2'], net['voxres6_conv1'])
	invnet['voxres6_relu1'] = InverseLayer(invnet['voxres6_conv1'], net['voxres6_relu1'])
	invnet['voxres6_bn1'] = InverseLayer(invnet['voxres6_relu1'], net['voxres6_bn1'])

	invnet['voxres5_conv2'] = InverseLayer(invnet['voxres6_bn1'], net['voxres5_conv2'])
	invnet['voxres5_relu2'] = InverseLayer(invnet['voxres5_conv2'], net['voxres5_relu2'])
	invnet['voxres5_bn2'] = InverseLayer(invnet['voxres5_relu2'], net['voxres5_bn2'])
	invnet['voxres5_conv1'] = InverseLayer(invnet['voxres5_bn2'], net['voxres5_conv1'])
	invnet['voxres5_relu1'] = InverseLayer(invnet['voxres5_conv1'], net['voxres5_relu1'])
	invnet['voxres5_bn1'] = InverseLayer(invnet['voxres5_relu1'], net['voxres5_bn1'])

	invnet['conv4'] = InverseLayer(invnet['voxres5_bn1'], net['conv4'])
	invnet['relu4'] = InverseLayer(invnet['conv4'], net['relu4'])
	invnet['bn4'] = InverseLayer(invnet['relu4'], net['bn4'])

	invnet['voxres3_conv2'] = InverseLayer(invnet['bn4'], net['voxres3_conv2'])
	invnet['voxres3_relu2'] = InverseLayer(invnet['voxres3_conv2'], net['voxres3_relu2'])
	invnet['voxres3_bn2'] = InverseLayer(invnet['voxres3_relu2'], net['voxres3_bn2'])
	invnet['voxres3_conv1'] = InverseLayer(invnet['voxres3_bn2'], net['voxres3_conv1'])
	invnet['voxres3_relu1'] = InverseLayer(invnet['voxres3_conv1'], net['voxres3_relu1'])
	invnet['voxres3_bn1'] = InverseLayer(invnet['voxres3_relu1'], net['voxres3_bn1'])

	invnet['voxres2_conv2'] = InverseLayer(invnet['voxres3_bn1'], net['voxres2_conv2'])
	invnet['voxres2_relu2'] = InverseLayer(invnet['voxres2_conv2'], net['voxres2_relu2'])
	invnet['voxres2_bn2'] = InverseLayer(invnet['voxres2_relu2'], net['voxres2_bn2'])
	invnet['voxres2_conv1'] = InverseLayer(invnet['voxres2_bn2'], net['voxres2_conv1'])
	invnet['voxres2_relu1'] = InverseLayer(invnet['voxres2_conv1'], net['voxres2_relu1'])
	invnet['voxres2_bn1'] = InverseLayer(invnet['voxres2_relu1'], net['voxres2_bn1'])

	invnet['conv1c'] = InverseLayer(invnet['voxres2_bn1'], net['conv1c'])
	invnet['relu1b'] = InverseLayer(invnet['conv1c'], net['relu1b'])
	invnet['bn1b'] = InverseLayer(invnet['relu1b'], net['bn1b'])

	invnet['conv1b'] = InverseLayer(invnet['bn1b'], net['conv1b'])
	invnet['relu1a'] = InverseLayer(invnet['conv1b'], net['relu1a'])
	invnet['bn1a'] = InverseLayer(invnet['relu1a'], net['bn1a'])
	invnet['conv1a'] = InverseLayer(invnet['bn1a'], net['conv1a'])


	return invnet

# In[ ]:

# Logging setup
logging.basicConfig(format='[%(asctime)s]  %(message)s',
					datefmt='%d.%m %H:%M:%S',
					level=logging.DEBUG)


#### Training function definition

# In[ ]:

def run_training(first_class, second_class, results_folder,
				 num_epochs=1, batchsize=1):
	"""Iterate minibatches on train subset.

	Parameters
	----------
	first_class : {'AD', 'LMCI', 'EMCI', 'Normal'}
		String label for target == 0.
	second_class : {'AD', 'LMCI', 'EMCI', 'Normal'}
		String label for target == 1.
	results_folder : string
		Folder to store results.
	num_epochs : integer
		Number of epochs for all of the experiments. Default is 70.
	batchsize : integer
		Batchsize for network training. Default is 3.
	"""
	
	if first_class not in {'AD', 'LMCI', 'EMCI', 'Normal'}:
		msg = "First class must be 'AD', 'LMCI', 'EMCI' or 'Normal', not {0}"
		raise ValueError(msg.format(first_class))
	
	if second_class not in {'AD', 'LMCI', 'EMCI', 'Normal'}:
		msg = "Second class must be 'AD', 'LMCI', 'EMCI' or 'Normal', not {0}"
		raise ValueError(msg.format(second_class))
		
	if first_class == second_class:
		raise ValueError("Class labels should be different")
		
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
	
	metadata = pd.read_csv('data/test1/metadata.csv')
	smc_mask = ((metadata.Label == first_class) |
				(metadata.Label == second_class)).values.astype('bool')
	data = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')
	dic = {}
	for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
					   total=smc_mask.sum(), desc='Reading MRI to memory'):
		dic[it] = im
		
		mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
		data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx


	target = (metadata[smc_mask].Label == second_class).values.astype('int32')
	



	X_train, y_train = data[0], target[0]
	X_test, y_test = data[0], target[0]


	net = build_net()
	invnet = build_invnet(net)

	prediction = lasagne.layers.get_output(net['prob'])

	m1 = lasagne.layers.get_output(net['voxres2_bn1'])
	m2 = lasagne.layers.get_output(net['voxres3_bn1'])
	m3 = lasagne.layers.get_output(net['voxres5_bn1'])
	m4 = lasagne.layers.get_output(net['voxres6_bn1'])
	m5 = lasagne.layers.get_output(net['voxres8_bn1'])
	m6 = lasagne.layers.get_output(net['voxres9_bn1'])
	m7 = lasagne.layers.get_output(net['fc11'])


	backprop = lasagne.layers.get_output(invnet['conv1a'])


	loss = lasagne.objectives.categorical_crossentropy(prediction,
													   target_var)
	loss = loss.mean()

	params = lasagne.layers.get_all_params(net['prob'], trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, 0.001)
	test_prediction = lasagne.layers.get_output(net['prob'],
												deterministic=True)
	test_loss = categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	train_fn = theano.function([input_var, target_var], loss,
							   updates=updates)
	val_fn = theano.function([input_var, target_var], test_loss)
	test_fn = theano.function([input_var], test_prediction)

	prob_weights = net['prob'].W.get_value()
	#prob_weights_diff = np.repeat([prob_weights[:, 0] - prob_weights[:, 1]], 3, axis=0)
	#prob_weights_diff = np.repeat([prob_weights[:, 0]], 3, axis=0) #AD
	prob_weights_diff = np.repeat([prob_weights[:, 1]], 1, axis=0) #NC
	print(np.shape(prob_weights_diff))

	mids = theano.function([input_var, input_var2], [m1, m2, m3, m4, m5, m6, m7, backprop])

	with np.load('results_resnet/model0.npz') as f:
		lasagne.layers.set_all_param_values(net['prob'], np.array(f['arr_0']))

	logging.debug("Done building net")

	eps = []
	tr_losses = []
	val_losses = []
	val_accs = []
	val_rocs = []

	logging.debug("Starting training...")
	den = X_train.shape[0] / batchsize

	train_err = 0
	train_batches = 0
	start_time = time.time()

	val_err = 0
	val_batches = 0
	preds = []
	targ = []


	inputs = [X_test]
	targets = [y_test]
	err = val_fn(inputs, targets)
	val_err += err
	val_batches += 1
	out = test_fn(inputs)
	mid_vals = mids(inputs, prob_weights_diff)
	
	print(out)
	#save
	print(targets)

	aaa = inputs[0][0]
	aaa_ = mid_vals[7][0][0]
	a_max = np.unravel_index(np.argmax(aaa_[20:70, 20:60, 40:70], axis=None), [50, 40, 30]) 
	_a_max = [a_max[0], a_max[1], a_max[2]]
	a_max = np.add(_a_max, [20, 20, 40])
	print(a_max)
	aa_max = np.amax(aaa_)
	aa_min = np.amin(aaa_)
	
	#save orig

	make_image(np.reshape(aaa[:, a_max[1], :], [110, 110]), 'data/test1/1.png', 'bone', size=(1, 1), dpi=110)
	make_image(np.reshape(aaa[a_max[0], :, :], [110, 110]), 'data/test1/2.png', 'bone', size=(1, 1), dpi=110)
	make_image(np.reshape(aaa[:, :, a_max[2]], [110, 110]), 'data/test1/3.png', 'bone', size=(1, 1), dpi=110)

	seg = nib.load('data/test1/out_subcortical_seg_prec.nii.gz')
	seg_val = seg.get_data()
	
	templ = nib.load('data/test1/tmp/r_tmp.nii.gz')
	templ2 = nib.load('data/test1/tmp/MNI_subcortical_mask.nii.gz')
	templ_val = templ.get_data() + templ2.get_data()
	templ_val[templ_val > 1] = 1

	preproc = templ_val * aaa

	aaa_ = templ_val * aaa_

	relarr = np.reshape(aaa_[:, a_max[1], :], [110, 110])
	relarr[relarr<aa_max*0.2] = float('nan')
	make_masked_image(relarr, np.reshape(preproc[:, a_max[1], :], [110, 110]), 'data/test1/_1.png', 'seismic', size=(1, 1), dpi=110, vmin=-aa_max, vmax=aa_max)
	relarr = np.reshape(aaa_[a_max[0], :, :], [110, 110])
	relarr[relarr<aa_max*0.2] = float('nan')
	make_masked_image(relarr, np.reshape(preproc[a_max[0], :, :], [110, 110]), 'data/test1/_2.png', 'seismic', size=(1, 1), dpi=110, vmin=-aa_max, vmax=aa_max)
	relarr = np.reshape(aaa_[:, :, a_max[2]], [110, 110])
	relarr[relarr<aa_max*0.2] = float('nan')
	make_masked_image(relarr, np.reshape(preproc[:, :, a_max[2]], [110, 110]), 'data/test1/_3.png', 'seismic', size=(1, 1), dpi=110, vmin=-aa_max, vmax=aa_max)




	make_image(np.reshape(preproc[:, a_max[1], :], [110, 110]), 'data/test1/__1.png', 'bone', size=(1, 1), dpi=110)
	make_image(np.reshape(preproc[a_max[0], :, :], [110, 110]), 'data/test1/__2.png', 'bone', size=(1, 1), dpi=110)
	make_image(np.reshape(preproc[:, :, a_max[2]], [110, 110]), 'data/test1/__3.png', 'bone', size=(1, 1), dpi=110)


	seg_val[seg_val==0] = float('nan')
	make_masked_image(np.reshape(seg_val[:, a_max[1], :], [110, 110]), np.reshape(preproc[:, a_max[1], :], [110, 110]), 'data/test1/___1.png', 'gist_rainbow', size=(1, 1), dpi=110)
	make_masked_image(np.reshape(seg_val[a_max[0], :, :], [110, 110]), np.reshape(preproc[a_max[0], :, :], [110, 110]), 'data/test1/___2.png', 'gist_rainbow', size=(1, 1), dpi=110)
	make_masked_image(np.reshape(seg_val[:, :, a_max[2]], [110, 110]), np.reshape(preproc[:, :, a_max[2]], [110, 110]), 'data/test1/___3.png', 'gist_rainbow', size=(1, 1), dpi=110)


	'''
	plt.imshow(np.reshape(aaa[:, a_max[1], :], [110, 110]), cmap = 'bone')
	plt.axis('off')
	plt.savefig('data/test1/1.png',bbox_inches='tight', pad_inches=0)
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(aaa[a_max[0], :, :], [110, 110]), cmap = 'bone')
	plt.savefig('data/test1/2.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(aaa[:, :, a_max[2]], [110, 110]), cmap = 'bone')
	plt.savefig('data/test1/3.png')
	plt.clf()
	plt.cla()

	#save cam
	plt.imshow(np.reshape(aaa_[:, a_max[1], :], [110, 110]), cmap = 'seismic', vmin = aa_min, vmax = aa_max)
	plt.savefig('data/test1/_1.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(aaa_[a_max[0], :, :], [110, 110]), cmap = 'seismic', vmin = aa_min, vmax = aa_max)
	plt.savefig('data/test1/_2.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(aaa_[:, :, a_max[2]], [110, 110]), cmap = 'seismic', vmin = aa_min, vmax = aa_max)
	plt.savefig('data/test1/_3.png')
	plt.clf()
	plt.cla()

	seg = nib.load('data/test1/out_subcortical_seg_prec.nii.gz')
	seg_val = seg.get_data()
	
	templ = nib.load('data/test1/tmp/r_tmp.nii.gz')
	templ2 = nib.load('data/test1/tmp/MNI_subcortical_mask.nii.gz')
	templ_val = templ.get_data() + templ2.get_data()
	templ_val[templ_val > 1] = 1


	#save preprocessed
	preproc = templ_val * aaa
	plt.imshow(np.reshape(preproc[:, a_max[1], :], [110, 110]), cmap = 'bone')
	plt.savefig('data/test1/__1.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(preproc[a_max[0], :, :], [110, 110]), cmap = 'bone')
	plt.savefig('data/test1/__2.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(preproc[:, :, a_max[2]], [110, 110]), cmap = 'bone')
	plt.savefig('data/test1/__3.png')
	plt.clf()
	plt.cla()
	

	#save seg
	plt.imshow(np.reshape(seg_val[:, a_max[1], :], [110, 110]), cmap = 'cubehelix')
	plt.savefig('data/test1/___1.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(seg_val[a_max[0], :, :], [110, 110]), cmap = 'cubehelix')
	plt.savefig('data/test1/___2.png')
	plt.clf()
	plt.cla()
	plt.imshow(np.reshape(seg_val[:, :, a_max[2]], [110, 110]), cmap = 'cubehelix')
	plt.savefig('data/test1/___3.png')
	plt.clf()
	plt.cla()
	'''
	key = {}

	key['right lateral ventricle'] = np.sum(aaa_[np.isin(seg_val, [1])]) / (np.sum(seg_val))
	key['left lateral ventricle'] = np.sum(aaa_[np.isin(seg_val, [2])]) / (np.sum(seg_val)/2)
	key['right caudate'] = np.sum(aaa_[np.isin(seg_val, [3])]) / (np.sum(seg_val)/3)
	key['left caudate'] = np.sum(aaa_[np.isin(seg_val, [4])]) / (np.sum(seg_val)/4)
	key['right putamen'] = np.sum(aaa_[np.isin(seg_val, [5])]) / (np.sum(seg_val)/5)
	key['left putamen'] = np.sum(aaa_[np.isin(seg_val, [6])]) / (np.sum(seg_val)/6)
	key['right pallidum'] = np.sum(aaa_[np.isin(seg_val, [7])]) / (np.sum(seg_val)/7)
	key['left pallidum'] = np.sum(aaa_[np.isin(seg_val, [8])]) / (np.sum(seg_val)/8)
	key['right hippocampus'] = np.sum(aaa_[np.isin(seg_val, [9])]) / (np.sum(seg_val)/9)
	key['left hippocampus'] = np.sum(aaa_[np.isin(seg_val, [10])]) / (np.sum(seg_val)/10)
	key['right amygdala'] = np.sum(aaa_[np.isin(seg_val, [11])]) / (np.sum(seg_val)/11)
	key['left amygdala'] = np.sum(aaa_[np.isin(seg_val, [12])]) / (np.sum(seg_val)/12)

	total = np.sum(key.values())
	key = {k: v / total for k, v in key.iteritems()}

	max_value = max(key.values())
	max_keys = [k for k, v in key.items() if v == max_value]

	print(max_value)
	print(max_keys)

	f = open('data/test1/result.txt', 'w')
	f.write('result(AD vs. NC): ' + str(out) + '\nkeyword: ' + str(max_keys) + '\nrelevance score: ' + str(max_value) + '\%')
	f.close()


### Run training and save results

# In[ ]:

run_training('AD', 'Normal', './results_resnet/ad_vs_norm')

