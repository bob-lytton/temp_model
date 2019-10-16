import argparse
import hashlib
import math
import operator
import os
import time
from functools import reduce
from random import randint, random, sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import data
import model
from model import ONLSTMEncoder, LSTMDecoder
from splitcross import SplitCrossEntropyLoss
from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model, training and evaluating.')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--hidsize', type=int, default=512,
					help='size of hidden states in lstm')
parser.add_argument('--nodesize', type=int, default=512,
					help='size of nodes presentation in tree/graph')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--encoderlr', type=float, default=30,
                    help='initial encoder learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.4,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--cluster', action='store_true',
					help='Use dlib cluster')
args = parser.parse_args()
args.tied = True

def model_save(fn, model_encoder, model_decoder, optimizer):
	"""
	save model parameters to a binary file fn
	"""
	if args.cluster:
		pass
	with open(fn, 'wb') as f:
		torch.save([model, criterion, optimizer], f)

def model_load(fn):
	"""
	load model parameters from the binary file fn
	"""
	if args.cluster:
		pass
	with open(fn, 'rb') as f:
		model, criterion, optimizer = torch.load(f)
	return model, criterion, optimizer

def data_load(corpus, train_batch_size=args.batch_size, eval_batch_size=10, test_batch_size=1):
	train_data = batchify(corpus.train, args.batch_size, args)
	val_data = batchify(corpus.valid, eval_batch_size, args)
	test_data = batchify(corpus.test, test_batch_size, args)
	return train_data, val_data, test_data

# Initiate optimizer
def init_optimizer(optim_type, params):
	optimizer = None
	# Ensure the optimizer is optimizing params, 
	# which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
	if optim_type == 'sgd':
		optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
		return optimizer
	if optim_type == 'adam':
		optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
		return optimizer, scheduler

# train
# TODO: data related
# TODO: fuse the encoder and decoder into a single model class
def train(epoch_num, corpus, train_data, optimizer, criterion):
	total_loss = 0
	start_time = time.time()
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(args.batch_size)
	batch, i = 0, 0
	while i < train_data.size(0) - 1 - 1:
		bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2
		# Prevent excessively small or negative sequence lengths
		seq_len = max(5, int(np.random.normal(bptt, 5)))	# TODO: why np.random.normal?

		lr2 = optimizer.param_groups[0]['lr']
		optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
		model.train()	# set the model to training mode
		data, targets = get_batch(train_data, i, args, seq_len=seq_len)

		hidden = repackage_hidden(hidden)
		optimizer.zero_grad()

		# Get the output of the model, then calculate loss
		output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
		raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
		loss = raw_loss
        # Activiation Regularization
		if args.alpha:
			loss = loss + sum(
				args.alpha * dropped_rnn_h.pow(2).mean()
				for dropped_rnn_h in dropped_rnn_hs[-1:]
			)
        # Temporal Activation Regularization (slowness)
		if args.beta:
			loss = loss + sum(
				args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
				for rnn_h in rnn_hs[-1:]
			)

		loss.backward()
		if args.clip:
			torch.nn.utils.clip_grad_norm_(params, args.clip)
		optimizer.step()

		total_loss += raw_loss.data
		optimizer.param_groups[0]['lr'] = lr2
		if batch % args.log_interval == 0 and batch > 0:
			cur_loss = total_loss.item() / args.log_interval
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
				'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
				epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
						elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
			total_loss = 0
			start_time = time.time()

		batch += 1
		i += seq_len

if __name__ == "__main__":

	# Set the random seed manually for reproducibility.
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		if not args.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			torch.cuda.manual_seed(args.seed)

	# Load corpus
	fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
	if os.path.exists(fn):
		print('Loading cached dataset...')
		corpus = torch.load(fn)
	else:
		print('Producing dataset...')
		corpus = data.Corpus(args.data)
		torch.save(corpus, fn)

		
	# Load data
	train_data, val_data, test_data = data_load(corpus)

	# Initiate model
	ntokens = len(corpus.dictionary)
	model_encoder = ONLSTMEncoder(ntokens, args.hdim, args.emsize,
								 args.nlayers, args.chunksize, args.wdrop, args.dropouth)
	model = LSTMDecoder()

	# Load criterion
	criterion = None
	if not criterion:
		splits = []
		if ntokens > 500000:
			# One Billion
			# This produces fairly even matrix mults for the buckets:
			# 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
			splits = [4200, 35000, 180000]
		elif ntokens > 75000:
			# WikiText-103
			splits = [2800, 20000, 76000]
		print('Using', splits)
		criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

	# Counting params
	params = list(model.parameters()) + list(criterion.parameters())
	total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params)
	print('Args:', args)
	print('Model total parameters:', total_params)

	if args.optimizer == 'sgd':
		optimizer = init_optimizer(args.optimizer, params)
	elif args.optimizer == 'adam':
		optimizer, scheduler = init_optimizer(args.optimizer, params)

	# If use saved model to continue training
	if args.resume:
		print('Resuming models ...')
		model, criterion, optimizer = model_load(args.resume)	# reload model and optimizer
		if args.wdrop:
			for rnn in model.rnn.cells:
				rnn.hh.dropout = args.wdrop

	# Load to GPU
	if args.cuda:
		# model_encoder = model_encoder.cuda()
		model = model.cuda()
		criterion = criterion.cuda()

	#--- START TRAINING ---#
	# use Ctrl+C to break out of training at any point
	try:
		print('--TRAINDATA--', train_data)
		for epoch in range(args.epochs):
			train(epoch, corpus, train_data, optimizer, criterion)		# training for one epoch
	except KeyboardInterrupt:
		print('-' * 89)
		print('Exiting from training early')
		model_save('checkpoint'+str(int(time.time()))+'.pth', model_encoder, model_decoder, optimizer)
		print('Model saved!')
		# print('| End of training | pos loss/epoch {:5.2f} | decoder ppl/epoch {:5.2f}'.format(mean(global_pos_losses), mean(global_decoder_losses)))
