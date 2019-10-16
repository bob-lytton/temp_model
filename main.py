import argparse
import hashlib
import math
import operator
import os
import time
from functools import reduce
from itertools import chain
from random import randint, random, sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import data
import model
import samplers
from args import model_args
from losses import sequential_set_no_stop_loss, sequential_set_loss
from model import LSTMDecoder, ONLSTMEncoder
from oracle import Oracle, LeftToRightOracle
from splitcross import SplitCrossEntropyLoss
from utils import (batchify, build_tok2i, get_batch, load_jsonl,
                   repackage_hidden)

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model, training and evaluating.')
model_args(parser)
args = parser.parse_args()
args.tied = True

def model_save(fn, model, criterion, optimizer):
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
		optimizer = torch.optim.SGD(params, lr=args.encoderlr, weight_decay=args.wdecay)
		return optimizer
	if optim_type == 'adam':
		optimizer = torch.optim.Adam(params, lr=args.encoderlr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
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

		oracle = Oracle(data, model.n_classes, tok2i, i2tok, **oracle_flags)

		# Get the output of the model, then calculate loss
		scores, samples, p_oracle, encoder_output = model(data, oracle, return_p_oracle=True)
		output, hidden = encoder_output
		encoder_raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
		encoder_loss = encoder_raw_loss
        # # Activiation Regularization
		# if args.alpha:
		# 	loss = loss + sum(
		# 		args.alpha * dropped_rnn_h.pow(2).mean()
		# 		for dropped_rnn_h in dropped_rnn_hs[-1:]
		# 	)
        # # Temporal Activation Regularization (slowness)
		# if args.beta:
		# 	loss = loss + sum(
		# 		args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
		# 		for rnn_h in rnn_hs[-1:]
		# 	)

		decoder_loss = loss_fn(scores, samples, p_oracle, tok2i['<end>'], **loss_flags)
		loss = [encoder_loss, decoder_loss]
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
			torch.cuda.set_device(args.gpu)
			args.device = torch.device('cuda:%d' % args.gpu)
			torch.cuda.manual_seed(args.seed)
	else:
		args.device = torch.device('cpu')

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
	train_tokens = load_jsonl(args.data+'/'+'train.jsonl')
	valid_tokens = load_jsonl(args.data+'/'+'valid.jsonl')
	test_tokens = load_jsonl(args.data+'/'+'test.jsonl')
	tok2i = build_tok2i(list(chain.from_iterable([d['tokens'] for d in (train_tokens + valid_tokens)])))
	i2tok = {j: i for i, j in tok2i.items()}

	# Add more attributes to args
	args.n_classes = len(tok2i)

	# # Set decoder config
	decoder_config = {
		'fc_dim':        		args.fc_dim,
		'dec_lstm_dim':  		args.dec_lstm_dim,
		'dec_n_layers':  		args.dec_n_layers,
		'n_classes':     		args.n_classes,
		# 'enc_lstm_dim': 		args.enc_lstm_dim,
		'word_emb_dim':  		300,  			# glove
		'dropout':       		args.dropout,
		'device':        		str(args.device),
		'longest_label': 		10,  			# gets adjusted during training
		'share_inout_emb': 		args.share_inout_emb,
		'nograd_emb': 			args.nograd_emb,
		# 'enc_n_layers':			args.num_layers_enc,
		# 'num_dir_enc':			2,
		'batch_size': 	 		args.batch_size,
		'model_type': 	 		args.model_type,
		'aux_end': 				args.aux_end 	# if string x is all in lowercase, x is input string
	}
	sampler = samplers.initialize(args)

	model_encoder = ONLSTMEncoder(ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers, 
								 args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
	model = LSTMDecoder(decoder_config, tok2i, sampler, model_encoder).to(args.device)		# bind ONLSTMEncoder into the model

	# Initiate decode oracle
	oracle_flags = {}
	if 'uniform' in args.oracle:
		Oracle = Oracle
	elif 'leftright' in args.oracle:
		Oracle = LeftToRightOracle

	# Decoder related setup
	# -- loss
	loss_flags = {}
	if args.aux_end:
		loss_fn = sequential_set_loss
	else:
		loss_fn = sequential_set_no_stop_loss
	loss_flags['self_teach_beta'] = args.self_teach_beta

	# -- save things for eval time loading
	with open(os.path.join(args.log_directory, 'model_config.json'), 'w') as f:
		json.dump(model_config, f)
	with open(os.path.join(args.log_directory, 'tok2i.json'), 'w') as f:
		json.dump(tok2i, f)

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
		model_save('checkpoint'+str(int(time.time()))+'.pth', model, criterion, optimizer)
		print('Model saved!')
		# print('| End of training | pos loss/epoch {:5.2f} | decoder ppl/epoch {:5.2f}'.format(mean(global_pos_losses), mean(global_decoder_losses)))
