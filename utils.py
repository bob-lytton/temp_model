import torch
import numpy as np

def repackage_hidden(h):
	"""
	Wraps hidden states in new Tensors to detach them from their history.
	"""
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, args):
	"""
	Transfer the data into batches.
	"""
	# Work out how cleanly we can divide the dataset into bsz parts.
	nbatch = data.size(0) // bsz
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	data = data[0:(nbatch * bsz)]
	print(data)
	print(type(data))
	# Evenly divide the data across the bsz batches.
	data = data.reshape((bsz, -1)).T ###.contiguous()
	data = torch.Tensor(data)
	if args.cuda:
		data = data.cuda()
	return data

def get_batch(source, i, args, seq_len=None, evaluation=False):
	"""
	Fetch a batch from the corpus
	"""
	seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
	data = source[i:i+seq_len]
	target = source[i+1:i+1+seq_len].view(-1)
	return data, target

def build_tok2i(tokens):    
	"""
	The location of an tok first appears in the tokens
	"""
	def _add(t, d):
		if t not in d:
			d[t] = len(d)
	tok2i = {}
	for tok in ['<s>', '<p>', '</s>', '<unk>', '<end>'] + tokens:
		_add(tok, tok2i)
	return tok2i