import torch
import jsonlines
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
	print('data:', data)
	print('type of data:', type(data))
	# Evenly divide the data across the bsz batches.
	data = data.reshape((bsz, -1)).T.contiguous()	# contiguous memory in C order
	# data = torch.Tensor(data)
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

def load_jsonl(filepath, log=True):
    with open(filepath, 'r') as f:
        with jsonlines.Reader(f) as reader:
            data = [line for line in reader]
    if log:
        print("%d sentences" % (len(data)))
    return data

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

# --- Pytorch
def masked_softmax(vec, mask, dim=1, epsilon=1e-40, alpha=0.):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float() + alpha
    masked_sums = torch.clamp(masked_exps.sum(dim, keepdim=True), min=epsilon)
    ps = masked_exps / masked_sums
    return ps

