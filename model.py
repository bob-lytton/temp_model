import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention import AttentionLayer
from locked_dropout import LockedDropout
from ON_LSTM import ONLSTMStack
from utils import build_tok2i
from samplers import GreedySampler, MultinomialSampler, StochasticSampler, TopkSampler, PolicyCorrectSampler

"""
Take in a sentence, return its encoded embedding and hidden states (for attention)\\
TODO: encoder_output required by decoder model: encoder states, encoder hidden, attention mask
TODO: add attention layer
"""
class ONLSTMEncoder(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, chunk_size, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(ONLSTMEncoder, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )
        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(
            self.encoder, input,
            dropout=self.dropoute if self.training else 0
        )

        emb = self.lockdrop(emb, self.dropouti)

        raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
        self.distance = distances

        output = self.lockdrop(raw_output, self.dropout)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        else:
            return result, hidden

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

class LSTMDecoder(nn.Module):
	"""
	inspired from 'non-monotonic generation'\\
	copied from 'non-monotonic generation'...\\
	Though the name is 'decoder', this class is in fact the whole model, including encoder and decoder.\\
    self.encoder is the BOWEncoder according to the train.py.\\
    However, it provides an interface for different encoders.\\
    self.decode is the decoder, elected from forward_decode_attention() or forward_decode()
	"""
	def __init__(self, config, tok2i, sampler, encoder):
		super(LSTMDecoder, self).__init__()
		self.fc_dim = config['fc_dim']  # fc?
		self.dec_lstm_dim = config['dec_lstm_dim']
		self.dec_n_layers = config['dec_n_layers']  # decoder layers number
		self.n_classes = config['n_classes']        # number of token classes
		self.word_emb_dim = config['word_emb_dim']  # dimension of word embedding
		self.device = config['device']
		self.longest_label = config['longest_label']
		self.model_type = config['model_type']
		self.aux_end = config.get('aux_end', False)
		self.encoder = encoder		# encoder is ONLSTMEncoder

		# -- Decoder
		self.dec_lstm_input_dim = config.get('dec_lstm_input_dim', self.word_emb_dim)
		self.dec_lstm = nn.LSTM(self.dec_lstm_input_dim, self.dec_lstm_dim, self.dec_n_layers, batch_first=True)    # use torch implemented LSTM
		self.dec_emb = nn.Embedding(self.n_classes, self.word_emb_dim)
		if config['nograd_emb']:
			self.dec_emb.weight.requires_grad = False
		self.dropout = nn.Dropout(p=config['dropout'])

		# Layers for mapping LSTM output to scores
		self.o2emb = nn.Linear(self.dec_lstm_dim, self.word_emb_dim)
		# Optionally use the (|V| x d_emb) matrix from the embedding layer here.
		if config['share_inout_emb']:   # in bagorder, default is True
			self.out_bias = nn.Parameter(torch.zeros(self.n_classes).uniform_(0.01))
			self.emb2score = lambda x: F.linear(x, self.dec_emb.weight, self.out_bias)  # emb2score(x) = x*dec_emb.weight + out_bias
		else:
			self.emb2score = nn.Linear(self.word_emb_dim, self.n_classes)

		self.register_buffer('START', torch.LongTensor([tok2i['<s>']]))
		self.sampler = sampler
		self.end = tok2i['<end>']

		if self.aux_end:
			self.o2stop = nn.Sequential(nn.Linear(self.dec_lstm_dim, self.word_emb_dim),
										nn.ReLU(),
										self.dropout,
										nn.Linear(self.word_emb_dim, 1),
										nn.Sigmoid())

		if self.model_type == 'translation':
			self.enc_to_h0 = nn.Linear(config['enc_lstm_dim'] * config['num_dir_enc'],
										self.dec_n_layers * self.dec_lstm_dim)
			self.attention = AttentionLayer(input_dim=self.dec_lstm_dim,
											hidden_size=self.dec_lstm_dim,
											bidirectional=config['num_dir_enc'] == 2)
			self.decode = self.forward_decode_attention
			self.decode = self.forward_decode	# temporarily use this
			self.dec_emb.weight = self.encoder.emb.weight
		else:
			self.decode = self.forward_decode   # in bagorder, decode is forward_decode

	def o2score(self, x):
		x = self.o2emb(x)   # calculate embedding
		x = F.relu(x)
		x = self.dropout(x)
		x = self.emb2score(x)
		return x

	def forward(self, xs=None, oracle=None, max_steps=None, num_samples=None, return_p_oracle=False):   # xs means input strings
		B = num_samples if num_samples is not None else xs.size(0)  # B is number of samples
		encoder_output = self.encode(xs)
		hidden = self.init_hidden(encoder_output if encoder_output is not None else B)  # initialize hidden each time runs forward
		scores = []
		samples = []
		p_oracle = []
		self.sampler.reset(bsz=B)   # calling stack seems to be deep
		if self.training:
			done = oracle.done()
			xt = self.START.detach().expand(B, 1)   # START is a register buffer
			t = 0
			while not done:
				hidden = self.process_hidden_pre(hidden, xt, encoder_output)    # just return hidden
				score_t, _, hidden = self.decode(xt, hidden, encoder_output)    # in bagorder, self.decode is self.forward_decode
				xt = self.sampler(score_t, oracle, training=True)               # here the model samples from the oracle (?)
				hidden = self.process_hidden_post(hidden, xt, encoder_output)   # in bagorder, hidden[0] + encoder_output
				p_oracle.append(oracle.distribution())                      # distribution generated by oracle
				oracle.update(xt)                                           # p_oracle adds the output of oracle
				samples.append(xt)
				scores.append(score_t)
				t += 1
				done = oracle.done()
				if max_steps and t == max_steps:
					done = True

			self.longest_label = max(self.longest_label, t)
		else:		# evaluating and using
			with torch.no_grad():
				xt = self.START.detach().expand(B, 1)
				for t in range(self.longest_label):
					hidden = self.process_hidden_pre(hidden, xt, encoder_output)
					score_t, _, hidden = self.decode(xt, hidden, encoder_output)
					xt = self.sampler(score_t, oracle=None, training=False)
					hidden = self.process_hidden_post(hidden, xt, encoder_output)
					scores.append(score_t)
					samples.append(xt)

		samples = torch.cat(samples, 1)
		if not self.aux_end:
			scores = torch.cat(scores, 1)
		if return_p_oracle:
			p_oracle = torch.stack(p_oracle, 1)
			return scores, samples, p_oracle, encoder_output
		return scores, samples, encoder_output

	def encode(self, xs):
		if self.model_type == 'unconditional':
			encoder_output = None
		elif self.model_type == 'bagorder':
			encoder_output = self.encoder(xs)
			encoder_output = encoder_output.unsqueeze(0).expand(self.dec_n_layers, xs.size(0),
																self.dec_lstm_dim).contiguous()
		elif self.model_type == 'translation':
			encoder_output = self.encoder(xs)
		else:
			raise NotImplementedError('Unsupported model type %s' % self.model_type)
		return encoder_output

	def forward_decode(self, xt, hidden, encoder_output):
		"""
		encoder_output does nothing here
		"""
		xes = self.embed_input(xt)  # make xt into embedding
		xes = self.dropout(xes)
		lstm_output, hidden = self.dec_lstm(xes, hidden)
		scores = self.o2score(lstm_output)
		if self.aux_end:
			stop = self.o2stop(lstm_output).squeeze(2)
			scores = (scores, stop)
		return scores, lstm_output, hidden

	def forward_decode_attention(self, xt, hidden, encoder_output):
		enc_states, enc_hidden, attn_mask = encoder_output
		xes = self.embed_input(xt)
		xes = self.dropout(xes)
		lstm_output, hidden = self.dec_lstm(xes, hidden)
		lstm_output, _ = self.attention(lstm_output, hidden, (enc_states, attn_mask))
		scores = self.o2score(lstm_output)
		if self.aux_end:
			stop = self.o2stop(lstm_output).squeeze(2)
			scores = (scores, stop)
		return scores, lstm_output, hidden

	def embed_input(self, xt):
		return self.dec_emb(xt)

	def init_hidden(self, encoder_output):
		N = self.dec_n_layers
		D = self.dec_lstm_dim
		if self.model_type == 'unconditional':
			B = encoder_output
			hidden = (torch.zeros(N, B, D, device=self.device),
						torch.zeros(N, B, D, device=self.device))
		elif self.model_type == 'bagorder':
			B = encoder_output.size(1)
			hidden = (encoder_output,
						torch.zeros(N, B, D, device=self.device))
		elif self.model_type == 'translation':
			_, last_hidden, _ = encoder_output
			B = last_hidden.size(0)
			hidden = (self.enc_to_h0(last_hidden).view(B, N, D).transpose(0, 1),
						torch.zeros(N, B, D, device=self.device))
		else:
			raise NotImplementedError('Unsupported model type %s' % self.model_type)
		return hidden

	def process_hidden_pre(self, hidden, input_token, encoder_output):
		return hidden

	def process_hidden_post(self, hidden, sampled_token, encoder_output):
		"""Add the encoder embedding for bagorder task"""
		if self.model_type == "bagorder":
			hidden = (hidden[0] + encoder_output, hidden[1])
		return hidden

if __name__ == "__main__":
	# -- MODEL
	tokens = ['<s>', '<p>', '<end>']
	tok2i = build_tok2i(tokens)
	model_config = {
		'fc_dim':        		512, 			# args.fc_dim,
		'dec_lstm_dim':  		1024, 			# args.dec_lstm_dim,
		'dec_n_layers':  		2, 				# args.dec_n_layers,
		'n_classes':     		len(tok2i), 	# args.n_classes,
		'word_emb_dim':  		300,  			# glove
		'dropout':       		0.2, 			# args.dropout,
		'device':        		'cpu', 			# str(args.device),
		'longest_label': 		10,  			# gets adjusted during training
		'share_inout_emb': 		'true',			# args.share_inout_emb,
		'nograd_emb': 			'true',			# args.nograd_emb,
		'batch_size': 	 		32,				# args.batch_size,
		'model_type': 	 		'translator', 	# args.model_type,
		'aux_end': 				'true'			# args.aux_end # if string x is all in lowercase, x is input string
	}
	n_classes = len(tok2i)
	print(tokens, tok2i, model_config, n_classes)
	ntoken = len(tokens)
	h_dim = 10
	emb_dim = 2
	nlayers = 2
	chunk_size = 1

	model = LSTMDecoder(model_config, tok2i, GreedySampler(model_config['aux_end']), 
						ONLSTMEncoder(ntoken, emb_dim, h_dim, chunk_size, nlayers))
	
	print('Init succefully!')