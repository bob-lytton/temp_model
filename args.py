import time

def model_args(parser):
    # dataset
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')

    # encoder model setup
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--hidsize', type=int, default=512,
                        help='size of hidden states in lstm')
    parser.add_argument('--nodesize', type=int, default=512,
                        help='size of nodes presentation in tree/graph')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='number of units per chunk')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')

    # decoder model setup
    parser.add_argument("--dec-lstm-dim", type=int, default=1024,
                        help='decoder LSTM dimension')
    parser.add_argument("--dec-n-layers", type=int, default=2,
                        help='number of decoder layers')
    parser.add_argument("--fc-dim", type=int, default=512,
                        help='')
    parser.add_argument("--num_layers_enc", type=int, default=1)
    parser.add_argument("--share_inout_emb", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='use embedding weights for RNN output-to-scores')
    parser.add_argument("--nograd_emb", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="don't update embedding weights when True")
    parser.add_argument("--decoder", choices=['LSTMDecoder'], default='LSTMDecoder',
                        help='choose decoder model from (LSTMDecoder)')
    parser.add_argument("--model-type", choices=['bagorder', 'unconditional', 'translation', 'transformer'], default='unconditional',
                        help='task type of the model (bagorder, unconditional, translation)')
    parser.add_argument("--aux-end", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='i don\'t know what it is for either...')

    # epochs and batchsize and sequence length
    parser.add_argument('--epochs', type=int, default=1000,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')

    # dropout setup
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.25,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.5,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.4,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

    # sampler and rollin
    parser.add_argument("--rollin", choices=["mixed"], default='mixed', 
                        help='learned rollin via --rollin-beta 0.0')
    parser.add_argument("--rollin-mix-type", choices=["trajectory", "state"], default="state", 
                        help="mix rollin at the trajectory or state level")					
    parser.add_argument('--training-sampler', type=str, default='policy_correct_greedy',
                        help='Choose sampler for training')
    parser.add_argument('--eval-sampler', type=str, default='greedy',
                        help='Choose sampler for evaluation')
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--encoderlr', type=float, default=30,
                        help='initial encoder learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')

    # devices
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--cluster', action='store_true',
                        help='Use dlib cluster (not implemented yet)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    # oracle
    parser.add_argument("--oracle", choices=['uniform', 'leftright'], default='uniform',
                        help='oracle policy for generating sentence')
    parser.add_argument("--rollin-beta", type=float, default=1.0,
                        help="probability of using oracle for a full roll-in")
    parser.add_argument("--beta-step", type=float, default=0.01,
                        help="per-epoch decrease of rollin-beta")
    parser.add_argument("--beta-min", type=float, default=0.00,
                        help='min value of rollin_beta')
    parser.add_argument("--beta-burnin", type=int, default=20,
                        help="number of epochs before we start to decrease beta")

    # save and logging
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--resume', type=str, default='',
                        help='path of model to resume')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')