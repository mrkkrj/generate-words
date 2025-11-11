import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%matplotlib inline # --> for Jupyter context

# ---> As in MLP part 3

# read all the words
words = open('names.txt', 'r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])

# build the vocabulary of characters
chars = sorted(list(set(''.join(words))))
stoi = { s:i+1 for i,s in enumerate(chars) }
stoi['.'] = 0 # start/stop special char
itos = { i:s for s,i in stoi.items() }
vocab_size = len(itos)
print(itos)
print(vocab_size)

# shuffle up the words
import random
random.seed(44)
random.shuffle(words)

# build the dataset
#block_size = 3 # context length - how many chars needed to predict the next one?
block_size = 8 # -> even, to be able to build a binary tree !

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(ix)
            context = context[1:] + [ix] # crop dots & append

        X = torch.tensor(X) # inputs
        Y = torch.tensor(Y) # results
        print(X.shape, Y.shape)
        return X, Y

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])      # 80% for training
Xdev, Ydev = build_dataset(words[n1:n2])  # 10% for dev opt.
Xtst, Ytst = build_dataset(words[n2:])    # 10% for test

for x,y in zip(Xtr[:20], Ytr[:20]):
    print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])


#
# (near copy of...) classes developed in Part 3
#

# ------------------------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weights = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])
        
# ------------------------------------------------------------------------------
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # params (trained with backprop)
        self.gamma = torch.zeros(dim)
        self.beta = torch.ones(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)        

    def __call__(self, x):
        # forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0 # dim we want reduce over (old MLP impl...)
            elif x.ndim == 3:
                dim = (0, 1) # over both initial dims (for WaveNet impl!!!)
            xmean = x.mean(dim, keepdim=True) # mean values from batch!
            xvar = x.var(dim, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance / zero mean
        self.out = self.gamma * xhat + self.beta
        # update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

# ------------------------------------------------------------------------------
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []        

# new:

# ------------------------------------------------------------------------------
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]  
    
# ------------------------------------------------------------------------------
class Flatten:       
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1) # -1: torch will find suitable dim for that!
        return self.out
    def parameters(self):
        return [] 

# ------------------------------------------------------------------------------
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n # number of elements to concatenate in the last dim!        

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n) 
        if x.shape[1] == 1:
            x = x.squeeze(1) # fallback to Flatten!!!
        self.out = x                
        return self.out
    
    def parameters(self):
        return [] 
            
# ------------------------------------------------------------------------------
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        # params of all layers, streched into one list
        return [p for layer in self.layers for p in layer.parameters()] 
    



#
# MLP neural net 
#

torch.manual_seed(44) # mk. reproducible

n_emb = 10      # dim. of the character embedding vectors
#n_hidden = 200  # neurons in the hidden layer
n_hidden = 68  # changed for FlattenConsecutive, to reduce no of params to the previous count (i.e. Flatten())

# -> now increase the size of the model -> longer trainig, better loss value!
n_emb =  24     # dim. of the character embedding vectors
n_hidden = 128  # neurons in the hidden layer

#C  = torch.randn((vocab_size, n_emb))
model = Sequential([
    Embedding(vocab_size, n_emb), # added
    #Flatten(), # added
    FlattenConsecutive(block_size), # like plain Flatten()!    
    Linear(n_emb * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])


#
#  - WaveNet's idea - don't squash contect inito a single vector but use a binary tree!!!
#

# note - matrix mult works only on the last dim, the dims before are left unchanged!
#  - e.g.: torch.randn(4, 5, 2, 80) @ torch.randnd(80, 200) 
#           -> shape: [4, 5, 2, 200]
#
#  we want to group 8 vals like: (1 2) (3 4) (5 6) (7 8) so we will take:
#   - torch.randn(4, 4, 20) -> i.e. 4 groups of 2, and each of them is 10-dim vector!
#       @ torch.randn(20, 200)
#  - So we need extended Flatten 
# e = torch.randn(4, 8, 10) -> we want it to be (4, 4, 20)! 
#    -> take: e[:, ::2, :] - even elems, e[:, 1::2, :] - odd elems, AND concat them!
#     - note: 1::2 - start with 1, step=2 (odd elems), ::2 - start with 0, step size=2
#   -> or just request new dims from view() !!!

#  ----> SO: group input elems in pairs, 3 times!
model = Sequential([
    Embedding(vocab_size, n_emb), # added
    FlattenConsecutive(2), Linear(n_emb * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    # last layer: make less confident!    
    model.layers[-1].weight *= 0.1 #  -> IF: no batch norm.
    #layers[-1].gamma *= 0.1 # we use BatchNorm now! --> OPEN::: why not this one???

#parameters = [C] + [p for layer in layers for p in layer.parameters()]
parameters = model.parameters()
print("num param=", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

# Debug: inspect shapes:
for layer in model.layers:
    print(layer.__class__.__name__, ':', tuple(layer.out.shape))



# same optimizations as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # constr. minibatch
    ix  = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    # emb = C[Xb] # emb. chars in vectors
    # x = emb.view(emb.shape[0], -1) # concatenate
    logits = model(Xb) 
    loss = F.cross_entropy(logits, Yb) # loss fn

    # backward pass
    for p in parameters:
        p.grad = None        
    loss.backward() 

    # gradient descent
    lr = 0.1 if i < 100000 else 0.01  # learning rate decay after 100,000!
    for p in parameters:
        p.data += -lr * p.grad 

    # track stats
    if i % 10000 == 0: # print every NN steps
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')
    lossi.append(loss.log10().item())

# show losses
plt.plot(lossi)

# plot the means  of 1000 lossi values
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))


# put layers into eval mode (needed for batchnorm esp.!)
for layer in model.layers:
    layer.training = False

@torch.no_grad() # disable gradient tracking
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'dev': (Xtr, Ytr),
        'test': (Xtr, Ytr)
        }[split]
    # emb = C[x] # (N, vocab_size)
    # x = emb.view(emb.shape[0], -1) # concat int (N, block_size * n_embed)
    logits = model(x)
    loss = F.cross_entropy(logits, Yb)
    print(split, loss.item())

split_loss('train')
split_loss('dev')

# sample form the model

for _ in range(20):
    out = []
    context = [0] * block_size # init with all ...
    while True:
        # forward pass of the NN
        # emb = C[torch.tensor([context])] # (1, block_size, n_embed)
        # x = emb.view(1, -1) # concatenate the vectors
        # for layer in layers:
        #     x = layer(x)
        # logits = x
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1) # output = softmax layer
        # sample from the distr.
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift context window
        context = context[1:] + [ix]
        out.append(ix)
        # EOW special char?
        if ix == 0:
            break

    # decode & print generated word
    print(''.join(itos[i] for i in out ))

# EOF
