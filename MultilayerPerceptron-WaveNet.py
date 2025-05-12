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
block_size = 3 # context length - how many chars needed to predict the next one?

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
            xmean = x.mean(0, keepdim=True) # mean values from batch!
            xvar = x.var(0, keepdim=True, unbiased=True)
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


#
# MLP neural net 
#

torch.manual_seed(44) # mk. reproducible

n_emb = 10      # dim. of the character embedding vectors
n_hidden = 200  # neurons in the hidden layer

C  = torch.randn((vocab_size, n_emb))

layers = [
    Linear(n_emb * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]

with torch.no_grad():
    # last layer: make less confident!    
    layers[-1].weight *= 0.1 #  -> IF: no batch norm.
    #layers[-1].gamma *= 0.1 # we use BatchNorm now! --> OPEN::: why not this one???

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print("num param=", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True


# same optimizations as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # constr. minibatch
    ix  = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    emb = C[Xb] # emb. chars in vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for layer in layers:
        layer.out.retain_grad() # for tests - needed for histograms
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




..................


#
# visualize histograms
#  - Diagnostic Toos: to check how the network parameters are working!

# visualize histograms - tanh.out
plt.figure(figsize(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]): # exc. last
    if(isinstance(layer, Tanh)):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, saturated: %.2f%%' 
              % (i, layer.__class__.__name__, t.mean(), t.std(), (t.mean() > 0.97).float().mean()*100))
        hx, hy = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} {layer.__class__.__name__}')
plt.legend(legends)
plt.title('activation distribution')


# visualize histograms - gradients
plt.figure(figsize(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]): # exc. last
    if(isinstance(layer, Tanh)):
        t = layer.out.grad
        print('layer %d (%10s): mean %+.2f, saturated: %.2f%%' 
              % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        hx, hy = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} {layer.__class__.__name__}')
plt.legend(legends)
plt.title('gradient distribution')


# visualize histograms - weights gradient
plt.figure(figsize(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2: # weigths!
        print('weight %10s | mean %+f | grad:data ratio %e' 
              % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
        hx, hy = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weigths gradient distribution')

# visualize histograms - update ratios for weights
plt.figure(figsize(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2: # weigths!
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate it on plot! (good rough heuristic)
plt.legend(legends)

# EOF
