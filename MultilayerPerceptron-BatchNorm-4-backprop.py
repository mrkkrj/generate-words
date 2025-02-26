import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%matplotlib inline # --> for Jupyter context


# read all the words
words = open('names.txt', 'r').read().splitlines()

print(words[:8])
print(len(words))


# build the vocabulary of characters
chars = sorted(list(set(''.join(words))))

stoi = { s:i+1 for i,s in enumerate(chars) }
stoi['.'] = 0 # start/stop special char

itos = { i:s for s,i in stoi.items() }
vocab_size = len(itos)

print(itos)
print(vocab_size)

# build the dataset
block_size = 3 # context length

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


import random

random.seed(44)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])      # 80% for training
Xdev, Ydev = build_dataset(words[n1:n2])  # 10% for dev opt.
Xtst, Ytst = build_dataset(words[n2:])    # 10% for test

#
# cmp() - utility function to compare gradients
#
def cmp(strg, dt, t):
    ex = torch.all(dt == t.grad).item()
    approx = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{strg:15s} | exact: {str(ex):5s} | approximate: {str(approx):5s} | maxdiff: {maxdiff}')


# MLP neural net impl.

n_emb = 10     # dim. of the character embedding vectors
n_hidden = 200 # neurons in the hidden layer

g = torch.Generator().manual_seed(2147483647) # mk. reproducible
C  = torch.randn((vocab_size, n_emb),            generator=g)
# Layer 1
W1 = torch.randn((n_emb * block_size, n_hidden), generator=g) (5/3)/((n_emb * block_size)**0.5)
b1 = torch.randn(n_hidden,                       generator=g) * 0.01 # using b1 just for fun of it
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),         generator=g) * 0.1 
b2 = torch.randn(vocab_size,                     generator=g) * 0.1  # init chaged here to small numbers
# Batch Norm. params
bnorm_gain = torch.randn((1, n_hidden))*0.1 + 1.0   # Note: init chaged here to small numbers
bnorm_bias = torch.randn((1, n_hidden))*0.1         #   -- to unmask errors in calculations

parameters = [C, W1, b1, W2, b2, bnorm_gain, bnorm_bias]
print("num param=", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True


# train the NN

batch_size = 32
n = batch_size # for convenience
   
# constr. minibatch
ix  = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

# forward pass, "chunkaded" in smaller steps

emb = C[Xb] # emb. chars in vectors
embcat = emb.view(emb.shape[0], -1) # concatenate
# Linear layer 1
hpreact_bn = embcat @ W1 + b1 # hidden layer pre-activation 
# Batch Norm layer
bnmean_i = 1/n*hpreact_bn.sum(0, keepdim=True)
bndiff = hpreact_bn - bnmean_i
bndiff_sq = bndiff**2
bnvar = 1/(n-1)*(bndiff_sq).sum(0, keepdim=True) # Bessel correction: n-1 (as we have small batches!!!)
bnvar_inv = (bnvar + 1e-5)**0.5
bnraw = bndiff * bnvar_inv
hpreact = bnorm_gain * bnraw + bnorm_bias
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
# Linear Layer 2
logits = h @ W2 + b2    # output layer   
# Cross entropy loss <=< F.cross_entropy(logits, Yb)
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # substr max for num. stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1 # if 1/counts_sum - problems with PyTorch backprop!
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
    p.grad = None
for t in [emb, embcat, hpreact_bn, bnmean_i, bndiff, bndiff_sq, bnvar, bnvar_inv, bnraw, 
          hpreact, h, logits, logit_maxes, norm_logits, counts, counts_sum, counts_sum_inv,
          probs, logprobs]:
    t.retain_grad()   
loss.backward() 
print(loss)

#
# Excercise 1: backprop throgh all that manually
#

# dlogprobs i.e loss'(logprobs) derivation:
#   loss = -logprobs[range(n), Yb].mean()
#   loss = -(a + b + c)/3 # the selected logprops
#   dloss/da = - 1/3
#   -> the unselected logprops: no influence, so derivative will be 0!
dlogprobs = torch.zeros_like(logprobs)  # <=> torch.zeros((32,27))
dlogprobs[range(n), Yb] = -1.0/n  # at selected places - set deriv. of the mean()!
cmp('logprobs', dlogprobs, logprobs)    # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# d/dx log(x) = 1/x
dprobs = (1.0 / probs) * dlogprobs  # use chain rule!
cmp('probs', dprobs, probs)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# dcounts_sum_inv: counts.shape=(32,27), counts_sum_inv.shape=(32, 1)
#  c = a * b, but with tensor broadcasting a(3,3)*b(3,1)
#  a11*b1 a12+b1 a13*b1
#  a21*b2 a22*b2 a23*b3
#  a31* ....             --------> c(3,3)
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True) # bc. d/db (a * b) = a, AND tensor broadcasting replicates columns 
                                                         #   so we have to sum up the replicated inputs                                                         
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)   # --> logprobs | exact: true | approx: True | maxdiff: 0.0

dcounts = counts_sum_inv * dprobs   # bc. d/db (a * b) = b, and counts is b in probs=counts*counts_sum_inv
                                    #  no summation required, broardcast is fine: (32,1)*(32,27)

# d/dx  x**-1 = -x**-2
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# dcounts_sum: counts.shape=(32,27), counts_sum.shape=(32, 1)
#   a11 a12 a13 ---> b1 (=a11+a12+a13)
#   a21 a22 a23 ---> b2 (=a21+a22+a23)
#   a31 a32 a33 ---> b3 (=a31+a32+a33)
#    d/da(i,j) b1 = 1, 1, 1 for a(1,j), otherwise 0
dcounts += torch.ones_like(counts) * dcounts_sum # broadcasting here implements the replication of 1's,
                                                 #  AND: += becouse its's a second brach leading to counts!
cmp('counts', dcounts, counts)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: counts = norm_logits.exp() ...
dnorm_logits = (counts # == norm_logits.exp()
) * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# dlogits & dlogit_maxes  (at line: norm_logits = logits - logit_maxes)
# norm_logits: (32,27), logits: (32,27), logit_maxes: (32,1)
#   - c11 c12 c13 = a11 a12 a13 - b1 (b1 b1) --> broadcasting again!!
#   so: need to sum again!
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: logit_maxes = logits.max(1, keepdim=True).values 
#    <=> backprop to logits throgh the 2nd branch!
# - populate the affected values (i.e where max value was found) with 1s!
#     --> as in dlogprobs, but here with one_hot !!! (???? :-O)  [alternative impl.]
dlogits += F.one_hot(logits.max(1).inidces, num_classes=logits.shape[1]) * dlogit_maxes
cmp('logits', dlogits, logits)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: logits = h @ W2 + b2
#  - "+" will broadcast b2 here!!! 
#  --> write out an example on paper, like 3 2x2 matrices: d = a @ b + c (as b2 will be broadcasted!!!)
#  - d11 = a11+b11 + a12+b12 + c1
#  - dLoss/da11 = dLoss/dd11 /*incoming deriv.*/ * b11 /*local deriv*/ + dLoss/dd12 * b12 /*in 2nd column*/
#  - dLoss/da = [dLoss/dd11 dLoss/dd12] * [b11 b21]  --> can be simplified to this matric mult!
#               [dLoss/dd21 dLoss/dd22]   [b21 b22]
#  - dLoss/da = dLoss/dd @ bTransp  -- matrix ops!
# for b: dLoss/db = aTransp @ dLoss/dd
# for c: dLoss/dc1 = dLoss/dd11 * 1 + dLoss/d21 * 1
#  - dLoss/dc = dLoss/dd.sum(0) --> sum the columns!
dh = dlogits @ W2.T # dh is the dLoss/da in the above example!
dW2 = h.T @ dlogits # dW2 is the dLoss/db in the above example
db2 = dlogits.sum(0) # db2 is the dLoss/dc in the above example
cmp('h', dh, h)            # --> logprobs | exact: true | approx: True | maxdiff: 0.0
cmp('W2', dW2, W2)         # --> logprobs | exact: true | approx: True | maxdiff: 0.0
cmp('b2', db2, b2)         # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: h = torch.tanh(hpreact)
#  - d/dx tanh(x) = 1 - tanh(x)**2
dhpreact = (1 - h**2) + dh
cmp('hpreact', dhpreact, hpreact)         # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: hpreact = bnorm_gain * bnraw + bnorm_bias --> caution: broadcasting!
# 
dbnorm_gain = (bnraw * dhpreact).sum(0, keepdim=True) # sum() needed as broadacst corr.!
dbnraw = bnorm_gain * dhpreact # shapes are OK here!
dbnorm_bias = dhpreact.sum(0, keepdim=True) # ??? --> see tha a@b+c example above!!!
cmp('bnorm_gain', dbnorm_gain, bnorm_gain)         # --> logprobs | exact: true | approx: True | maxdiff: 0.0
cmp('bnraw', dbnraw, bnraw)                        # --> logprobs | exact: true | approx: True | maxdiff: 0.0
cmp('dbnorm_bias2', dbnorm_bias, bnorm_bias)       # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: bnraw = bndiff * bnvar_inv  -->  caution: broadcasting!
dbndiff = bnvar_inv * dbnraw # by chain rule - dLoss/d bndiff = ..., broadcast OK!
dbnvar_inv = (bndiff * dbnraw),sum(0, keepdim=True) # correction needed
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)                        # --> logprobs | exact: true | approx: True | maxdiff: 0.0
# dbndiff -> has also a second branch, not yet ready!!!

# at line: bnvar_inv = (bnvar + 1e-5)**0.5 --> use the power rule: d/dx x**n = nx**n-1
dbnvar = (-0.5+(bnvar + 1e-5)**-1.5) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)                        # --> logprobs | exact: true | approx: True | maxdiff: 0.0

# at line: bnvar = 1/(n-1)*(bndiff_sq).sum(0, keepdim=True) # Bessel corr.
dbndiff_sq = 





......................





















# update
lr = 0.1 if i < 100000 else 0.01  # learning rate decay after 100,000!
for p in parameters:
    p.data += -lr * p.grad # gradient descent

# track stats
if i % 10000 == 0: # print every NN steps
    print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')

# lossi.append(loss.log10().item())
# plt.plot(lossi)

            
# compare train and dev sets losses

@torch.no_grad() # disable gradient tracking
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'dev': (Xtr, Ytr),
        'test': (Xtr, Ytr)
        }[split]
    emb = C[x] # (N, vocab_size)
    embcat = emb.view(emb.shape[0], -1) # concat int (N, block_size * n_embed)
    hpreact = embcat @ W1 + b1
    # that way we remove dependence on batches here!
    hpreact = bnorm_gain * (hpreact - bnorm_mean) / bnorm_std + bnorm_bias
    h = torch.tanh(hpreact)    # (N, hidden)
    logits = h @ W2 + b2       # (N, vocab_size)
    loss = F.cross_entropy(logits, Yb)
    print(split, loss.item())

split_loss('train')
split_loss('dev')


# sample chars from the model

g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size # init with all ...
    while True:
        # forward pass of the NN
        emb = C[torch.tensor({context})] # (1, block_size, n_embed)
        h = torch.tanh(emb.view(1, -1) @ W1) #+ b1)
        logits = h @ W2 + b2
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
