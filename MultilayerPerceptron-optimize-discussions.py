import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

#
# Contains discussions of several optimizations in initialization of the MLP
#  - lastly, only Batch Norm. is important, but will be cleanly impl. in a different file
#


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

Xtr, Ytr = build_dataset words[:n1]      # 80% for training
Xdev, Ydev = build_dataset words[n1:n2]  # 10% for dev opt.
Xtst, Ytst = build_dataset words[n2:]    # 10% for test


# MLP neural net impl.
#  - source: ................

n_emb = 10     # dim. of the character embedding vectors
n_hidded = 200 # neurons in the hidden layer



# randomly init NN's params
g = torch.Generator().manual_seed(2147483647) # mk. reproducible

C  = torch.randn((vocab_size, n_emb),            generator=g)

'''
## optimize init 1:
## => FIX softmax being confidently wrong!
##  - expected initial (uninformed) loss:
#> exp_loss = -torch.tensor(1/27.0).log() ## --> so we want the logits to be equal AND small !!!

## optimize init 2:
## => FIX tanh layer too saturated on init
##  - plot this after the 1st step
plt.hist(h.view(-1).tolist(), 50);

##  - h : if you draw a histogram, most of values are 0 or 1 !!! :-O
##  - hlayer_input : (histogram) - distr. very broad!
## -> all gradient of h will vanish (0)
## - visualize h:
plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')

## if all gradients for some neuron vanish (a column in this img. is all white), this is a DEAD neuron then!!
## -> so we want to have hlayer_input to have a small range!

## optimize init 3
## => calculating the init scale: "Kaiming init" (afer Kaiming He et al's article)!
##
##   - question: what are good multipliers for W1, b1, etc? ??
##   - if we have random inputs x (normal distr m=0, st.dev=1) and random weights W (also normal) and
##     y = x @ W --> y will have soma normal distr, but differnt from x (std dev. greater!)
##     so how can we scale W so the distr. won't change?
##   - resp: divide by square root of the fan-in (fan-in: num if input elems)
##  Kai-ming article uses ReLU neurons and says; initialize with mean=0 and std.dev=square root of 2/fan-in.
##  Plus: also good for the backward pass!
##   --> implemented in Torch: init.kaiming_normal. 

## Kai-ming: gain/sq.root of fan-in, for tanh() gain is 5/3, so:'''

#W1 = torch.randn((n_emb * block_size, n_hidden), generator=g) * 0.2  # small h preactivations!!
# --> Kai-Ming:
W1 = torch.randn((n_emb * block_size, n_hidden), generator=g) (5/3)/((n_emb * block_size)**0.5) #* 0.2
#b1 = torch.randn(n_hidden,                       generator=g) * 0.01 # same!
# BUT: Batch Norm. removes bias by (substr. of the mean), so no need to use it (wasteful)! It has its own bias instead
W2 = torch.randn((n_hidden, vocab_size),         generator=g) * 0.01 # small & roughly equal logits!!!
b2 = torch.randn(vocab_size,                     generator=g) * 0  

## Modern innovatios make this less an issue (Adam, etc...) -> Andrej just normalizes with sq.root of fan-in!
## Batch Normalization, S. Ioffe, X. Xxx (2015) -> we want hidden laeyer activations roughly gaussian (not too
## small/big bc. tanh() is saturated then!)
##  -> so: why won't we just normalize it to be gaussian? Insight - that's completely OK!
## note: here the NN is very simple, so BN not bringing much. BUT: for very deep networks it nicely replaces the init. maths!
## note: BN introduces dependence on batch elements - noise! But this added entropy is good. 

# we want pre-activation to be gaussian at init only, need these corrections (see paper)
b_norm_gain = torch.ones((1, n_hidden))     
b_norm_bias = torch.zeroes((0, n_hidden))
b_norm_mean_running = torch.zeroes((0, n_hidden))
b_norm_std_running = torch.ones((0, n_hidden))

parameters = [C, W1, b1, W2, b2, b_norm_gain, b_norm_bias]
print("num param=", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# train the NN

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

    hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation, note: B.N. doesn't need bias
    # normalize to be gaussian (Batch Norm. paper!)
    #hpreact = b_norm_gain * (hpreact - hpreact.mean(0, keepdim=True) / hpreact.std(0, keepdim=True) + b_norm_bias

    b_norm_mean_i = hpreact.mean(0, keepdim=True)
    b_norm_std_i = hpreact.std(0, keepdim=True)    
    hpreact = b_norm_gain * (hpreact - b_norm_mean_i) / (b_norm_std_i  + epsilon) + b_norm_bias
    
    # note: use epsilon to fix things should b_norm_std_i be zero! (see paper)
    #hpreact = b_norm_gain * (hpreact - b_norm_mean_i) / (b_norm_std_i  + epsilon) + b_norm_bias

    with torch.no_grad():
        # estimate mean/std for the training set
        b_norm_mean_running = 0.999 * b_norm_mean_running + 0.001 * b_norm_mean_i
        b_norm_std_running = 0.999 * b_norm_std_running + 0.001 * b_norm_std_i
      
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2    # output layer   
    loss = F.cross_entropy(logits, Yb) # <=> log softmax + negative log likelihood !!!   

    # backward pass
    for p in parameters:
        p.grad = None
        
    loss.backward() # auto diff.!!!

    # update
    lr = 0.1 if i < 100000 else 0.01  # learning rate decay after 100,000!
    for p in parameters:
        p.data += -lr * p.grad # gradient descent

    # track stats
    if i % 10000 == 0: # print every NN steps
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')

    lossi.append(loss.log10().item())
    plt.plot(lossi)


# calibrate the batch norm at the end of training
#  -> BN paper: to be able to forward a single example!

'''
## b_norm_mean/b_norm_std can be replaced by b_norm_mean_running/b_norm_std_running
##  -> so we do not need that pass!

with torch.no_grad():
    # pass the training set through
    emb = C{Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    # get mean/std
    b_norm_mean = hpreact.mean(0, keepdim=True)
    b_norm_std = hpreact.std(0, keepdim=True)
'''
b_norm_mean =  b_norm_mean_running                            
b_norm_std =  b_norm_std_running                            


# compare train and dev sets losses

@torch.no_grad() # disable gradient tracking
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'dev': (Xtr, Ytr),
        'test': (Xtr, Ytr)
        }[split}
    emb = C[x] # (N, vocab_size)
    embcat = emb.view(emb.shape[0], -1) # concat int (N, block_size * n_embed)
    hpreact = embcat @ W1 + b1
    #hpreact = b_norm_gain * (hpreact - hpreact.mean(0, keepdim=True) / hpreact.std(0, keepdim=True) + b_norm_bias
    # that way we remove dependence on batches here!
    hpreact = b_norm_gain * (hpreact - b_norm_mean / b_norm_std + b_norm_bias
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
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
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

    



