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


# MLP neural net impl.
#  - source: ................

n_emb = 10     # dim. of the character embedding vectors
n_hidden = 200 # neurons in the hidden layer

# randomly init NN's params
g = torch.Generator().manual_seed(2147483647) # mk. reproducible

C  = torch.randn((vocab_size, n_emb),     generator=g)  # embeddings lookup table
W1 = torch.randn((n_emb * block_size, n_hidden), generator=g) # weights hidden L.
b1 = torch.randn(n_hidden,               generator=g)         # biases hidden L.
W2 = torch.randn((n_hidden, vocab_size), generator=g)   # weights output L.
b2 = torch.randn(vocab_size,             generator=g)   # biases output L.

parameters = [C, W1, b1, W2, b2]
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
    hlayer_input = embcat @ W1 + b1
    h = torch.tanh(hlayer_input) # hidden layer
    logits = h @ W2 + b2         # output layer

    #  --> F.cross_entropy(...) same (but more efficient!) as :
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y},log().mean()
    
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


# compare train and dev sets losses

@torch.no_grad() # disable gradient tracking here
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'dev': (Xtr, Ytr),
        'test': (Xtr, Ytr)
        }[split]
    emb = C[x] # (N, vocab_size)
    embcat = emb.view(emb.shape[0], -1) # concat int (N, block_size * n_embed)
    h = torch.tanh(embcat @ W1 + b1)    # (N, hidden)
    logits = h @ W2 + b2         # (N, vocab_size)
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
