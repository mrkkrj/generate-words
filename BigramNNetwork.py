import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline


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


# create the trainig set of bigrams (x,y)

xs, ys = [], []

for w in words:
    chars = ['.'] + list(w) + ['.'] # chars of word + START/STOP
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        #print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)


# create a NN
#  -- 27 linear neurons wo. bias, each neuron has 27 inputs
#  -- single layer + softmax

g = torch.Generator().manual_seed(2147483647) # mk. reproducible
W = torch.randn((27, 27), generator=g, requires_grad=True) # normal distr.


# train the NN

for k in range(100)

    # forw pass
    xenc = F.one_hot(xs, num_classes=27).float() # one-hot encoding, 27 bc. of num of chars

    logits = xenc @ W # aka. log-counts  ::: (num_inputs, 27) @ (27, 27) -> (num_inputs, 27), i.e. predicted next char in bigram!
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True) # the last 2 lines called "Softmax" !!!!

    # loss: negative mean log probablility of the desired outputs!
    #  --> regularization <=> pushing the distr toward uniform distr!!!
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # + regularization loss!
    print(loss.item())

    # backw pass
    W.grad = None # set to zero
    loss.backward()

    # update
    W.data += -50 * W.grad


# sampling from NN model

for i in ramge(5):
    out = []
    ix = 0

    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # prodict log counts
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True) # probabilities of next character after ix

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        
        if ix == 0:
            break
        
    print(''.join(out))
 
# EOF
