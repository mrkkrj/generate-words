
# we are using tinyshakesperate as input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('lenth of dataset:', len(text))

# get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size) # =65

# create a mapping from chars to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder
decode = lambda l: ''.join([itos[i] for i in l]) # decoder

print(encode("hii there"))
print(decode(encode("hii there")))

# use sub-word token encoding in practice!
#  - tik-token, SentencePiece (Google), ...

import torch # use PyTorch !
data = torch.tensor(encode(text), dtype=torch.long)

# split te data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# 
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

# print 8 (block_size) examples (len=1..8) hidden in 9 (block_size+1) characters form data
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the tartget: {target}")

#
torch.manual_seed(1337)

batch_size = 4 # how many independent sequences will be processed in parallel?
block_size = 8 # what is the maximum context length fo the predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,)) # batch_size array of random offsets (between 0 and len(data) - block_size)
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack stacks the arrays as rows of a 4x8 tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

# print 32 independent examples in 4 batches
for b in range(batch_size): # batch dim.
    for t in range(block_size): # time dim.
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context} the tartget: {target}")


# bigram network model:

import torch 
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 
        
        # ...An Embedding layer is essentially just a Linear layer. So you could define a your layer 
        # as nn.Linear(1000, 30), and represent each word as a one-hot vector, e.g., [0,0,1,0,...,0] 
        # (the length of the vector is 1,000).
        #
        # As you can see, any word is a unique vector of size 1,000 with a 1 in a unique position, 
        # compared to all other words. Now giving such a vector v with v[2]=1 (cf. example vector 
        # above) to the Linear layer gives you simply the 2nd row of that layer. 
        # 
        # nn.Embedding just simplifies this. Instead of giving it a big one-hot vector, you just 
        # give it an index. This index basically is the same as the position of the single 1 in the 
        # one-hot vector.
        #
        #   FOR --> 1000 words, 30 dim vectors embeddding for each


    def forward(self, idx, targets): # idx and targets are both (B, T) tensor of integers
        
        # for each elem if idx get the indexed row from embedding tbl.
        logits = self.token_embedding_table[idx] # (B,T,C) --> (Batch, Time, Channel(=vocab_size)) = (4, 8, 65)
        
        # NOW: interpret it as scores (logits) for the next char in the sequence
        #  - predicting based on a just single token!!!        

        # -- loss

        # won't run, PyTorch expects (B,C,T) not (B,T,C)!
        #loss = F.cross_entropy(logits, targets) # quality of logits wrt. the targets
        B, T, C = logits.shape
        logits = logits.view(B*T, C) # bc. C must be second!
        targets = target.view(B*T)
        loss = F.cross_entropy(logits, targets) # quality of logits wrt. the targets

        return logits, loss
    
    def generate(self, idx, max_new_tokens):        
        # idx is (B, T) array of indices in the currentz context
        #  -- i.e. a current context of some chars in a batch
        #  -- generate() wants extend by 1 in the Time dimention (i.e. predict next!)
        for _ in range(max_new_tokens):
            # get predictions
            logits, looss = self(idx)
            # focus only on ..
            logits = logits[:, -1, :] # becomes (B, C)
            # ...
            probs = F.softmax(logits, dim=1)
            # sample
            idx_next = torch.multinomial(p, num_samples=1)
            # append ...
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

        


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape) # torch.Size([4, 8, 65]) - we get the sores for every one of the (4,8) positions
print(loss) # we expect -ln(1/65) !!! -> see prev. installments!

