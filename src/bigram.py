import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(0xc0ffee)

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# read text
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)
print(''.join(chars))
print(vocab_size) # 65 vocabs
stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode('hi, chris'))
print(decode(encode('hi, chris')))

data = torch.tensor(encode(text))

# train & split data
n = int(0.9 * len(data)) # split data into training set and validation set
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))

batch_size = 4
block_size = 8

def get_batch(split):
  # seperate train & val using split param
  data = train_data if split == 'train' else val_data
  # generate 4 random indices
  ix = torch.randint(len(data) - block_size, (batch_size,))
  # print(ix.shape, ix)
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

xb,yb = get_batch('train')

print(xb.shape, xb)
print(yb.shape, yb)

class BigramLanguageModule(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # 创建 embedding table vector x vector


  def forward(self, idx, targets=None):
    # logits: 模型的原始层输出
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C) # expand
      targets = targets.view(B * T) # expand
      loss = F.cross_entropy(logits, targets) # evaluate loss
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :] # (B, C) block out T
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

m = BigramLanguageModule(vocab_size)
logits, loss = m(xb, yb)
print("logits shape:", logits.shape)
print("loss: ", loss)