with open('VLM\input.txt', "r", encoding = 'utf-8') as f:
    text = f.read()

print("Length of dataset in characters: ", len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: [itos[c] for c in s]
print (encode("Hi there"))

import torch
data = torch.tensor(encode(text))
print(data.shape, data.dtype)
print(data[:1000])

# split the data into train and vlidation sets
n = int(0.9* len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]