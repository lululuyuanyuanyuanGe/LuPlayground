with open('VLM\input.txt', "r", encoding = 'utf-8') as f:
    text = f.read()

print("Length of dataset in characters: ", len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
