import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

# Create a 4D tensor: [batch_size, num_heads, seq_len, d_k]
tensor = torch.randn(2, 2, 2, 2)
print(f"Original shape: {tensor}")  # [2, 8, 10, 64]

# transpose(1, 2) - swaps dimensions 1 and 2
result1 = tensor.transpose(1, 2)
print(f"transpose(1, 2): {result1.shape}")  # [2, 10, 8, 64]
# Swapped num_heads (8) and seq_len (10)

# transpose(-2, -1) - swaps last two dimensions  
result2 = tensor.transpose(-2, -1)
print(f"transpose(-2, -1): {result2.shape}")  # [2, 8, 64, 10]
# Swapped seq_len (10) and d_k (64)

# tensor.view(tensor.shape[0], tensor.shape[1], 1, 3)


word_to_ix = {"hello": 0, "world":1}
print(word_to_ix["world"])
#
lookup_tensor = torch.tensor([0, 1], dtype=torch.long)
print(lookup_tensor.shape)
model = Embedding(2, 10)
embeds = model(lookup_tensor)
print("\n" + "-----------" + "\n")
print(embeds)


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

vocab = set(test_sentence)


word_to_ix_sent = {word: i for i, word in enumerate(vocab)}
print(word_to_ix_sent)