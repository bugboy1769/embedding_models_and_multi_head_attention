import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world":1}
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor(word_to_ix["world"], dtype=torch.long)
print(f"Look up Tensor: {lookup_tensor.data}")
current_embeds = embeds(lookup_tensor)
print(f"Current Embeds: {current_embeds}")


