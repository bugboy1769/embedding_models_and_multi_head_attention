import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def vocab_tuple_tensor_store(vocab_tuple_tensor, context_size, raw_text):
    
    def concat_tensor(vocab_num, tuple_num):
        nonlocal vocab_tuple_tensor
        new_pair = torch.tensor([[vocab_num, tuple_num]], dtype=torch.int32)
        vocab_tuple_tensor = torch.cat([vocab_tuple_tensor, new_pair], dim = 0)
        return vocab_tuple_tensor

    split_raw_text = raw_text.split()

    vocab = set(split_raw_text)
    vocab_num = len(vocab)
    print(f"Number of unique words: {len(vocab)}")
    tuple_num = 0
    for i in range(context_size, len(split_raw_text) - context_size):
        tuple_num = tuple_num + 1

    print (f"Number of tuples: {tuple_num}")

    result = concat_tensor(vocab_num, tuple_num)
    return result
