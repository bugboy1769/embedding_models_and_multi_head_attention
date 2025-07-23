import torch
import torch.nn as nn
import torch.nn.functional as F


# CONTEXT_SIZE = 5
# EMBEDDING_DIM = 10
# test_sentence = """When forty winters shall besiege thy brow,
# And dig deep trenches in thy beauty's field,
# Thy youth's proud livery so gazed on now,
# Will be a totter'd weed of small worth held:
# Then being asked, where all thy beauty lies,
# Where all the treasure of thy lusty days;
# To say, within thine own deep sunken eyes,
# Were an all-eating shame, and thriftless praise.
# How much more praise deserv'd thy beauty's use,
# If thou couldst answer 'This fair child of mine
# Shall sum my count, and make my old excuse,'
# Proving his beauty by succession thine!
# This were to be new made when thou art old,
# And see thy blood warm when thou feel'st it."""

# CONTEXT_SIZE_2 = 2  # 2 words to the left, 2 to the right
# raw_text_2 = """We are about to study the idea of a computational process.
# Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data.
# The evolution of a process is directed by a pattern of rules
# called a program. People create programs to direct processes. In effect,
# we conjure the spirits of the computer with our spells."""

# # we should tokenize the input, but we will ignore that for now
# # build a list of tuples.
# # Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)

# test = [([test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
#         test_sentence[i])
#         for i in range(CONTEXT_SIZE, len(test_sentence))
#         ]

# print(test[:3])


# #Code below is a great illustration of how embeddings are stored. One big personal insight is that embedding vectors never really store the actual words in the forms of embddings, they only store the mappings, all processing happens against these mappings, this is great because once the training is done for the embeddings, the mappings now just point to the same stuff but these vectors are now trained.

# sentence_dict = {
#     "indices": 0,
#     "are": 1,
#     "used": 2,
#     "for": 3,
#     "lookup": 4
# }

# words_to_embed = [word for word in set(sentence_dict.keys())]
# indices = [sentence_dict[word] for word in words_to_embed]
# print(indices)
# embeds = nn.Embedding(5, 4)
# print(embeds.weight)
# var = torch.LongTensor(indices)
# var_embeds = embeds(var)
# print(var_embeds)

# inputs = torch.LongTensor([1, 2, 4])  # 2 context word indices, these are indices!!!!
# embed_new = nn.Embedding(5, 10)
# print(f"\nSquashed Embeds: {embed_new(inputs).view(1, -1)}") #this is squashing the entire context window and creating a unified vector, which preserves semantic connection in some way


# m = nn.LogSoftmax(dim = 1)
# input = torch.randn(2, 3)
# print(f"Input: {input}")
# output = m(input)
# print(f"Softmaxing: {output}")


# data = [[1, 2],[3, 4]]
# x_data = torch.rand((2, 3))
# print(x_data)

# new_data = torch.randn(3, 3)
# print(new_data)

# tensor_store = torch.zeros([0, 2], dtype=torch.int32)
# from num_vocab_tuple_tensor_creator import vocab_tuple_tensor_store
# print(vocab_tuple_tensor_store(tensor_store, CONTEXT_SIZE, test_sentence))


# Create a 4D tensor: [batch_size, num_heads, seq_len, d_k]
tensor = torch.randn(2, 8, 10, 64)
print(f"Original shape: {tensor.shape}")  # [2, 8, 10, 64]

# transpose(1, 2) - swaps dimensions 1 and 2
result1 = tensor.transpose(1, 2)
print(f"transpose(1, 2): {result1.shape}")  # [2, 10, 8, 64]
# Swapped num_heads (8) and seq_len (10)

# transpose(-2, -1) - swaps last two dimensions  
result2 = tensor.transpose(-2, -1)
print(f"transpose(-2, -1): {result2.shape}")  # [2, 8, 64, 10]
# Swapped seq_len (10) and d_k (64)