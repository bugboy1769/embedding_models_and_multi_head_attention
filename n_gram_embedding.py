import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#indexing
word_to_ix = {"hello": 0, "world":1}
#creating an embedding object
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor(word_to_ix["world"], dtype=torch.long)
print(lookup_tensor.data)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

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
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)


#this creates essentially tuples of the form ([all preceding words in CONTEXT_SIZE], target_word) for the entire sentence
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

print(ngrams[:3])

#remove duplicates
vocab = set(test_sentence)
#assigning indices
#sort of one hot encoded
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        #creates embeddings for each vocab with a dimensionality given by embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #first linear transform that takes in the total features in a given context window (context_size * embedding_dim) and creates an output with 128 representations or feature size
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        #takes the 128 represenations for each output feature from linear1 and condenses it back to num(vocab_size) representations
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        #squash and create a unified vector for 
        embeds = self.embeddings(inputs).view((1, -1))
        #first layer output, applied relU to the output of the first layer
        out = F.relu(self.linear1(embeds))
        #second layer fed the output of the first layer, there does not seem to be any activation function, pure dimensionality compression
        out = self.linear2(out)
        #taking the log of the probability of the output layer
        log_probs = F.log_softmax(out, dim=1)

        return log_probs
    
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:
        #index mapping
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        #instances of torch accumulate gradients, clear them out
        model.zero_grad()
        #forward pass and subsequent log_prob generation
        log_probs = model(context_idxs)
        #computing the loss function, torch needs the word index to be wrapped up in a tensor
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        #backward pass and gradient update
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        losses.append(total_loss)

print(losses)

print(model.embeddings.weight[word_to_ix["beauty"]])








