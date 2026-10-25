import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# BLOCK 1: DATA PREPARATION
# ==========================================
# We will use a character-level tokenizer to keep it simple.
# It turns letters into numbers.

text = """
Q: What is our company policy?
A: Employees must work 40 hours a week.
Q: How do I request time off?
A: Submit a form to HR two weeks in advance.
"""

# Find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a dictionary to map characters to numbers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Function to turn text into numbers
def encode(s):
    return [stoi[c] for c in s]

# Function to turn numbers back into text
def decode(l):
    return ''.join([itos[i] for i in l])

# Convert our text to a tensor (a multi-dimensional array of numbers)
data = torch.tensor(encode(text), dtype=torch.long)

# ==========================================
# BLOCK 2: THE EMBEDDING LAYER (Pink box in diagram)
# nn.Embedding is just a trainable lookup table.
# ==========================================
class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        """
        vocab_size: Number of unique characters in the text
        n_embd: Number of dimensions in the dense vecto

        Word "I" = Number 0

        Word "love" = Number 1

        Word "AI" = Number 2

        Our vocab_size = 3.

        Let's say our block_size (maximum sentence length) is 10.

        It creates a grid with vocab_size rows and n_embd columns. Inside this grid are random numbers. The model will learn to adjust 
        these numbers during training so that words with similar meanings have similar numbers.
        Row 0 ("I"): [0.1, 0.5, 0.9, 0.2]

        Row 1 ("love"): [0.8, 0.1, 0.3, 0.4]

        Row 2 ("AI"): [0.2, 0.9, 0.1, 0.7]
        """
        # This turns the integer numbers into dense vectors of numbers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        """
         This is another lookup table, but instead of looking up words, it looks up positions (0, 1, 2, 3...).
          It creates a grid with block_size rows (10) and n_embd columns (4).
          Example output: nn.Embedding(10, 4) creates a table like:

            Position 0 (first word): [0.01, 0.02, 0.03, 0.04]

            Position 1 (second word): [0.05, 0.06, 0.07, 0.08]

            Position 2 (third word): [0.09, 0.10, 0.11, 0.12]
        """
        
        # This adds positional encoding (the circle with + in your diagram)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        # idx is the input text (converted to numbers)
        # We unpack the shape of our input => idx is [[0, 1, 2]]. So, B = 1 (one sentence) and T = 3 (three words long).
        B, T = idx.shape
        
        # Output: Looks up the vector for each word and each position, then adds them.
        """
        The computer goes to the table and grabs the 4 numbers for "I", the 4 numbers for "love", and the 4 numbers for "AI".
        [
  [ [0.1, 0.5, 0.9, 0.2],  <-- Numbers for "I"
    [0.8, 0.1, 0.3, 0.4],  <-- Numbers for "love"
    [0.2, 0.9, 0.1, 0.7] ] <-- Numbers for "AI"
]
        """
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        """
        The computer grabs the numbers for position 0, position 1, and position 2.
        [
  [0.01, 0.02, 0.03, 0.04],  <-- Numbers for Position 0
  [0.05, 0.06, 0.07, 0.08],  <-- Numbers for Position 1
  [0.09, 0.10, 0.11, 0.12]   <-- Numbers for Position 2
]

        """
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        
        """
         This is the + circle in your diagram! We simply add the two matrices together.
Word 1 ("I"): [0.1, 0.5, 0.9, 0.2] + [0.01, 0.02, 0.03, 0.04] = [0.11, 0.52, 0.93, 0.24]
Word 2 ("love"): [0.8, 0.1, 0.3, 0.4] + [0.05, 0.06, 0.07, 0.08] = [0.85, 0.16, 0.37, 0.48]
Word 3 ("AI"): [0.2, 0.9, 0.1, 0.7] + [0.09, 0.10, 0.11, 0.12] = [0.29, 1.00, 0.21, 0.82]

Why do we ADD them instead of Combining them?
When you add two vectors (lists of numbers), the resulting vector contains information about both. It's like adding a red light 
to a green light creates a yellow light. The model later learns how to separate the "meaning" from the "position" inside the
Attention blocks.
        """
        
        x = tok_emb + pos_emb # (B,T,C)
        return x

# ==========================================
# BLOCK 3: MASKED MULTI-HEAD ATTENTION (Orange box)
# ==========================================
class Head(nn.Module):
    """A single head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # This creates the "Mask" (the triangle matrix)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        
        # Get Key, Query, and Value
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        
        # Calculate Attention Scores (How much should words look at each other?)
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B,T,T)
        
        # THE MASK: We hide future words so the model can't "cheat" and look ahead.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Apply Softmax to turn scores into probabilities (0 to 1)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        
        # Multiply probabilities by Value
        out = wei @ v # (B,T,C)
        return out

"""
The orange "Multi-Head Attention" box.

Imagine a group of experts reading the sentence. One expert looks for grammar, another for meaning, another for relationships. These are the "Heads". We run them in parallel and combine their notes at the end.

The Code Explained:

self.heads: We create multiple "Heads". If n_embd=4 and num_heads=2, each head gets a head_size of 2.
torch.cat(...): This glues the outputs of all the heads together. If Head 1 outputs 2 numbers, and Head 2 outputs 2 numbers, gluing them gives us 4 numbers.
self.proj: A linear layer that mixes the glued outputs together to create a unified final output.

Example:

Head 1 (looks for subjects): Focuses heavily on "I".
Head 2 (looks for actions): Focuses heavily on "love".
torch.cat combines their findings, and self.proj mixes them so the next layer understands the whole sentence context.
"""

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Run all heads at the same time, then combine them.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# ==========================================
# BLOCK 4: FEED FORWARD (Blue box)
# ==========================================
"""
After the Attention block (orange) has allowed words to talk to each other, the Feed Forward block allows each word to think about what it just learned independently.

"""

class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    
    Expand (nn.Linear): It takes the 4-number vector and expands it to 16 numbers. This gives the model more "space" to find complex patterns.
    ReLU: This is a filter that turns negative numbers into zero. It allows the model to learn non-linear relationships (like "if it's a verb, do X; if it's a noun, do Y").
    Shrink (nn.Linear): It squashes the 16 numbers back down to 4 so it fits back into the main pipeline.
    
    Input for "love": [0.8, 0.1, 0.3, 0.4]

    Expanded (16 numbers): [0.5, -0.2, 1.2, ...]

    After ReLU (negatives become 0): [0.5, 0.0, 1.2, ...]

    Shrunk back to 4: [0.9, 0.2, 0.4, 0.5]
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expand
            nn.ReLU(),                    # Non-linear activation
            nn.Linear(4 * n_embd, n_embd), # Shrink back
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# BLOCK 5: THE TRANSFORMER BLOCK (The big blue box)
# ==========================================
"""
Ex: Let’s pretend our vector size (n_embd) is 4, and we have a sentence of 3 words: ["I", "love", "AI"]


This class wires the pieces together. It adds the Yellow Boxes (Add & Norm) and the bypass arrows (Residual Connections).

Summary
MultiHeadAttention (Orange): Words talk to each other to figure out context.

FeedForward (Blue): Each word thinks individually about what it learned.

Block (The whole package): Does Attention -> Normalize -> Add -> Feed Forward -> Normalize -> Add. Then the next Block does it again!
"""

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        
        """
        self.ln1(x) and self.ln2(x): These are the LayerNorm (Yellow boxes). They normalize the numbers (like turning [100, -50, 20] into [1.5, -0.5, 0.2]) so the model doesn't blow up.
        
        """
               
        # THE YELLOW BOXES (Add & Norm)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        x = x + self.sa(self.ln1(x)):

First, it Normalizes x.
Then it runs it through Attention (self.sa).
The x + is crucial. Look at your diagram. See the arrow that goes around the orange box and adds back to the main line? That's the x +. We take the original input, add the attention's output to it, and pass it on. This helps the model remember the original text and prevents it from "forgetting" during training
        
        """
        
        # 1. Attention + Add & Norm (Note the x + ... is the "Add" part)
        x = x + self.sa(self.ln1(x))
        
        """
        
        x = x + self.ffwd(self.ln2(x)):

It does the exact same thing, but using the Feed Forward block instead.

Example of the x + (Residual Connection):

Original input x: [1.0, 2.0, 3.0]

Attention output: [0.1, -0.1, 0.2]

If we only used the attention output, we lose the original information.

Because of x +, the new x becomes [1.1, 1.9, 3.2]. We keep the original meaning but update it with new context.
        """
        
        # 2. Feed Forward + Add & Norm
        x = x + self.ffwd(self.ln2(x))
        return x

# ==========================================
# BLOCK 6: THE FULL GPT MODEL
# ==========================================
"""
self.blocks = The brain doing the heavy lifting (reading context and finding patterns) repeatedly.

self.ln_f = Cleaning up the messy numbers.

self.lm_head = The mouth that speaks, choosing the most likely next word from our dictionary.
"""
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Pink Box (Embeddings)
        self.embedding = GPTEmbedding(vocab_size, n_embd)
        
        # The Nx Blocks
        """
        we turned words into vectors in the previous step? Those vectors aren't very smart yet. We need to pass them through a series of "processing stations.
        
        Block(...) is the single Transformer block (Masked Attention + Feed Forward + Add & Norm).
        for _ in range(n_layer) creates a list of these blocks. If n_layer = 4, it makes 4 of them.
        nn.Sequential(...) is a PyTorch container. It chains these blocks together so data flows through Block 1, then Block 2, then Block 3, etc.

        Example:

        Input: The vector for "AI" enters Block 1. Block 1 looks at the context and updates the vector.
        That updated vector enters Block 2, gets refined again.
        If n_layer = 4, it goes through 4 rounds of refinement. This is how the model learns complex patterns. (GPT-3 has 96 of these layers!)
        """
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # Final Layer Norm
        """
        Where it is in the diagram: It's the very last "Add & Norm" (Yellow box) before the final Linear layer.
        After passing through all the Transformer blocks, the numbers in our vectors might become very large, very small, or unstable. LayerNorm acts like a volume normalizer.
        It takes the final vectors and scales them so they have a mean of 0 and a standard deviation of 1.
        
        If your vector after the blocks is [100.5, -50.2, 300.1], LayerNorm squashes it down to something neat like [0.5, -1.2, 1.8]. This makes it much easier for the final layer to make a prediction without the numbers blowing up.
        """        
        self.ln_f = nn.LayerNorm(n_embd)
        
        # The Top "Linear" Box (Projects back to vocabulary size)
        """
        The "Linear" box at the very top (just below Softmax).
        This is the final translator. The model has been thinking in "hidden dimensions" (vectors of size n_embd, like 64). But we need to predict the next word.
        
        It takes the vector of size n_embd (e.g., 64 numbers).
        It multiplies it by a matrix to shrink or expand it to vocab_size.
        Output: It gives us one raw score (called a logit) for every single word in our dictionary.
        
        Example: Let's say our dictionary is ["I", "love", "AI"] (vocab size = 3).

            The model just finished processing the word "I love". We want it to predict "AI"
            lm_head takes the vector for "love" and outputs 3 numbers: [0.1, 2.5, 5.0].
            These numbers correspond to: ["I", "love", "AI"].
            Because 5.0 is the highest number, the model strongly predicts the next word is "AI".
            
        Note: In the generate function of the code, we apply F.softmax to these raw numbers [0.1, 2.5, 5.0] to turn them into percentages like [2%, 8%, 90%]. This is the very top "Softmax" box in your diagram!
        """
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # Get embeddings
        x = self.embedding(idx)
        
        # Pass through the Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get final predictions (Logits)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        # If we have targets, calculate loss (how wrong the model is)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

# ==========================================
# SETTINGS & TRAINING
# ==========================================
"""
block_size = 32: This is the model's "short-term memory." It can only look at 32 characters at a time to predict the next one. If you give it a 100-character paragraph, it will only read the last 32 characters.

n_embd = 64: Remember the GPTEmbedding block? This means every single character will be converted into a list of 64 numbers. (GPT-3 uses 12,288 numbers!).
n_head = 4: The "Multi-Head Attention" will have 4 different "experts" (heads) looking at the text simultaneously.
n_layer = 4: The big blue Nx block will be repeated 4 times. The data will go through 4 layers of processing.

optimizer = ...: This is the teacher. The model will make mistakes. The optimizer's job is to look at those mistakes and adjust the millions of numbers (weights) inside the model so it does better next time. lr=1e-3 (learning rate) means it takes small, careful steps when adjusting.
"""

block_size = 32 # How many characters the model looks at at once
n_embd = 64     # How big the internal vectors are
n_head = 4      # Number of attention heads
n_layer = 4     # Number of Transformer blocks (the Nx in diagram)

model = MiniGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

"""
We are going to show the model 2,000 flashcards.
"""

print("Training started...")
for steps in range(2000):
    # Get a random chunk of data
    """
    How do we make a flashcard for an AI?
    We take a random chunk of our text and split it into a Question (x) and an Answer (y).
    
    Example: Let's say our text is "I love AI" and block_size is 3.

x (The Question) = "I l" (index 0, 1, 2)

y (The Answer) = " lo" (index 1, 2, 3)

When the model sees "I", it must predict " " (space).

When the model sees "I ", it must predict "l".

When the model sees "I l", it must predict "o".
    """
    
    ix = torch.randint(len(data) - block_size, (1,))
    x = data[ix:ix+block_size].unsqueeze(0)
    y = data[ix+1:ix+block_size+1].unsqueeze(0)
    
    """
    logits: The model's guesses. It looks at x and outputs raw scores for what it thinks the next character is.

Example output: [0.1 (for 'A'), 5.0 (for 'B'), 0.2 (for 'C')].

loss: The score of how wrong it was. The model compares its guess (logits) to the actual answer (y). If it guessed right, loss is low. If it guessed wrong, loss is high. (This is calculated inside the MiniGPT class using F.cross_entropy).

    """
    
    # Forward pass
    logits, loss = model(x, y)
    
    """
    logits: The model's guesses. It looks at x and outputs raw scores for what it thinks the next character is.

Example output: [0.1 (for 'A'), 5.0 (for 'B'), 0.2 (for 'C')].

loss: The score of how wrong it was. The model compares its guess (logits) to the actual answer (y). If it guessed right, loss is low. If it guessed wrong, loss is high. (This is calculated inside the MiniGPT class using F.cross_entropy).
    """

    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if steps % 500 == 0:
        print(f"Step {steps}, Loss: {loss.item():.4f}")

# ==========================================
# GENERATION (How it answers)
# ==========================================
"""
We want 50 new characters. So, we will run this loop 50 times. Every time the loop runs, the model generates exactly one new character and sticks it to the end of the text.

"""

def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # Crop context to block_size
        # The model can only look at 32 characters at a time
        idx_cond = idx[:, -block_size:]
        
        # We pass those 32 characters (or fewer) through the entire Transformer (Embeddings -> Attention -> Feed Forward -> Linear).
        # Get predictions
        logits, loss = model(idx_cond)
        
        # Output: logits. For every character in our sequence, the model gives us a list of raw scores for what it thinks the next character is.
        # Focus on the last character
        logits = logits[:, -1, :]
        
        # Apply Softmax (Top box in diagram)
        # The logits are raw, wild numbers like [2.5, -1.2, 8.0, 0.5]. Softmax turns these into percentages (probabilities) that add up to 100%.
        """
        Example Output: If our vocabulary is ['A', 'B', 'C', 'W'], softmax turns the raw scores into:

A: 5%

B: 1%

C: 14%

W: 80%
(The model is 80% sure the next letter is 'W' because it learned that "Q: " is usually followed by "What...")
        """        
        probs = F.softmax(logits, dim=-1)
        
        """
        Why don't we just pick the highest probability (W)?
Because if we always picked the highest, the model would be boring and repeat itself. torch.multinomial acts like rolling a loaded dice. It has an 80% chance of picking 'W', a 14% chance of picking 'C', etc. This gives the AI creativity!

Result: It picks 'W'.
        """
        
        # Sample from the probabilities
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

print("\n--- Model Output ---")
# Let's ask it a question
prompt = "Q: What is our"

# encode(prompt): Converts "Q: What is our" into a list of numbers (e.g., [12, 14, 56, 78, ...]).
context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)

# generate(context, max_new_tokens=50): We tell the model: "Here is the start. Now, write 50 more characters to finish the sentence."
output = generate(context, max_new_tokens=50)

# decode(...): The model will output numbers. We translate them back into English letters so we can read it.
print(decode(output[0].tolist()))