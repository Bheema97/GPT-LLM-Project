# GPT From Scratch — Character-Level Transformer on OpenWebText

This repository documents a hands-on experiment in building and training a small GPT-style language model from scratch using PyTorch.

The goal was **not** to build a production-grade LLM. The project was created to understand the complete training pipeline behind autoregressive language models, including:

- character-level tokenization
- token and positional embeddings
- self-attention
- multi-head attention
- feed-forward layers
- Transformer blocks
- next-token prediction
- cross-entropy loss
- backpropagation
- AdamW optimization
- training / validation loss
- model checkpointing
- continued training
- text generation

The project began with a simple bigram language model and was gradually extended into a small GPT-style Transformer trained on OpenWebText.

---

## Final Model Configuration

The final experiment used:

```python
block_size = 64
batch_size = 32

n_embd = 384
n_head = 8
n_layer = 8

dropout = 0.2
```

Initial training used:

```python
learning_rate = 3e-4
```

Later refinement used:

```python
learning_rate = 1e-4
```

The model is a **character-level autoregressive Transformer**, meaning that it predicts the next character from previous characters.

---

## Dataset

The project uses:

**OpenWebText**

Hugging Face dataset:

```text
Skylion007/openwebtext
```

OpenWebText is a large collection of web documents and is significantly larger than the small text files normally used in introductory GPT tutorials.

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv cuda
```

Activate it on Windows:

```bash
cuda\Scripts\activate
```

Install dependencies:

```bash
pip install torch
pip install datasets
pip install tqdm
pip install ipykernel
```

If CUDA-enabled PyTorch is required, install the appropriate PyTorch build for your GPU from the official PyTorch installation instructions.

---

## Optional Jupyter Kernel Setup

The virtual environment can be registered as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=cuda --display-name="cuda-gpt"
```

Then start browser-based Jupyter with:

```bash
jupyter notebook
```

and select:

```text
cuda-gpt
```

as the notebook kernel.

---

## Downloading OpenWebText

Install the Hugging Face `datasets` package if it is not already installed:

```bash
pip install datasets
```

Then download the dataset:

```python
from datasets import load_dataset

ds = load_dataset(
    "Skylion007/openwebtext",
    cache_dir=r"PATH_TO_YOUR_DATASET_DIRECTORY"
)
```
The python code to download the dataset has already been provided in "download_dataset.py" and can be readily run after imstalling the dataset library.

Hugging Face stores the downloaded and processed dataset inside the selected cache directory.

If the download is interrupted, rerunning the same command with the same cache directory will normally reuse existing cached data.

For a large download like OpenWebText, downloading through a normal Python script or terminal is often more reliable than doing the full transfer through a browser-based Jupyter notebook.

---

## Arrow Dataset Format

Older tutorials may show OpenWebText as compressed `.xz` files.

Recent Hugging Face versions may instead cache the processed dataset as:

```text
.arrow
```

files.

The preprocessing script in this repository therefore reads the Hugging Face Arrow representation rather than using `lzma` on `.xz` files.

---

## Dataset Processing

Run:

```bash
python data_processing.py
```

The preprocessing script:

1. Locates `.arrow` files in the OpenWebText cache directory
2. Opens each Arrow shard
3. Extracts the `text` field from each record
4. Splits the source shards into:
   - 90% training
   - 10% validation
5. Creates:
   - `output_train.txt`
   - `output_val.txt`
6. Builds:
   - `vocab.txt`

The output structure is approximately:

```text
project/
├── data_processing.py
├── output_train.txt
├── output_val.txt
├── vocab.txt
├── train.py
├── chatbot.py
└── README.md
```

The dataset path inside `data_processing.py` must be updated to point to the directory containing the `.arrow` files.

---

## Character-Level Vocabulary

The project uses a character-level tokenizer.

Each unique character is assigned an integer ID:

```python
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
```

Encoding:

```python
encode = lambda s: [string_to_int[c] for c in s]
```

Decoding:

```python
decode = lambda l: ''.join([int_to_string[i] for i in l])
```

The processed OpenWebText vocabulary in this experiment contained approximately:

```text
32,171 unique characters
```

The large vocabulary is caused by the many Unicode characters found in web data.

---

## Training Data Sampling

The processed OpenWebText files are too large to load completely into RAM.

The training script therefore uses Python's:

```python
mmap
```

module.

The process is:

```text
Large text file
      ↓
Select random byte position
      ↓
Read a small chunk
      ↓
Decode text
      ↓
Convert characters to token IDs
      ↓
Construct training batch
```

This allows random sampling from very large files without loading the whole dataset into memory.

---

## Training Objective

The model performs **next-character prediction**.

For example:

```text
Input:
T h e   k i n

Target:
h e   k i n g
```

The target is the same sequence shifted by one character.

The model produces logits for every possible next character.

Cross-entropy loss compares those logits to the correct next-character targets.

The training loop is:

```text
Get random batch
      ↓
Forward pass
      ↓
Calculate logits
      ↓
Cross-entropy loss
      ↓
Backpropagation
      ↓
AdamW parameter update
      ↓
Repeat
```

---

## Model Architecture

### Token Embeddings

```python
nn.Embedding(vocab_size, n_embd)
```

Transforms token IDs into learned vector representations.

### Positional Embeddings

```python
nn.Embedding(block_size, n_embd)
```

Adds information about where each token occurs in the sequence.

### Self-Attention

Each attention head creates:

```text
Key
Query
Value
```

Attention weights are calculated from:

```text
Q × Kᵀ
```

followed by scaling, causal masking and softmax.

### Multi-Head Attention

Several attention heads operate in parallel, and their outputs are concatenated and projected back into the embedding dimension.

### Feed-Forward Network

Each Transformer block contains a feed-forward layer:

```text
n_embd
  ↓
4 × n_embd
  ↓
ReLU
  ↓
n_embd
```

### Transformer Stack

The model pipeline is:

```text
Token IDs
   ↓
Token embeddings
   +
Position embeddings
   ↓
Transformer blocks
   ↓
Final LayerNorm
   ↓
Language-model head
   ↓
Vocabulary logits
```

---

## Context Window

The final model used:

```python
block_size = 64
```

During generation:

```python
index_cond = index[:, -block_size:]
```

ensures the model only sees the most recent 64 characters.

A larger context window was considered, but the experiment was stopped at 64 because the project was intended as a learning exercise on laptop hardware.

---

## Training Experiments

The model was trained incrementally so that the effect of additional training could be observed.

### Stage 1 — Initial Training

The first phase used a learning rate of approximately:

```python
3e-4
```

Training was performed in several runs:

```text
1000 iterations
1000 iterations
1000 iterations
2500 iterations
```

Total:

```text
~5,500 iterations
```

Validation loss decreased from roughly the high `2.x` range to around:

```text
~1.58
```

The model progressed from random character output toward recognizable English-like words and sentence fragments.

---

### Stage 2 — Continued Refinement

The saved model was reloaded and training continued with:

```python
learning_rate = 1e-4
```

Two further runs were performed:

```text
3000 iterations
3000 iterations
```

Total additional training:

```text
~6,000 iterations
```

Validation loss moved toward approximately:

```text
~1.47
```

Generated text became more structured and contained more real English vocabulary, punctuation and sentence-like phrasing.

---

### Stage 3 — Final Fine-Tuning

A final run of approximately:

```text
5000 iterations
```

was performed at:

```python
learning_rate = 1e-4
```

The loss began to plateau in the mid-`1.4` range.

At this point the experiment was intentionally stopped because further training was giving diminishing returns and the goal of the project was educational.

---

## Example Progression

The qualitative progression looked roughly like this:

### Untrained Model

```text
Random characters
Broken symbols
No meaningful words
No coherent sentence structure
```

### Early Training

```text
English-like fragments
Common letter combinations
Some real words
Basic punctuation
```

### Mid Training

```text
There was a king. Game maliking trade...
```

### Later Training

```text
There was a kings of decades and employers...
The next explained analysis...
```

The final model is still imperfect, but it clearly demonstrates the transition from random prediction toward learned language structure.

---

## Saving the Model

The project saves the trained model using `pickle`:

```python
with open("model-01.pkl", "wb") as f:
    pickle.dump(model, f)
```

To reload:

```python
with open("model-01.pkl", "rb") as f:
    model = pickle.load(f)

model = model.to(device)
```

The model can then continue training from its existing parameters.

---

## Running the Training Script

If the training script uses `argparse` for the batch size:

```bash
python train.py -batch_size 32
```

Example terminal output:

```text
batch_size: 32
cuda
```

If CUDA is available, the model trains on the GPU.

Otherwise it falls back to CPU.

---

## Running the Chatbot / Generator

After training:

```bash
python chatbot.py -batch_size 32
```

The script loads:

```text
model-01.pkl
```

and waits for a prompt:

```text
Prompt:
```

Example:

```text
Prompt:
There was a king
```

The model then generates additional character tokens.

---

## Important: Model Configuration Must Match

The chatbot/inference script must use the same architecture as the training script.

These must match:

```python
block_size
n_embd
n_head
n_layer
vocab_size
```

The vocabulary mapping must also remain identical.

For example, if the model was trained with:

```python
block_size = 64
n_embd = 384
n_head = 8
n_layer = 8
```

the chatbot must use those same architecture settings.

Changing `block_size` after training caused an attention-mask dimension mismatch during this experiment.

Training-only parameters such as these can be changed safely:

```python
learning_rate
batch_size
max_iters
eval_iters
```

---

## Recommended Project Structure

```text
fcc-gpt-course/
├── data_processing.py
├── train.py
├── chatbot.py
├── README.md
├── vocab.txt
├── output_train.txt
├── output_val.txt
└── model-01.pkl
```

---

## Recommended `.gitignore`

Large dataset files and checkpoints should usually not be committed directly to GitHub.

```gitignore
# Virtual environments
cuda/
venv/

# Python cache
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Hugging Face / Arrow cache
.cache/
*.arrow

# Processed training data
output_train.txt
output_val.txt

# Model checkpoints
*.pkl
*.pt
*.pth
```

If model checkpoints need to be distributed, Git LFS or an external model hosting service is preferable to normal Git tracking.

---

## Hardware Notes

This experiment was run on a CUDA-capable laptop GPU.

A larger model was considered using settings such as:

```text
block_size = 128
n_layer = 12
n_head = 12
```

but the project was intentionally stopped before that point.

Self-attention scales approximately as:

```text
O(T²)
```

so doubling context length from 64 to 128 significantly increases memory and compute requirements.

The objective was to understand GPT mechanics rather than maximize model scale.

---

## Key Lessons

This project helped demonstrate that:

1. Token IDs are arbitrary labels.
2. Embeddings create learned numerical representations.
3. Positional embeddings provide sequence-order information.
4. Self-attention allows tokens to interact with previous context.
5. Multi-head attention can learn different relationships simultaneously.
6. Logits are raw prediction scores.
7. Cross-entropy measures next-token prediction error.
8. Backpropagation calculates gradients.
9. AdamW updates the model parameters.
10. Validation loss helps measure real improvement.
11. Continued training produces diminishing returns.
12. Model architecture must match between training and inference.
13. Large datasets require chunk-based or memory-mapped processing.
14. A base language model is fundamentally a next-token predictor.
15. Instruction-following behavior requires additional training beyond base language modelling.

---

## Limitations

This model:

- uses character-level tokenization
- has a short context window
- has limited model capacity
- is not instruction-tuned
- is not preference-aligned
- is not RLHF-trained
- can generate incorrect grammar
- can invent words
- can produce incoherent passages
- should not be considered a production chatbot

A prompt such as:

```text
Hello!
```

does not necessarily result in a conversational answer because the model was trained only for next-token prediction.

Modern assistants typically involve additional stages such as:

```text
Large-scale pretraining
       ↓
Instruction tuning
       ↓
Preference alignment / RLHF
       ↓
Assistant behaviour
```

This project focuses mainly on the first stage.

---

## Conclusion

The purpose of this project was to understand what happens inside a GPT-style language model rather than simply use an existing pretrained model.

The full pipeline was:

```text
OpenWebText
    ↓
Arrow preprocessing
    ↓
Character vocabulary
    ↓
Token IDs
    ↓
Random training batches
    ↓
Embeddings
    ↓
Self-attention
    ↓
Transformer blocks
    ↓
Next-character prediction
    ↓
Cross-entropy loss
    ↓
Backpropagation
    ↓
Continued training
    ↓
Generated text
```

The most important result was observing the model gradually move from random character generation toward recognizable English sentence structure as training progressed.

The final model is intentionally small and imperfect, but it serves as a complete practical demonstration of the core mechanics behind GPT-style autoregressive language models.

---

## Disclaimer

This repository is an educational implementation created for learning purposes.

It is not intended to be a production-ready LLM or chatbot.
