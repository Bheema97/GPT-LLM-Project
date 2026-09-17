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
```
<img width="393" height="242" alt="Screenshot 2026-09-17 235228" src="https://github.com/user-attachments/assets/6c4644f7-0550-42d3-b6aa-2cc9958ab614" />

```text
1000 iterations
```
<img width="410" height="245" alt="Screenshot 2026-09-18 000034" src="https://github.com/user-attachments/assets/824ca817-bbbb-4863-8292-e6f864559e00" />

```text
1000 iterations
```
<img width="410" height="235" alt="Screenshot 2026-09-18 000729" src="https://github.com/user-attachments/assets/1b13031c-f105-406b-a991-bdc07648a22b" />

```text
2500 iterations
```
<img width="447" height="562" alt="Screenshot 2026-09-18 001928" src="https://github.com/user-attachments/assets/91f65ed1-0bb2-4cea-8182-8ff358126ba0" />

Total:

```text
~5,500 iterations
```

Validation loss decreased from roughly the high `2.x` range to around:

```text
~1.58
```

The model progressed from random character output toward recognizable English-like words and sentence fragments.
<img width="1463" height="572" alt="Screenshot 2026-09-18 004451" src="https://github.com/user-attachments/assets/e0366434-7649-48cd-a49f-dcae9ae43c91" />

---

### Stage 2 — Continued Refinement

The saved model was reloaded and training continued with:

```python
learning_rate = 1e-4
```

Two further runs were performed:

```text
3000 iterations
```

<img width="416" height="663" alt="Screenshot 2026-09-18 005606" src="https://github.com/user-attachments/assets/b4a44517-5487-45e3-bb49-6abfeaea7c32" />


```text
3000 iterations
```
<img width="425" height="662" alt="Screenshot 2026-09-18 011058" src="https://github.com/user-attachments/assets/fc8320c8-513b-462d-aaaf-511a9d4eca6d" />


Total additional training:

```text
~6,000 iterations
```

Validation loss moved toward approximately:

```text
~1.47
```

Generated text became more structured and contained more real English vocabulary, punctuation and sentence-like phrasing.

<img width="1310" height="600" alt="Screenshot 2026-09-18 011308" src="https://github.com/user-attachments/assets/d00588b3-a06b-4e26-82e9-0a1956f369e3" />


---

### Stage 3 — Final Fine-Tuning

A final run of approximately:

```text
5000 iterations
```
<img width="607" height="746" alt="Screenshot 2026-09-18 013942" src="https://github.com/user-attachments/assets/f456d610-87a5-4ea7-aad4-5ec6ccf874c1" />
<img width="542" height="402" alt="Screenshot 2026-09-18 013948" src="https://github.com/user-attachments/assets/cf78d666-f015-4b19-9f23-bf8994c5306d" />


was performed at:

```python
learning_rate = 1e-4
```

The loss began to plateau in the mid-`1.4` range.

At this point the experiment was intentionally stopped because further training was giving diminishing returns and the goal of the project was educational.

<img width="1450" height="572" alt="Screenshot 2026-09-18 014050" src="https://github.com/user-attachments/assets/4b8113e2-361e-4f80-a44f-9e1c9a0d4f6f" />

---

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

