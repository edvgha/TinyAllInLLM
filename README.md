# TinyAllInLLM

A lightweight language model built entirely from scratch, designed to be simple, understandable, and easy to train on a standard laptop.

## 🎯 About the project
**Tiny-All-In-LLM** is a minimalist, yet complete, implementation of the 
transformer architecture. It was created with three primary goals in mind:
* **Education:** To demystify the core mechanics of the transformer architecture by providing a clear and heavily commented codebase.
* **Accessibility:** To create a model that can be trained and run without requiring expensive, high-end hardware.
* **Simplicity:** To focus on a "from-scratch" approach with minimal dependencies, allowing learners to understand every component.


### 0. Env setup
Create environment

``` sh
conda env create -f environment.yml
```

### 1. Download data
Download the TinyStories data

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### 2. Train BPE
Train BPE on the TinyStories data

``` sh
python train_bpe_tokenizer.py --vocab_size=[vocab_size:int] [params ...]
```

### 3. Encoding
Tokenizing text

``` sh
python run_tokenizer.py [params ...]
```

### 4. Train TransformerLM
Train the model

``` sh
python train.py --vocab_size=[vocab_size:int] [params ...]
```

### 5. Decoding
Generating text

``` sh
python decode.py --vocab_size=[vocab_size:int] --device=[device:str] --model_file=[path:str] --prompt='...' [params ...]
```
