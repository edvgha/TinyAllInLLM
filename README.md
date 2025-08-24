# TinyAllInLLM

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
python train_bpe_tokenizer.py --vocab_size=[vocab_size:int]
```
