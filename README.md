# TinyAllInLLM

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

### 4. Decoding
Generating text

``` sh
python decode.py --vocab_size=2500 --device=[device] --model_file=model.pth --promp='...' [params ...]
```
