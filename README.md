# TinyAllInLLM

A lightweight language model built entirely from scratch, designed to be simple, understandable, and easy to train on a standard laptop.

## About the project
**Tiny-All-In-LLM** is a minimalist, yet complete, implementation of the 
transformer architecture. It was created with three primary goals in mind:
* **Education:** To demystify the core mechanics of the transformer architecture by providing a clear and heavily commented codebase.
* **Accessibility:** To create a model that can be trained and run without requiring expensive, high-end hardware.
* **Simplicity:** To focus on a "from-scratch" approach with minimal dependencies, allowing learners to understand every component.
![Alternative Text](https://github.com/edvgha/TinyAllInLLM/blob/main/doc/loss.png)

## Table of Contents

- [Experiment](#experiment)
- [Env setup](#env-setup)
- [Data](#download-data)
- [Train BPE](#train-bpe)
- [Encoding](#encoding)
- [Train TransformerLM](#train-transformer-lm)
- [Decoding](#decoding)

## Experiment
  * Data: TinyStoriesV2-GPT4
  * Machine:
    - Chip: Apple M4 Pro
    - Memory: 24 GB
  * Eval:
    - Train loss: 1.439
    - Val loss: 1.447
    - Train perplexity: 4.22
    - Val perplexity: 4.25
  * Decoding example:
    - Prompt: > *The next day, Lily and her mom went to the part. They saw*
    - Complition: > *The next day, Lily and her mom went to the park. They saw the big tree again. This time, Lily was ready. She saw the shiny, smooth, red fruit. She took a bite and said, "Wow, this fruit is so nice!" Her mom smiled and said, "Yes, it is. The star made the fruit so yummy. Let's eat it later." They both sat under the tree and enjoyed the rest of the day at the park.*
  * Parameters:
     Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `api_key` | `string` | Yes | - | Your unique API key for authentication. |
| `model` | `string` | Yes | `gpt-4-turbo` | The AI model to use for processing the request. |
| `temperature` | `number` | No | `0.7` | Controls randomness (0.0 = deterministic, 1.0 = creative). |
| `max_tokens` | `integer` | No | `1000` | The maximum number of tokens to generate in the response. |
| `stream` | `boolean` | No | `false` | If set to `true`, responses are streamed back incrementally. |
| `filters` | `array` | No | `[]` | An array of filter objects to apply to the input. |

  
## Env setup
Create environment

``` sh
conda env create -f environment.yml
```

## Download data
Download the TinyStories data

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

## Train BPE
Train BPE on the TinyStories data

``` sh
python train_bpe_tokenizer.py --vocab_size=[vocab_size:int] [params ...]
```

## Encoding
Tokenizing text

``` sh
python run_tokenizer.py [params ...]
```

## Train TransformerLM
Train the model

``` sh
python train.py --vocab_size=[vocab_size:int] [params ...]
```

## Decoding
Generating text

``` sh
python decode.py --vocab_size=[vocab_size:int] --device=[device:str] --model_file=[path:str] --prompt='...' [params ...]
```
