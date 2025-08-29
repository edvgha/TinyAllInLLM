# TinyAllInLLM

A lightweight language model built entirely from scratch, designed to be simple, understandable, and easy to train on a standard laptop.

## About the project
**Tiny-All-In-LLM** is a minimalist, yet complete, implementation of the 
transformer architecture.
* **Education:** To demystify the core mechanics of the transformer architecture by providing a clear codebase.
* **Accessibility:** To create a model that can be trained and run without requiring expensive, high-end hardware.
* **Simplicity:** To focus on a "from-scratch" approach with minimal dependencies.
  
![Alternative Text](https://github.com/edvgha/TinyAllInLLM/blob/main/img/loss.png)

## Table of Contents

- [Experiment](#experiment)
- [Env setup](#env-setup)
- [Data](#download-data)
- [Train BPE](#train-bpe)
- [Encoding](#encoding)
- [Train TransformerLM](#train-transformer-lm)
- [Decoding](#decoding)

## Experiment
  * Data:
     | Data |
     | :--- |
     | `TinyStoriesV2-GPT4` |

  * Machine:
     | Chip | Memory |
     | :--- | :--- |
     | `Apple M4 Pro` | `24 GB` |

  * Eval:
     | Train loss | Val loss | Train perplexity | Val perplexity |
     | :--- | :--- | :--- | :--- |
     | `1.439` | `1.447` | `4.22` | `4.25` |

  * Parameters:
     |Parameter | Value |
     | :--- | :--- |
     | `vocab_size` | `10000` |
     | `embedding_dim` | `256` |
     | `num_layers` | `8` |
     | `num_heads` | `8` |
     | `d_ff` | `682` |
     | `rope_theta` | `10000` |
     | `weight_decay` | `1e-5` |
     | `grad_clip` | `1.0` |
     | `max_lr` | `1e-3` |
     | `min_lr` | `1e-4` |
     | `warmup_iters` | `50` |
     | `cosine_cycle_iters` | `40000` |
     | `batch_size` | `128` |
     | `context_length` | `128` |
     | `max_iters` | `50000` |
     | `device` | `MPS` |

     |Parameter | Value |
     | :--- | :--- |
     | `temperature` | `1` |
     | `top_p` | `0.8` |

  * Decoding example:
    - Prompt: > ***The next day, Lily and her mom went to the part. They saw***
    - Complition: > ***The next day, Lily and her mom went to the park. They saw the big tree again. This time, Lily was ready. She saw the shiny, smooth, red fruit. She took a bite and said, "Wow, this fruit is so nice!" Her mom smiled and said, "Yes, it is. The star made the fruit so yummy. Let's eat it later." They both sat under the tree and enjoyed the rest of the day at the park.***
  
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
