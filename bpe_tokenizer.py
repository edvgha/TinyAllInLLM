import regex
import json
import base64
from collections import defaultdict


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i + 1: bytes([i]) for i in range(256)}
    vocab[0] = special_tokens[0].encode("utf-8")
    return vocab


def pre_tokenize(input_path: str, special_tokens: list[str]) -> dict[str, int]:
    """
        1. Splits on special token
        2. Pre tokenzie
        3. Word counts
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokens = defaultdict(int)

    with open(input_path, 'rb') as f:
        text = f.read().decode("utf-8")
        documents = regex.split(regex.escape(special_tokens[0]), text)
        for document in documents:
            for match in regex.finditer(PAT, document):
                pre_tokens[match.group()] += 1
    return pre_tokens


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    pre_tokens = pre_tokenize(input_path, special_tokens)

    byte_lists = defaultdict(list)
    pair_tokens_cnt = defaultdict(int)
    for pre_token, cnt in pre_tokens.items():
        if pre_token == special_tokens[0]:
            continue
        byte_list = [bytes([byte]) for byte in pre_token.encode('utf-8')]
        for f, s in zip(byte_list, byte_list[1:]):
            pair_tokens_cnt[(f, s)] += cnt
        byte_lists[pre_token] = byte_list

    while len(vocab) < vocab_size:
        max_pair_tokens_cnt = max(pair_tokens_cnt.values())
        candidates = [token_pair for token_pair, cnt in pair_tokens_cnt.items() if cnt == max_pair_tokens_cnt]
        candidate = max(candidates)
        merges.append(candidate)
        # apply merge
        for pre_token, byte_list in byte_lists.items():
            i = 0
            while i < len(byte_list) - 1:
                if (byte_list[i], byte_list[i + 1]) != candidate:
                    i += 1
                    continue
                # update counts
                if i != 0:
                    pair_tokens_cnt[(byte_list[i - 1], byte_list[i])] -= pre_tokens[pre_token]
                    pair_tokens_cnt[(byte_list[i - 1], byte_list[i] + byte_list[i + 1])] += pre_tokens[pre_token]
                pair_tokens_cnt[(byte_list[i], byte_list[i + 1])] -= pre_tokens[pre_token]
                if i + 2 < len(byte_list):
                    pair_tokens_cnt[(byte_list[i + 1], byte_list[i + 2])] -= pre_tokens[pre_token]
                    pair_tokens_cnt[(byte_list[i] + byte_list[i + 1], byte_list[i + 2])] += pre_tokens[pre_token]
                # merge
                byte_list = byte_list[:i] + [byte_list[i] + byte_list[i + 1]] + byte_list[(i + 2):]
            byte_lists[pre_token] = byte_list
        assert pair_tokens_cnt[candidate] == 0
        pair_tokens_cnt.pop(candidate)
        # add to vocab
        vocab[len(vocab)] = candidate[0] + candidate[1]

    return vocab, merges


def save_vocab_json(vocab: dict[int, bytes], file_path: str):
    encoded_vocab = {
        str(k): base64.b64encode(v).decode('ascii')
        for k, v in vocab.items()
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(encoded_vocab, f, indent=2)


def load_vocab_json(file_path: str) -> dict[int, bytes]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                loaded_encoded_vocab = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Error failed to load json: {file_path}")
        return {
            int(k): base64.b64decode(v.encode('ascii'))
            for k, v in loaded_encoded_vocab.items()
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocabulary file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading vocabulary from {file_path}: {e}")


def save_merges_json(merges: list[tuple[bytes, bytes]], file_path: str):
    list_of_lists_b64 = []
    for pair in merges:
        b1_b64 = base64.b64encode(pair[0]).decode('ascii')
        b2_b64 = base64.b64encode(pair[1]).decode('ascii')
        list_of_lists_b64.append([b1_b64, b2_b64])
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(list_of_lists_b64, f, indent=2)


def load_merges_json(file_path: str) -> list[tuple[bytes, bytes]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                list_of_lists_b64 = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Error failed to load json: {file_path}")
        merges = []
        for pair_b64 in list_of_lists_b64:
            b1 = base64.b64decode(pair_b64[0].encode('ascii'))
            b2 = base64.b64decode(pair_b64[1].encode('ascii'))
            merges.append((b1, b2))
        return merges
    except FileNotFoundError:
        raise FileNotFoundError(f"Merges file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading Merges from {file_path}: {e}")