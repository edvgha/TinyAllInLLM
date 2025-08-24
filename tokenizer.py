import regex
import os
import concurrent.futures
from collections.abc import Iterable, Iterator
from bpe_tokenizer import load_vocab_json, load_merges_json


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, reverse=True) if special_tokens is not None else None
        self.token2id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        documents = self._split(text)

        n_jobs = os.cpu_count() or 1
        if n_jobs == 1 or len(documents) < 10_000:
            return self.serial_encoder(documents)

        print(f"n_jobs = {n_jobs}, n_docs = {len(documents)}")
        return self.parallel_encoder(documents)
    
    def serial_encoder(self, documents: list[str]) -> list[int]:
        enc: list[int] = []

        for document in documents:
            enc += self._encode_document(document)
            
        return enc
    
    def parallel_encoder(self, documents: list[str]) -> list[int]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(self._encode_document, documents)
            enc = [token_id for sublist in results for token_id in sublist]
            
        return enc

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        byte_sequence = []
        for id in ids:
            try:
                byte_sequence.append(self.vocab[id])
            except KeyError:
                byte_sequence.append(b'\xef\xbf\xbd') # UTF-8 for U+FFFD

        combined_bytes = b''.join(byte_sequence)

        return combined_bytes.decode('utf-8', errors='replace')
    
    def _encode_document(self, document: str) -> list[int]:
        enc: list[int] = []

        special_tokens_set = set(self.special_tokens) if self.special_tokens is not None else set()

        if document in special_tokens_set:
            enc.append(self.token2id[document.encode('utf-8')])
            return enc
        
        for match in regex.finditer(self.PAT, document):
            byte_list = self._apply_merges(match.group().encode('utf-8'))
            for byte in byte_list:
                enc.append(self.token2id[byte])
        
        return enc

    def _apply_merges(self, sequence: bytes) -> list[bytes]:
        byte_list = [bytes([byte]) for byte in sequence]
        for m in self.merges:
            i = 0
            while i < (len(byte_list) - 1):
                if (byte_list[i], byte_list[i + 1]) == m:
                    byte_list = byte_list[:i] + [byte_list[i] + byte_list[i + 1]] + byte_list[(i + 2):]
                else:
                    i += 1
        return byte_list

    def _split(self, text: str) -> list[str]:
        if self.special_tokens is None:
            return [text]
        
        pattern = '|'.join(map(regex.escape, self.special_tokens))
        documents = regex.split(f'({pattern})', text)

        return documents
    
    @staticmethod
    def _load_vocab(vocab_filepath: str) -> dict[int, bytes]:
        return load_vocab_json(vocab_filepath)

    @staticmethod
    def _load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
        return load_merges_json(merges_filepath)