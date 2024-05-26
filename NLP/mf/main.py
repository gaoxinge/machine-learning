"""
- https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
"""
import os
import logging
from typing import Optional, List
from collections import OrderedDict
from abc import ABC, abstractmethod

import tqdm
import numpy as np

ENG_CHARS = "abcdefghijklmnopqrstuvwxyz"


class Tokenizer(ABC):

    @abstractmethod
    def next(self) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> 'Tokenizer':
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.index = 0

    def next(self) -> Optional[str]:
        if self.index >= len(self.tokens):
            return None
        token = self.tokens[self.index]
        self.index += 1
        return token

    def reset(self) -> 'Tokenizer':
        self.index = 0
        return self


class FileTokenizer(Tokenizer):

    def __init__(self, filename: str):
        lines = []
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line.strip())

        cnt1, cnt2 = None, None
        for i, line in enumerate(lines):
            if line.startswith("Lines:"):
                try:
                    cnt1 = int(line[6:].strip())
                    break
                except Exception as e:
                    logging.warning("parse int with error %s", e)
            if cnt2 is None and len(line) == 0:
                cnt2 = i

        if cnt1 is not None:
            lines = lines[-cnt1:]
        elif cnt2 is not None:
            lines = lines[cnt2:]

        self.lines = "\n".join(lines)
        self.pos = 0

    def _next(self) -> Optional[str]:
        while self.pos < len(self.lines):
            c = self.lines[self.pos].lower()
            if c in ENG_CHARS:
                break
            self.pos += 1

        if self.pos >= len(self.lines):
            return None

        token = ""
        while self.pos < len(self.lines):
            c = self.lines[self.pos].lower()
            if c not in ENG_CHARS:
                break
            token += c
            self.pos += 1
        return token

    def next(self) -> Optional[str]:
        while True:
            token = self._next()
            if token is None or len(token) > 1:
                return token

    def reset(self) -> 'Tokenizer':
        self.pos = 0
        return self


def test_file_tokenizer(filename):
    tokenizer = FileTokenizer(filename)
    print(tokenizer.lines)

    tokens = []
    while True:
        token = tokenizer.next()
        if token is None:
            break
        tokens.append(token)
    print(tokens)


def check_file_tokenizer():
    files = [os.path.join("20_newsgroups", file) for file in os.listdir("20_newsgroups")]
    files = [os.path.join(file, file1) for file in files for file1 in os.listdir(file)]
    for file in tqdm.tqdm(files):
        try:
            FileTokenizer(file)
        except Exception as e:
            print(file, e)


def build(tokenizers: List[Tokenizer], mode: int = 0):
    tokens = OrderedDict()

    for tokenizer in tokenizers:
        while True:
            token = tokenizer.next()
            if token is None:
                break
            tokens[token] = 0

    for tokenizer in tokenizers:
        tokenizer.reset()

    m = np.array([[] for _ in range(len(tokens))])
    for tokenizer in tokenizers:
        for token in tokens:
            tokens[token] = 0
        while True:
            token = tokenizer.next()
            if token is None:
                break
            tokens[token] = (0 if mode == 0 else tokens[token]) + 1

        c = np.array([[cnt] for token, cnt in tokens.items()])
        m = np.append(m, c, axis=1)

    for token in tokens:
        tokens[token] = 0
    return tokens, m


if __name__ == "__main__":
    test_file_tokenizer("20_newsgroups\\alt.atheism\\49960")
    check_file_tokenizer()

    tokenizer1 = SimpleTokenizer(["this", "is", "a", "a", "sample"])
    tokenizer2 = SimpleTokenizer(["this", "is", "another", "another", "example", "example", "example"])
    tokens, m = build([tokenizer1, tokenizer2], mode=0)
    print(tokens.keys())
    print(m)

    tokenizer1 = SimpleTokenizer(["this", "is", "a", "a", "sample"])
    tokenizer2 = SimpleTokenizer(["this", "is", "another", "another", "example", "example", "example"])
    tokens, m = build([tokenizer1, tokenizer2], mode=1)
    print(tokens.keys())
    print(m)

    files = [os.path.join("20_newsgroups", file) for file in os.listdir("20_newsgroups")]
    files = [os.path.join(file, file1) for file in files for file1 in os.listdir(file)]
    files = files[:500]
    tokenizers = [FileTokenizer(train_file) for train_file in files]
    tokens, m = build(tokenizers, mode=0)
    print(len(tokens.keys()))
    print(m.shape)
    U, S, Vh = np.linalg.svd(m)
    print(U.shape, S.shape, Vh.shape)
