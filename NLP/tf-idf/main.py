"""
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
"""
import os
import math
import logging
from typing import Optional, List
from collections import OrderedDict
from abc import ABC, abstractmethod

ENG_CHARS = "abcdefghijklmnopqrstuvwxyz"


class Tokenizer(ABC):

    @abstractmethod
    def next(self) -> Optional[str]:
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

    def reset(self) -> 'SimpleTokenizer':
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
    for file in files:
        try:
            FileTokenizer(file)
        except Exception as e:
            print(file, e)


def _build_tf_from_tokenizer(tokenizer: Tokenizer):
    n = 0
    tf_map = OrderedDict()

    while True:
        token = tokenizer.next()
        if token is None:
            break

        n += 1
        if token not in tf_map:
            tf_map[token] = 0
        tf_map[token] += 1

    for token, tf in tf_map.items():
        tf_map[token] = tf / n
    return tf_map


def _build_idf_from_tokenizers(tokenizers: List[Tokenizer], thresh: int = 0, base: str = "e"):
    idf_map = OrderedDict()

    for tokenizer in tokenizers:
        token_set = set()
        while True:
            token = tokenizer.next()
            if token is None:
                break
            token_set.add(token)

        for token in token_set:
            if token not in idf_map:
                idf_map[token] = 0
            idf_map[token] += 1

    new_idf_map = OrderedDict()
    for token, idf in idf_map.items():
        if idf <= thresh:
            continue
        new_idf = len(tokenizers) / idf
        if base == "10":
            new_idf_map[token] = math.log10(new_idf)
        else:
            new_idf_map[token] = math.log(new_idf)
    return new_idf_map


class TFIDF:

    def __init__(self, tokenizers: List[Tokenizer], thresh: int = 0, base: str = "e"):
        self.idf_map = _build_idf_from_tokenizers(tokenizers, thresh, base)

    @classmethod
    def build_tf_map(cls, tokenizer: Tokenizer):
        return _build_tf_from_tokenizer(tokenizer)

    def build_tf_idf_map(self, tokenizer: Tokenizer):
        tf_map = _build_tf_from_tokenizer(tokenizer)
        tf_idf_map = {}
        for token, tf in tf_map.items():
            if token not in self.idf_map:
                tf_idf_map[token] = 0
            else:
                tf_idf_map[token] = tf * self.idf_map[token]
        return tf_idf_map

    def build_tf_idf_vector(self, tokenizer: Tokenizer):
        tf_map = _build_tf_from_tokenizer(tokenizer)
        tf_idf_vector = []
        for token, idf in self.idf_map.items():
            if token not in tf_map:
                tf_idf_vector.append(0)
            else:
                tf_idf_vector.append(tf_map[token] * idf)
        return tf_idf_vector


def test_tf_idf():
    tokenizer1 = SimpleTokenizer(["this", "is", "a", "a", "sample"])
    tokenizer2 = SimpleTokenizer(["this", "is", "another", "another", "example", "example", "example"])

    tf_idf = TFIDF([tokenizer1, tokenizer2], base="10")
    print(tf_idf.idf_map["this"], 0)
    print(tf_idf.idf_map["example"], 0.301)

    tokenizer1_tf_map = tf_idf.build_tf_map(tokenizer1.reset())
    tokenizer2_tf_map = tf_idf.build_tf_map(tokenizer2.reset())
    tokenizer1_tf_idf_map = tf_idf.build_tf_idf_map(tokenizer1.reset())
    tokenizer2_tf_idf_map = tf_idf.build_tf_idf_map(tokenizer2.reset())

    print(tokenizer1_tf_map["this"], 0.2)
    print(tokenizer2_tf_map["this"], 0.14)
    print(tokenizer1_tf_idf_map["this"], 0)
    print(tokenizer2_tf_idf_map["this"], 0)

    print("example" not in tokenizer1_tf_map, True)
    print(tokenizer2_tf_map["example"], 0.429)
    print("example" not in tokenizer1_tf_map, True)
    print(tokenizer2_tf_idf_map["example"], 0.129)


if __name__ == "__main__":
    # test_file_tokenizer("20_newsgroups\\alt.atheism\\49960")
    # check_file_tokenizer()
    # test_tf_idf()

    files = [os.path.join("20_newsgroups", file) for file in os.listdir("20_newsgroups")]
    files = [os.path.join(file, file1) for file in files for file1 in os.listdir(file)]

    tf_idf = TFIDF([FileTokenizer(train_file) for train_file in files], thresh=3)
    idf_sequence = [(k, v) for k, v in tf_idf.idf_map.items()]
    idf_sequence = sorted(idf_sequence, key=lambda kv: kv[1], reverse=True)
    with open("idf.txt", "w", encoding="utf-8") as f:
        for k, v in idf_sequence:
            f.write(f"{k}: {v}\n")

    with open("result.txt", "w", encoding="utf-8") as f:
        for file in files:
            tf_idf_map = tf_idf.build_tf_idf_map(FileTokenizer(file))
            tf_idf_sequence = [(k, v) for k, v in tf_idf_map.items()]
            tf_idf_sequence = sorted(tf_idf_sequence, key=lambda kv: kv[1], reverse=True)
            tf_idf_str = " ".join([k for k, v in tf_idf_sequence][:10])
            f.write(f"{file}: {tf_idf_str}\n")

