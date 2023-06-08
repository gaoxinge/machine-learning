import random
from typing import Optional, List
from abc import ABC, abstractmethod

ENG_CHARS = "abcdefghijklmnopqrstuvwxyz"


class Tokenizer(ABC):

    @abstractmethod
    def next(self) -> Optional[str]:
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def next(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def reset(self) -> 'SimpleTokenizer':
        self.pos = 0
        return self


class CharTokenizer(Tokenizer):

    def __init__(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.pos = 0

    def next(self) -> Optional[str]:
        if self.pos >= len(self.text):
            return None
        token = self.text[self.pos]
        self.pos += 1
        return token


class WordTokenizer(Tokenizer):

    def __init__(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.pos = 0

    def next(self) -> Optional[str]:
        while self.pos < len(self.text):
            c = self.text[self.pos].lower()
            if c == " ":
                self.pos += 1

        if self.pos >= len(self.text):
            return None

        c = self.text[self.pos].lower()
        if c not in ENG_CHARS:
            self.pos += 1
            return c

        token = ""
        while self.pos < len(self.text):
            c = self.text[self.pos].lower()
            if c not in ENG_CHARS:
                break
            self.pos += 1
            token += c
        return token


class BiGram:

    def __init__(self, tokenizer: Tokenizer):
        tokens = set()
        cnt1 = {}
        cnt2 = {}

        last_token = None
        while True:
            token = tokenizer.next()
            if token is None:
                break

            tokens.add(token)

            if token not in cnt1:
                cnt1[token] = 0
            cnt1[token] += 1

            if last_token is not None:
                p = (last_token, token)
                if p not in cnt2:
                    cnt2[p] = 0
                cnt2[p] += 1

            last_token = token

        frequency = {}
        for k, v in cnt2.items():
            frequency[k] = v / cnt1[k[0]]

        self.tokens = tokens
        self.frequency = frequency

    def next(self, token):
        if token not in self.tokens:
            return random.choice(list(self.tokens))

        population = []
        weights = []
        for next_token in self.tokens:
            population.append(next_token)
            p = (token, next_token)
            if p not in self.frequency:
                weights.append(0)
            else:
                weights.append(self.frequency[p])

        return random.choices(population=population, weights=weights, k=1)[0]


def test_bigram():
    tokenizer = SimpleTokenizer(["a", "c", "b", "c"])
    bigram = BiGram(tokenizer)
    print(bigram.tokens)
    print(bigram.frequency)
    print(bigram.next("a"))
    print(bigram.next("b"))
    print(bigram.next("c"))


def test_char_tokenizer():
    tokenizer = CharTokenizer("shakespeare.txt")
    bigram = BiGram(tokenizer)
    token = ""
    result = [token]
    for _ in range(1000):
        token = bigram.next(token)
        result.append(token)
    print("".join(result))


def test_word_tokenizer():
    tokenizer = WordTokenizer("shakespeare.txt")
    bigram = BiGram(tokenizer)
    token = ""
    result = [token]
    for _ in range(1000):
        next_token = bigram.next(token)
        if next_token[0] in ENG_CHARS:
            result.append(" ")
            if token[0] not in ENG_CHARS:
                result.append(next_token[0].upper() + next_token[1:])
            else:
                result.append(next_token)
        else:
            result.append(next_token)
        token = next_token
    print("".join(result))


if __name__ == "__main__":
    test_bigram()

    print("=" * 80)
    test_char_tokenizer()

    print("=" * 80)
    test_word_tokenizer()

