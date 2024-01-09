import collections
import os
from time import time

from tool_funcs import load_file_by_line


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

    def __repr__(self):
        s = ""
        first = True
        for k, v in self.children.items():
            if first:
                if v.is_word:
                    s += "{} -> {}\n".format(k, v)
                else:
                    s += "{} -> {}".format(k, v)
                first = False
                continue
            if v.is_word:
                s += "{}\n".format(k)
            else:
                s += "{} -> {}".format(k, v)
        return s


class Trie_tree:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def load_vocab(self, vocabs):
        if os.path.exists(vocabs):
            vocabs = load_file_by_line(vocabs)
        for v in vocabs:
            self.insert(v)

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def __repr__(self):
        return repr(self.root).replace("\n\n", "\n").replace("\n\n", "\n")

    def find_one(self, word):
        """找到第一个匹配的词

        :param word: str
        :return: 第一个匹配的词 or None

        >>> a = Trie()
        >>> a.insert('感冒')
        >>> a.find_one('我感冒了好难受怎么办')
        '感冒'
        """
        res = []
        for i in range(len(word)):
            c = word[i]
            node = self.root.children.get(c)
            if node:
                for j in range(i + 1, len(word)):
                    _c = word[j]
                    node = node.children.get(_c)
                    if node:
                        if node.is_word:
                            res.append(word[i : j + 1])
                    else:
                        break
        return res

    def query(self, word: str, threshold=2, max_length=2000):
        word = word[:max_length].lower()  # 限制单词长度
        times = 0
        node = self.root
        for c in word:
            if c in node.children:
                node = node.children[c]
                if node.is_word:
                    times += 1
                    if times > threshold:
                        return False
            else:
                node = self.root  # 重置为根节点
        return True


if __name__ == "__main__":
    a = Trie_tree()
    from tool_funcs import load_file_by_line

    a.load_vocab(load_file_by_line("./sens_words.txt"))
    import pandas as pd

    s = time()
    df = pd.read_parquet(
        "/home/zhenghao2022/common_crawl/100GB/1.text_extracted/batch1"
    )
    print(sum(df["text"].apply(a.query)))

    print(time() - s)
