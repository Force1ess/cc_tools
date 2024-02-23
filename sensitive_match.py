import collections
import os
from time import time

from tool_funcs import load_file_by_line


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False
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
