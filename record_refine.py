import os
import shutil
import random
from typing import List

import pandas as pd
import logging
from simhash import Simhash

from sensitive_match import Trie_tree
from tool_funcs import dir_check, pathjoin

# BUCKET_SIZE = 300_000
BUCKET_SIZE = 100_0


def record_refine(outdir: str, domain_list: List[str]):
    dir_check(pathjoin(outdir, "hash"))
    pid = str(os.getpid())
    dst_dir = pathjoin(outdir, "bucket" + pid)
    dir_check(dst_dir)
    trie_forest = {}
    badwords_path = "resource/badwords/"
    for file in os.listdir(badwords_path):
        if file.endswith(".txt"):
            trie_forest[file[:-4]] = Trie_tree()
            trie_forest[file[:-4]].load_vocab(pathjoin(badwords_path, file))

    num_records = 0
    num_files = 0
    records = []
    hash = []
    for domain in domain_list:
        df = pd.read_parquet(domain)
        # shutil.rmtree(domain)
        # ?? 暂时够用，就不删了吧
        for lang, df_slice in df.groupby("language"):
            try:
                data_slice = domain_dedup(lang, df_slice, trie_forest.get(lang, None))
                if isinstance(data_slice, pd.DataFrame):
                    records.append(data_slice)
                    num_records += len(data_slice)
            except Exception as e:
                logging.warning(f"error in {lang} {domain}: {e}")

        if num_records > BUCKET_SIZE or domain == domain_list[-1]:
            datapoints = pd.concat(records).drop("length", axis=1)
            del records
            dst = pathjoin(dst_dir, f"{num_files:06d}.parquet")
            logging.info(f"writing {num_records} reocrds to {dst}")
            datapoints.to_parquet(dst)
            hash.extend(
                [
                    (f"{pid}/{num_files:06d}", str(i), str(v))
                    for i, v in enumerate(
                        datapoints["text"].apply(lambda x: Simhash(x).value)
                    )
                ]
            )

            del datapoints
            records = []
            num_files += 1
            num_records = 0
    hash_file = pathjoin(outdir, "hash", f"{os.getpid()}.csv")
    hash.sort(key=lambda x: x[2])
    with open(hash_file, "w", buffering=128 * 1024 * 1024) as f_hash:
        f_hash.write("".join([",".join(i) + "\n" for i in hash]))
    del hash


def domain_dedup(
    lang: str,
    datapoints: pd.DataFrame,
    trie_tree: Trie_tree,
):
    del_idxs = set()
    orig_len = len(datapoints)
    sens_count = 0
    if trie_tree is not None:
        num_trie = min(50, len(datapoints))
        for i, row in enumerate(datapoints.itertuples(index=False)):
            if i == num_trie:
                break

            if not trie_tree.query(row.text):
                del_idxs.add(i)
                sens_count += 1

        if sens_count / orig_len > 0.1:
            return []  # 超出10%直接丢弃

    datapoints = datapoints.assign(length=datapoints["text"].apply(lambda x: len(x)))
    datapoints["text"] = datapoints["text"].apply(lambda x: x.split("\n"))
    datapoints: list = datapoints.to_dict("records")

    min_textlen, min_blocklen = get_min_len(lang)
    fail_count = 0
    repeat_time = max(20, min(500, len(datapoints) // 5))  # 最多1000次，最少20次
    while fail_count < repeat_time:
        if len(del_idxs) > 0:
            datapoints = [
                datapoints[i] for i in range(len(datapoints)) if i not in del_idxs
            ]
            del_idxs.clear()

        if len(datapoints) / orig_len < 0.6:
            return []
        if len(datapoints) < 7:
            break
        fail_count += 1
        idx_a = random.randint(0, len(datapoints) - 1)
        idx_b = random.randint(0, len(datapoints) - 1)

        if idx_a == idx_b:
            continue

        a, b = datapoints[idx_a], datapoints[idx_b]
        paras_a = a["text"]
        paras_b = b["text"]
        small_sus_pattern = {i: 0 for i in paras_a if i in paras_b and i}  # 小段完全相同

        if len(small_sus_pattern) == 0:
            continue

        for _ in range(repeat_time // 2):
            idx_c = random.randint(0, len(datapoints) - 1)
            if idx_c == idx_a or idx_c == idx_b or idx_c in del_idxs:
                continue
            paras_c = datapoints[idx_c]["text"]

            for pattern in small_sus_pattern:
                if small_sus_pattern[pattern] < 3 and pattern in paras_c:
                    small_sus_pattern[pattern] += 1
                if small_sus_pattern[pattern] != 3:
                    continue
                small_sus_pattern[pattern] += 1
                fail_count = 0
                for content_id in range(len(datapoints)):
                    if content_id in del_idxs:
                        continue
                    for para_idx in range(len(datapoints[content_id]["text"])):
                        if datapoints[content_id]["text"][para_idx] == pattern:
                            datapoints[content_id]["length"] -= len(pattern)
                            datapoints[content_id]["text"].pop(para_idx)
                            break
                    if datapoints[content_id]["length"] < min_textlen:
                        del_idxs.add(content_id)

    fail_count = 0
    while fail_count < repeat_time:
        fail_count += 1
        if len(del_idxs) > 0:
            datapoints = [
                datapoints[i] for i in range(len(datapoints)) if i not in del_idxs
            ]
            del_idxs.clear()
        if len(datapoints) / orig_len < 0.6:
            return []
        if len(datapoints) < 7:
            break
        idx_a = random.randint(0, len(datapoints) - 1)
        idx_b = random.randint(0, len(datapoints) - 1)
        if idx_a == idx_b:
            continue
        a, b = datapoints[idx_a], datapoints[idx_b]
        lcs = ngram_lcs("\n".join(a["text"]), "\n".join(b["text"]), min_blocklen)
        if not lcs:
            continue
        pattern_count = 0

        for _ in range(repeat_time):
            idx_c = random.randint(0, len(datapoints) - 1)
            if idx_c == idx_a or idx_c == idx_b or idx_c in del_idxs:
                continue
            if ngram_match(lcs, datapoints[idx_c]["text"], min_blocklen)[0]:
                pattern_count += 1
            if pattern_count != 3:
                continue
            fail_count = 0
            for content_id in range(len(datapoints)):
                para_idx, paralen = ngram_match(
                    lcs, datapoints[content_id]["text"], min_blocklen
                )
                if para_idx == 0:
                    continue
                datapoints[content_id]["text"].pop(para_idx - 1)
                datapoints[content_id]["length"] -= paralen
                if datapoints[content_id]["length"] < min_textlen:
                    del_idxs.add(content_id)
            break
    # TODO 平均段长度
    datapoints = pd.DataFrame(datapoints)
    datapoints["text"] = (
        datapoints["text"].apply(lambda x: "\n".join(x)).astype("string")
    )
    if sens_count != 0:
        datapoints = datapoints[datapoints["text"].apply(trie_tree.query)]

    return datapoints


def generate_ngrams(s, n, step):
    ngrams = []
    for i in range(0, len(s) - n + 1, step):
        ngrams.append(s[i : i + n])
    return ngrams


def ngram_lcs(pattern, text, min_n=30) -> str:
    pattern_len = len(pattern)
    if pattern_len > len(text):
        pattern, text = text, pattern
        pattern_len = len(pattern)
    if pattern_len < min_n:
        return ""

    n, step = min_n, 10
    ngrams = generate_ngrams(pattern, n, step)
    for ngram in ngrams:
        if ngram in text and "\n" not in ngram:
            return ngram
    return ""


def ngram_match(ngram: List[str], text: List[str], min_len: int) -> int:
    for idx, para in enumerate(text):
        if len(para) < min_len:
            continue
        if ngram in para:
            return idx + 1, len(para)
    return 0, 0


def get_min_len(language: str) -> tuple[int, int]:
    if language == "zh":
        return 100, 30
    else:
        return 160, 50
