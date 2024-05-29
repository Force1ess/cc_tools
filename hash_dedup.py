import logging
import sys
import json
from shutil import copyfile
import jsonlines
import os
import pathlib
import subprocess

pathjoin = os.path.join

# gen simhash   后使用sort
# 以dump为单位去重，所有一起去重开销太大
# 有了哈希去重感觉small_pattern可以去掉了
# regex应该在trafilatura的时候做？
def distance(value1, value2, dim_bits=64):
    x = (value1 ^ value2) & ((1 << dim_bits) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ans


def get_filtered_table():
    f_read_bad_para = open("hash-values-bad-para.txt", "r")
    f_read_hash_value = open("hash-values-sorted.csv", "r")
    f_write = open("hash-values-sorted-filtered.csv", "w")

    bad_paras = f_read_bad_para.readlines()
    bad_paras = [item.strip().split(",")[-1] for item in bad_paras]
    table = f_read_hash_value.readlines()

    uuid2idx = {}
    for idx, line in enumerate(table):
        uuid2idx[line.strip().split(",")[1]] = idx

    bad_idx_list = []
    for bad_para in bad_paras:
        bad_idx_list.append(uuid2idx[bad_para])

    good_idx_list = list(set(range(len(table))) - set(bad_idx_list))
    table_good = [table[i] for i in good_idx_list]
    f_write.writelines(table_good)

    print(len(table))
    print(len(bad_paras))
    print(len(table_good))


def gen_dedup_idx(to_output_orderly=False):
    """
    根据排序后的 SimHash 值，生成坏数据的 UUID
    其中对于重复的数据集合，只随机保留一个，剩余都视为坏数据

    input_path: 原数据文件路径
    to_output_orderly: 是否输出有序的坏数据的 UUID 的列表
    """
    f_write = open("hash-values-bad-para.txt", "w")
    f_read = open("hash-values-sorted.csv", "r")

    data = f_read.readlines()
    bad_para = set()  # set()，元素无序，无法对bad-para内容进行检索查看
    bad_para_original = []
    tmp_bad_para = set()

    ptr = 1
    idx = 0
    max_len = len(data)
    while idx < max_len - 1:
        cont1 = data[idx].split(",")
        cont2 = data[ptr].split(",")
        value1 = int(cont1[-1])
        value2 = int(cont2[-1])

        if distance(value1, value2) < 3:
            if to_output_orderly:
                bad_para_original.append(f"{cont1[0]},{cont1[1]}\n")
                bad_para_original.append(f"{cont2[0]},{cont2[1]}\n")

            # 将距离小于3的数据样本放置于临时的set中，并不是直接加入到最终的bad_para
            tmp_bad_para.add(f"{cont1[0]},{cont1[1]}\n")
            tmp_bad_para.add(f"{cont2[0]},{cont2[1]}\n")

            ptr += 1
        elif ptr - idx > 1:
            idx = ptr - 1
            ptr = idx + 1

            # 对于set内所有两两距离小于3的数据样本，只保留一个
            list_tmp_bad_para = list(tmp_bad_para)
            for i in range(1, len(list_tmp_bad_para)):
                bad_para.add(list_tmp_bad_para[i])
            tmp_bad_para = set()
        else:
            idx = ptr
            ptr = idx + 1

        if ptr >= max_len:
            idx += 1
            ptr = idx + 1

        if idx % 500000 == 0:
            print(idx, len(bad_para))

    # 处理剩余数据样本
    list_tmp_bad_para = list(tmp_bad_para)
    for i in range(1, len(list_tmp_bad_para)):
        bad_para.add(list_tmp_bad_para[i])
    tmp_bad_para = set()

    bad_para_list = list(bad_para)
    f_write.writelines(bad_para_list)
    f_write.close()

    if to_output_orderly:
        with open( "hash-values-bad-para-original.txt", "w") as f:
            f.writelines(bad_para_original)

    file2uuid = {}

    for line in bad_para_list:
        file = line.split(",")[0]
        uuid = line.split(",")[1]
        if file not in file2uuid.keys():
            file2uuid[file] = []
        file2uuid[file].append(uuid)

    for file, uuids in file2uuid.items():
        if len(uuids) != 0:
            print(f"{file}")
            f_bad_para_write = open(f"{file}-bad-para.txt", "w")
            f_bad_para_write.writelines(uuids)

def hash_dedup():
    # task 的操作要改一下 这个操作要改一下
    try:
        assert os.path.exists(pathjoin(dump_dir, "hash"))
        
        cmd = (
            "sort --parallel=16 -m -t , -nk3 * > hash-values-sorted.csv"  # 将 N 替换为你的线程数
        )
        subprocess.run(cmd, shell=True, capture_output=False, text=True)
        assert os.path.getsize("hash-values-sorted.csv") != 0
    except Exception as e:
        logging.error(f"{dump_dir}: Exception {e}")


if __name__ == "__main__":
    dump_dir = sys.argv[1]
    hash_folder = pathjoin(dump_dir, "hash")
    os.chdir(hash_folder)
    hash_dedup()
    gen_dedup_idx()
    get_filtered_table()
