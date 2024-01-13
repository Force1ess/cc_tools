import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from archive_process import warc_process
from path_get import (YearMonth, download_path, get_ccmain_index,
                      get_warc_path, merge_path, month_test)
from record_refine import record_refine
from tool_funcs import (NUM_PROCESS, args_wrapper, batchfy, blockfy, dir_check,
                        exec_command, get_path_name, pathjoin, pool_exec,
                        process_nested_dict, send_feishu, setup)
from warc_download import aria2c_download

parser = argparse.ArgumentParser(description="用于CommonCrawl数据的下载与处理")
subparsers = parser.add_subparsers(dest="action")

path_parser = subparsers.add_parser("getpath", help="下载的两个时间戳内的path, eg:2013/05")
path_parser.add_argument("--start_timestamp", "-s", help="eg:2013/05")
path_parser.add_argument("--end_timestamp", "-e", help="eg:2018/05")
path_parser.add_argument("--type", "-t", help="类型: cc-news, cc-main")

rank2domain_parser = subparsers.add_parser("rank2domain", help="转换ranks为domain,需要安装go")
rank2domain_parser.add_argument("rank_path", help="CommonCrawl rank地址")

process_parser = subparsers.add_parser("process", help="处理warc/wet")
process_parser.add_argument("--input", "-i", help="待处理文件的文件夹地址")
process_parser.add_argument("--output", "-o", help="输出文件的文件夹地址")
process_parser.add_argument("--type", "-t", help="warc or wet")
process_parser.add_argument("-n", "--num_process", help="warc: 同时处理的线程数")
process_parser.add_argument("-c", "--chunksize", help="wet: 一个job处理的chunk数，和内存大小接近较好，应为cpu核数-6的倍数")

download_parser = subparsers.add_parser("download", help="下载指定path中的文件")
download_parser.add_argument("-i", "--input", help="目标path文件")
download_parser.add_argument("-o", "--output", help="下载位置")
download_parser.add_argument("-n", "--num", help="同时下载的线程数")

refine_parse = subparsers.add_parser("refine", help="对处理后的数据进行语言分类、去重和分割")
refine_parse.add_argument("-i", "--input", help="输入文件夹地址")
refine_parse.add_argument("-o", "--output", help="输出文件夹地址")
refine_parse.add_argument("-n", "--num_process", help="同时处理的线程数")

if __name__ == "__main__":
    start_time = time.time()
    setup()
    logging.info(sys.argv)
    actions = ["getpath", "rank2domain", "process", "download", "refine"]
    args = args_wrapper(parser.parse_args())
    if args.action == actions[0]:
        dir_check("cache")
        ym = YearMonth(args.end_timestamp)
        num_process = args.num if args.num else 4
        if args.type == "cc-news":
            month_test(ym)
            path_links = []
            for timestamp in ym.iter(args.start_timestamp):
                path_links.append(get_warc_path(timestamp))
            failed_path = pool_exec(download_path, path_links, num_process=num_process)

        elif args.type == "cc-main":
            indexs = get_ccmain_index()
            y_start = (
                0
                if args.start_timestamp is None
                else int(args.start_timestamp.split("/")[0])
            )
            y_end = (
                9999
                if args.end_timestamp is None
                else int(args.end_timestamp.split("/")[0])
            )
            path_links = []
            for index in indexs:
                year = int(index[-7:-3])
                if y_start <= year and year <= y_end:
                    path_links.append(get_warc_path(index, type="cc-main"))
            failed_path = pool_exec(download_path, path_links, num_process=num_process)
        else:
            raise (ValueError("Invalid type"))

        logging.info(
            f"path download finished, expect {len(path_links)}, found: {len([i for i in os.listdir('cache') if i in map(get_path_name, path_links)])}"
        )
        merge_path("cache", args.type)
        if failed_path is not None:
            logging.info(f"FAILED: {failed_path}")

    elif args.action == actions[1]:
        assert os.path.isfile(args.rank_path)
        exec_command("go", ["run", "domainExtract.go", args.rank_path])
    elif args.action == actions[2]:
        download_dir = args.input
        output_dir = dir_check(args.output)
        downloaded_archives = {}
        files = []
        if args.type == "warc":
            num_process = int(args.num_process) if args.num_process else NUM_PROCESS
            for filename in os.listdir(download_dir):
                file_path = pathjoin(download_dir, filename)
                files.append(file_path)
            archive_process = partial(warc_process, output_dir)
            pool_exec(
            archive_process,
            files,
            num_process=num_process,
        )
        elif args.type == "wet":
            chunksize = int(args.chunksize) if args.num_process else NUM_PROCESS
            exec_command('./resource/bin/cc_tool',
                         [download_dir, output_dir, chunksize, chunksize//2]
                         )

    elif args.action == actions[3]:
        path_file = args.input if args.input else "merged_path"
        dir_check("cache")
        with open(path_file, "r") as f:
            download_table = json.load(f)
        output_dir = Path(args.output if args.output else "./0.downloaded")
        dir_check(output_dir)
        if isinstance(download_table, dict):
            download_table = process_nested_dict(download_table)
        undownload_table = deepcopy(download_table)
        idx = 0
        while True:
            corpus_id, links = download_table[idx]
            if int(corpus_id[8:12]) < 2019:  # 陆老师及Windows下载
                logging.info(f"{corpus_id} jumped")
                idx += 1
                continue
            dir_check(f"{output_dir}/{corpus_id}")
            files = os.listdir(f"{output_dir}/{corpus_id}")
            for i in files:
                if i.endswith(".aria2"):
                    files.pop(files.index(i.removesuffix(".aria2")))
                    files.remove(i)

            download_links = [i for i in links if i.split("/")[-1] not in files]
            if len(download_links) != 0:
                with open("temp/download.tmp", "w") as f:
                    f.write("\n".join(download_links))
                logging.info(f"Downloading {output_dir}/{corpus_id}")
                aria2c_download(f"{output_dir}/{corpus_id}", args.num)
                files = os.listdir(f"{output_dir}/{corpus_id}")
                for i in files:
                    if i.endswith(".aria2"):
                        files.pop(files.index(i.removesuffix(".aria2")))
                        files.remove(i)
                if len(files) != len(links):
                    logging.fatal(f"{corpus_id} download failed: ")
                    continue
            undownload_table = [i for i in undownload_table if i[0] != corpus_id]
            with open("temp/undownload.tmp", "w") as f:
                f.write(json.dumps(undownload_table))
            logging.info(f"Corpus {corpus_id} downloaded")
            idx += 1
    elif args.action == actions[4]:
        num_process = int(args.num_process) if args.num_process else NUM_PROCESS
        dump_dir = args.input
        out_dir = args.output
        record_refine = partial(record_refine, out_dir)
        domains = []
        for alpha in os.listdir(dump_dir):
            domains.extend(
                pathjoin(dump_dir, alpha, filename)
                for filename in os.listdir(pathjoin(dump_dir, alpha))
            )
        domain_batchs = blockfy(domains, num_process)
        assert len(domain_batchs) == num_process
        assert sum([len(i) for i in domain_batchs]) == len(domains)
        pool_exec(record_refine, domain_batchs, num_process)

    else:
        logging.warning(f"action must be one of {actions}")
    send_feishu(f"{args} 运行完成, 耗时{(time.time() - start_time)/60}")
