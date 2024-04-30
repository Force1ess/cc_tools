import gc
import gzip
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import handlers
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Any, Callable, Generator, Iterable, List, Optional, Union

import fasttext
import jsonlines
import psutil
import requests
from rich.traceback import install

hostname = socket.gethostname()


NUM_PROCESS = mp.cpu_count() // 3

FILE_FOLDER = os.getcwd()

COLOR_RED = "\x1b[31m"
COLOR_GREEN = "\x1b[32m"
COLOR_YELLOW = "\x1b[33m"
COLOR_RESET = "\x1b[0m"

DATE_TIME = datetime.now()


install()
model = None


class CacheFolderManager:
    def __init__(
        self, folders_to_cache: List[str], cache_directory: str, delay_deletion: int = 3
    ):
        """_summary_

        Args:
            folders_to_cache (List[str]): _description_
            cache_directory (str): _description_
            delay_deletion (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: _description_
        """
        if not folders_to_cache:
            raise ValueError("No folders provided for caching.")

        self.folders_to_cache = folders_to_cache
        self.num_folders = len(folders_to_cache)
        self.progress = 0
        self.cache_directory = cache_directory
        self.current_folder: Optional[str] = None
        self.current_thread: Optional[Thread] = None
        self.delay_deletion = max(delay_deletion, 0)
        self.delayed_folders: List[str] = []
        self.del_threads: Optional[Thread] = []
        self.tot_time = 0

        self._preload_next_folder()

    def _cp_folder_to_cache(self, folder: str):
        abs_folder = Path(folder).expanduser().absolute()
        dest = pathjoin(self.cache_directory, str(abs_folder)[1:])
        self.delayed_folders.append(dest)
        shutil.copytree(folder, dest, dirs_exist_ok=True, symlinks=True)

    def _preload_next_folder(self):
        """预加载下一个文件夹"""
        if self.folders_to_cache:
            self.current_folder = self.folders_to_cache.pop(0)
            # if self.progress<=561 and self.progress!=0:
            #     return
            self.current_thread = Thread(
                target=self._cp_folder_to_cache, args=(self.current_folder,)
            )
            self.current_thread.start()

    def _delete_folder(self, folder: str):
        shutil.rmtree(folder)

    def wait_for_current_folder(self, log=True):
        self.progress += 1
        if self.current_thread:
            s = time.time()
            self.current_thread.join()
            wait_time = time.time() - s
            logging.info(
                f"Cache {self.progress}/{self.num_folders}: {self.current_folder}, wait time:{wait_time:.2f}s"
            )
            self.tot_time += wait_time
            self._preload_next_folder()
        else:
            raise Exception("No ongoing operation")

    def finish_current_and_prepare_next(self):
        """处理延迟删除队列中的文件夹"""
        if len(self.delayed_folders) > self.delay_deletion:
            folder_to_delete = self.delayed_folders.pop(0)
            deletion_thread = Thread(
                target=self._delete_folder, args=(folder_to_delete,)
            )
            self.del_threads.append(deletion_thread)
            deletion_thread.start()

    def finalize(self):
        # Ensure all delayed folders are deleted
        logging.info(f"Cache ToT time: {self.tot_time:.2f}s")
        for folder in self.delayed_folders:
            self._delete_folder(folder)
        self.delayed_folders.clear()
        for thread in self.del_threads:
            thread.join()


def data_save(file_path: str, data: Any, compress: bool = True, append=True):
    if len(data) == 0:
        return
    while os.path.exists(file_path) and not append:
        file_path = file_path + str(random.randint(0, 1000000000))
    mode = "ab" if append else "wb"
    dir_check("/".join(file_path.split("/")[:-1]))
    f_write = open(file_path, mode)
    if compress:
        f_write = gzip.open(file_path, mode)
    writer = jsonlines.Writer(f_write)
    writer.write_all(data)
    f_write.close()
    writer.close()


def lang_detect(text: str) -> str:
    global model
    if model is None:
        model = fasttext.load_model("resource/models/cc_net-language/lid.176.bin")
    labels, scores = model.predict(text[:1000].replace("\n", ""))
    score = min(float(scores[0]), 1.0)
    if score > 0.4:
        return labels[0].replace("__label__", "")
    else:
        return "unknow"


def get_memory_used() -> int:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_used = mem_info.rss / 1024 / 1024 / 1024
    return memory_used


def memory_kill(bound: float = 0.9):
    total = psutil.virtual_memory().total
    avail = psutil.virtual_memory().available
    if (total - avail) / total > bound:
        gc.collect()
        logging.fatal("out of memory")
        send_feishu(" memory over bound " + " ".join(sys.argv))
        exit()


def process_nested_dict(data: dict, prefix=""):
    result = []
    for key, value in data.items():
        if isinstance(value, dict):
            new_prefix = f"{prefix}/{key}" if prefix else key
            nested_result = process_nested_dict(value, new_prefix)
            result.extend(nested_result)
        else:
            full_key = f"{prefix}/{key}" if prefix else key
            result.append((full_key, value))
    return result


def signal_handler(sig, frame):
    print(f"Caught {sig}. Terminating...")
    # ctrl +c
    if sig not in [signal.SIGINT, signal.SIGTERM]:
        send_feishu(sig)
    for process in mp.active_children():
        process.terminate()
    logging.info(locals())
    sys.exit(1)


def sig_register(sig_handler: Callable):
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGHUP, sig_handler)
    signal.signal(signal.SIGQUIT, sig_handler)
    signal.signal(signal.SIGABRT, sig_handler)


def is_file_locked(filename: Union[List, str]):
    "使用绝对路径"
    if isinstance(filename, str):
        filename = [filename]
    for i in filename:
        assert os.path.exists(i), f"{i} does not exist"
    found_files = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            files = proc.open_files()
            for f in files:
                if f.path in filename:
                    found_files.append(f.path)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return [f in found_files for f in filename]


def get_path_name(path_link: str):
    path_name = path_link[
        path_link.find("crawl-data/") + len("crawl-data/") : path_link.find("warc") - 1
    ].replace("/", "-")
    return path_name


def process_exec(
    fn: Callable[..., Any],
    iter_args,
    num_process: int = NUM_PROCESS,
):
    if isinstance(num_process, str):
        num_process = int(num_process)
    elif isinstance(iter_args, list) and num_process > len(iter_args):
        num_process = len(iter_args)
    elif num_process == 0:
        return
    # log_queue, log_listener = mp_log_listener()
    # log_listener.start()
    logging.info(f"Starting pool execution: {num_process} process")
    # args 要用(,)包一下
    process_pool = [mp.Process(target=fn, args=(args,)) for args in iter_args]
    results = []
    try:
        for p in process_pool:
            p.start()
            # sig_register(signal_handler)
        for p in process_pool:
            p.join()
    except Exception as e:
        logging.fatal(e)
        traceback.print_exc()
    # finally:
    # log_listener.stop()
    return results


def pool_exec(
    fn: Callable[..., Any],
    iter_args,
    num_process: int = NUM_PROCESS,
    chunk_size: int = 1,
):
    if isinstance(num_process, str):
        num_process = int(num_process)
    if isinstance(iter_args, list) and num_process > len(iter_args):
        num_process = len(iter_args)
    if num_process == 0:
        return
    # log_queue, log_listener = mp_log_listener()
    # log_listener.start()
    logging.info(f"Starting pool execution: {num_process} process")
    with ProcessPoolExecutor(
        num_process,  # , initializer=mp_logger_init, initargs=[log_queue]
        max_tasks_per_child=128,  # 进程生命周期
    ) as executor:
        # sig_register(signal_handler)
        try:
            results = list(
                executor.map(fn, iter_args, chunksize=chunk_size),
            )
            return results
        except Exception as e:
            logging.fatal(e)
            traceback.print_exc()
        # finally:
        # log_listener.stop()


class args_wrapper:
    def __init__(self, args: dict) -> None:
        self.args = vars(args)

    def __getattr__(self, name: str):
        return self.args.get(name, None)


def mp_logger_init(queue: mp.Queue):
    queue_handler = handlers.QueueHandler(queue)
    queue_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(asctime)s [PID: %(process)d] : %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(queue_handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"process:{os.getpid()} init sucessfully")


def mp_log_listener():
    # 移除全局日志器的处理器
    for h in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(h)

    log_queue = mp.Queue()
    lg_handler = logging.StreamHandler()
    lg_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(asctime)s [PID: %(process)d] : %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    )
    log_listener = handlers.QueueListener(log_queue, lg_handler)
    return log_queue, log_listener


def setup():
    dir_check("cache")
    dir_check("log")
    log_path = FILE_FOLDER + f"/log/common_crawl-{time.strftime('%m-%d-%H-%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s [PID: %(process)d] : %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, "w"),
        ],
    )
    logging.info(f"log saved to {log_path}")


def batchfy(sliceable_iter, batch_size):
    """Divide a list into batches of specified size."""
    return [
        sliceable_iter[i : i + batch_size]
        for i in range(0, len(sliceable_iter), batch_size)
    ]


def blockfy(lst, n):
    """Yield successive n-sized blocks from lst."""
    batchsize = math.ceil(len(lst) / n)
    return batchfy(lst, batchsize)


def load_file_by_line(path: str) -> List[str]:
    with open(path, "r") as fd:
        return [j for j in [i.strip() for i in fd.readlines()] if j]


def list_difference(a: List[str], b: List[str]) -> List[str]:
    diff = []
    for x in a:
        found = False
        for y in b:
            if x == y:
                found = True
                break
        if not found:
            diff.append(x)
    return diff


def file_download(link: str, head: dict = None, max_retries: int = 20):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(
                link, headers=head, verify=False, allow_redirects=True
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"Error: {e}")
            retry_count += 1
            logging.warning(f"Retrying ({retry_count}/{max_retries})...")
            sleep(random.random())
    return response


def dir_check(dir_path: str, overwrite: bool = True):
    if overwrite:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)  # 防止多个进程同时创建
        return dir_path

    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            logging.warninging(f"{dir_path} have been created, but it is not a dir")
        elif len(os.listdir(dir_path)) != 0:
            orig_path = dir_path
            dir_path = orig_path + time.strftime("%m-%d-%H-%M")
            logging.warninging(
                f"{orig_path} have been created, and it is not empty, thus dir_path moved to {dir_path}"
            )
        else:
            return dir_path
        os.makedirs(dir_path, exist_ok=True)
        return dir_path


def exec_command(cmd: str, args: list[str] = None):
    if isinstance(args, list):
        cmd = [cmd] + args
    logging.info(f"exec: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=False, text=True)
    if process.returncode != 0:
        logging.fatal(COLOR_RED + f"Error while exec: {' '.join(cmd)}" + COLOR_RESET)
        return -1
    return 0
