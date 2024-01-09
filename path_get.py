import logging
import os

import requests
import trafilatura

from tool_funcs import *


class YearMonth:
    def __init__(self, time_stamp: str):
        if time_stamp == None:
            time_stamp = f"{DATE_TIME.year}/{DATE_TIME.month:02d}"
        year, month = [int(i) for i in time_stamp.split("/")]
        self.year = year
        self.month = month

    def get_timestamp(self) -> str:
        return f"{self.year}/{self.month:02d}"

    def dec(self) -> None:
        self.month -= 1
        if self.month == 0:
            self.month, self.year = 12, self.year - 1

    def iter(self, start_time_stamp: str) -> str:
        if start_time_stamp is None:
            start_time_stamp = "2016/08"
        year, month = [int(i) for i in start_time_stamp.split("/")]
        assert not (year < 2016 or (year == 2016 and month < 8))
        while self.year > year or (self.year == year and self.month >= month):
            yield self.get_timestamp()
            self.dec()

    def get_all_timestamps(self, start_time_stamp) -> List[str]:
        return [i for i in self.iter(start_time_stamp)]


def month_test(ym: YearMonth):
    headers = {
        "Connection": "close",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    }
    # cc-news的发布可能滞后
    response = requests.get(get_warc_path(ym.get_timestamp()), headers=headers)
    logging.info(f"newest monthly update havn't been released yet")
    if response.status_code != 200:
        ym.dec()


def get_warc_path(timestamp: str, type: str = "cc-news") -> str:
    if type == "cc-news":
        return (
            f"http://data.commoncrawl.org/crawl-data/CC-NEWS/{timestamp}/warc.paths.gz"
        )
    elif type == "cc-main":
        return f"http://data.commoncrawl.org/crawl-data/{timestamp}/wet.paths.gz"
    logging.fatal("Error warc path type")


def download_path(path_link: str, max_retries: int = 20) -> str:
    path_name = get_path_name(path_link)
    headers = {
        "Connection": "close",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
    }
    response = file_download(path_link, headers, max_retries)
    if response.status_code == 200:
        with open(f"cache/{path_name}.gz", "wb") as f:
            f.write(response.content)
        exec_command("gunzip", ["-f", f"cache/{path_name}.gz"])
    else:
        logging.fatal(
            COLOR_RED
            + f" DOWNLOAD FAILED: WARC-PATH : {path_link} , reason: {response.reason}"
            + COLOR_RESET
        )
        return path_link


def update_path(path_link: str, old_paths: List[str] = None) -> None:
    path_name = download_path(path_link)
    if old_paths is not None:
        assert isinstance(old_paths, List)
        warc_paths = load_file_by_line(path_name)
        warc_paths = list_difference(warc_paths, old_paths)
        with open(f"./update_path", "w") as f:
            f.write("\n".join(warc_paths))
    logging.info(f"updated path saved to: {dir_PATH}/update_path")


def merge_path(dir_path: str, type: str) -> None:
    if dir_path is None:
        dir_path = "./cache"
    assert os.path.isdir(dir_path)
    assert len(os.listdir(dir_path)) != 0
    files = [i for i in os.listdir(dir_path) and "paths" in i]
    paths = {}
    for i in files:
        paths[i] = [
            "https://data.commoncrawl.org/" + i
            for i in load_file_by_line(dir_path + "/" + i)
        ]

    if type == "cc-main":
        for record_id, record_links in paths.items():
            record = {}
            for i in record_links:
                seg_id = i[65:81].removesuffix("/")
                record[seg_id] = record.get(seg_id, []) + [i]
            paths[record_id.removesuffix("-wet.paths")] = record

    with open(f"./merged_path", "w") as f:
        json.dump(paths, f)

    logging.info(f"merged path saved to: {dir_PATH}/merged_path")


def get_ccmain_index() -> List[str]:
    html = trafilatura.fetch_url("http://data.commoncrawl.org/crawl-data/index.html")
    content = trafilatura.extract(html)
    pos = content.find("CC-MAIN")
    indexs = []
    while pos != -1:
        indexs.append(content[pos : pos + 15])
        pos = content.find("CC-MAIN", pos + 15)
    return indexs
