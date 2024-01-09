import pandas as pd
import gzip
import logging
import re
from urllib.parse import urlparse
import trafilatura
from warcio import ArchiveIterator

from tool_funcs import *

qq_regex = r"\b[Qq]{2}.{1,3}\d{6,10}\b"
phone_regex = r"\b\+?\d{1,3}\s?\d{4,14}\b"
mail_regex = r"\b[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}\b"
all_upper = r"^\s*[A-Z]+\s*$"
all_digits = r"^\s*[0-9]+\s*$"
combined_regex = f"{qq_regex}|{phone_regex}|{mail_regex}|{all_upper}|{all_digits}"
compiled_combined_regex = None
black_list =None
def regex_clean(text: str):
    global combined_regex
    if combined_regex is None:
        compiled_combined_regex = re.compile(combined_regex)
    return compiled_combined_regex.sub("", text)
    
#? 需要debug
def warc_process(output_dir: str, file_path: str):
    global black_list
    if black_list is None:
        black_list =  set(load_file_by_line("./blacklist"))
    records = {}
    if os.path.exists(f"{file_path}") and file_path[-3:] == ".gz":
        f = gzip.open(f"{file_path}", "rb")
    elif os.path.exists(f"{file_path.removesuffix('.gz')}"):
        f = open(f"{file_path.removesuffix('.gz')}", "rb")
    else:
        logging.critical(f"file {file_path} does not exist")
        for record in ArchiveIterator(f):
            try:
                if record.rec_type != "response":
                    continue
                uri = record.rec_headers.get_header("WARC-Target-URI")
                domain = urlparse(uri).netloc.removeprefix("www.").removeprefix(" ")
                if not domain or domain[0] in [".", "/", "\\"] or domain in black_list:
                    continue
                datapoint = trafilatura.bare_extraction(record.content_stream().read().decode())
                lang = lang_detect(datapoint["text"])
                if lang is None:
                    continue
                datapoint.update({
                    "date": record.rec_headers.get_header("WARC-Date"),
                    "uri": uri,
                    "language": lang
                })
                domain_record = records.get(domain, [])
                domain_record.append(datapoint)
                records[domain] = domain_record
            except Exception as e:
                logging.warning(COLOR_YELLOW + f"Error while extract: {e}" + COLOR_RESET)
    for domain, record in records.items():
        df = pd.DataFrame(record)
        df.to_parquet(f"{output_dir}/{domain}.parquet", index=False)