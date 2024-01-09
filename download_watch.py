import os
import subprocess

from tool_funcs import load_file_by_line

if os.path.exists("cache/download.log"):
    pre_download_set = dict(
        [reversed(i.split()) for i in load_file_by_line("temp/download.log")]
    )
else:
    pre_download_set = {}
command = "du -sh --dereference /data1/data/0.downloaded/cc-main/* | grep T > temp/download.log"
subprocess.run(command, shell=True)
new_download = dict(
    [reversed(i.split()) for i in load_file_by_line("temp/download.log")]
)
for k, v in new_download.items():
    if k not in pre_download_set or v != pre_download_set[k]:
        print(k, v)
