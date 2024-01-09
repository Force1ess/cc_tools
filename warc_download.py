from tool_funcs import exec_command

headers = "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'"

# 用是否是一个list来判断

# 经测试aria2c的稳定性更好

# def wget_download(output_dir: Path, link: str):
#     logging.info(str(os.getpid()) + ": Downloading " + link)
#     retry = 0
#     filename = get_path_name(link[40:]) + ".gz"
#     while retry < 20:
#         if (
#             exec_command(
#                 "wget",
#                 [
#                     "-4",
#                     "-c",
#                     "--header",
#                     "'Connection: close'",
#                     "--output-document",
#                     str(output_dir / filename),
#                     "-U",
#                     headers,
#                     link,
#                 ],
#             )
#             == -1
#         ):
#             exec_command("rm", [str(output_dir / filename)])
#             sleep(randint(1, 30))
#             retry += 1
#         if os.path.exists(output_dir / filename):
#             # exec_command("gunzip", ["-f","--keep", str(output_dir/filename)])
#             # exec_command("mv", [str(output_dir/filename), f"archive/{filename}"])
#             return
#     logging.fatal(f"download failed {filename}:{link}")
#     return


def aria2c_download(output_dir: str, num_process: int):
    if num_process is None:
        num_process = 8
    if isinstance(num_process, str):
        num_process = int(num_process)
    exec_command(
        "aria2c",
        [
            "--auto-file-renaming=false",
            "-i",
            "temp/download.tmp",
            "-d",
            output_dir,
            "--max-tries=100",
            f"-j{num_process}",
            "-x8",
        ],
    )
