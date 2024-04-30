import pandas as pd

dump_dir = "/data_ssd/zhenghao2022/refined/CC-MAIN-2018-17_cn"
df = pd.read_parquet(dump_dir)
list_of_dicts = df.to_dict(orient="records")
list_of_dicts.sort(key=lambda x:x['quality_score'], reverse=True)
with open("output.txt", "w") as file:
    for record in list_of_dicts[:1000]:
        for key, value in record.items():
            file.write(f"{key}: {value}\n")

        file.write("\n***\n***\n")
breakpoint()
domain_statsdf = pd.read_parquet(dump_dir)
list_of_dicts = domain_statsdf.to_dict(orient="records")
with open("domain_stats.txt", "w") as file:
    for record in list_of_dicts:
        for key, value in record.items():
            file.write(f"{key}: {value}\n")

        file.write("\n")

dirty_df = domain_statsdf[
    (df["len"] < 0.1)
    | (df["avg_quality_score"] < 0)
    | (df["avg_paralen"] < 20)
    | (df["len"] / df["orig_len"] < 0.3)
]
from tool_funcs import load_file_by_line

blacklist = load_file_by_line("./resource/blacklist")
+[domain for domain in dirty_df.domain]
blacklist = set(blacklist)
text = "\n".join(sorted(list(blacklist)))+'\n'
breakpoint()
blacklist_fd = open("resource/blacklist", "w")
blacklist_fd.write(text)
blacklist_fd.close()
# sort resource/blacklist | uniq > new_blacklis
