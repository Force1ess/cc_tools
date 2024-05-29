from collections import defaultdict
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
import torch, json, os
from types import SimpleNamespace
from torch.utils.data import DataLoader
from utils.data import DocumentDatasetForPredict
from transformers import BertTokenizer, BertConfig
from network.model_architechure_bert_multi_scale import DocumentBertScoringModel
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_select_with_pred(
    pred_list, text_index_list, text_segments_list, dfs: pd.DataFrame, config
):
    highq_dps = []
    lowq_dps = []

    text_id_list = list(set(text_index_list))
    text_index_array = np.array(text_index_list)

    for text_id in text_id_list:
        indexes = np.where(text_index_array == text_id)[0]
        scores = [pred_list[i] for i in indexes if pred_list[i] > config.score_filter]
        filterd_text = "\n".join(
            text_segments_list[i] for i in indexes if pred_list[i] > config.score_filter
        )
        df = dfs[text_id]
        df["orig_text"] = df["text"]
        df["text"] = filterd_text
        df['scores'] = scores
        df['mean_scores'] = np.mean(scores)
        if np.mean(scores) > config.score_threshold:
            highq_dps.append(df)
        else:
            lowq_dps.append(df)
    return highq_dps, lowq_dps


def predict_setup(config):

    # 定义模型 (数据并行)
    model = DocumentBertScoringModel(config)
    checkpoint = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    #model = torch.nn.DataParallel(model)

    # dataloader
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)

    return model, tokenizer, device


def process(config, record_path):

    model, tokenizer, device = predict_setup(config)
    filtered_files = os.path.join(
        record_path, "hash", "hash-values-sorted-filtered.csv"
    )
    filterd = open(filtered_files, "r", encoding="utf-8")
    hash_index = set()
    filterd_list = defaultdict(list)
    for line in filterd.readlines():
        file, linenum, hash_value = line.strip().split(",")
        if hash_value not in hash_index:
            hash_index.add(hash_value)
        else:
            filterd_list[file] = filterd_list.get(file, []) + [int(linenum)]
    data_path = record_path + "/data"
    parquet_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    filterd.close()
    model_cfg = BertConfig.from_pretrained(config.bert_model_path)
    hq_datapoints = []
    lowq_datapoints = []
    text_index_list = []
    representation_segment_list = []
    text_segments_list = []
    model.eval()
    for file in parquet_files:
        datapoints = []
        filter_list = filterd_list.get(Path(file).stem, [])
        if not file.endswith(".parquet"):
            continue
        dataset = DocumentDatasetForPredict(
            file,
            tokenizer,
            model_cfg.max_position_embeddings,
            model_cfg.doc_cfg,
            model_cfg.segment_cfg,
            config,
            filter_list,
        )
        print(f"开始推理， 文件: {file} , 长度: {len(dataset)}")
        dataloader = DataLoader(
            dataset, batch_size=None, num_workers=config.num_workers, shuffle=False
        )
        with torch.inference_mode():
            for dp, text_index, text_segments, representation in tqdm(
                dataloader, desc=f"Evaluation", leave=False
            ):
                text_segments_list.extend(text_segments)
                text_index_list.extend(text_index)
                representation_segment_list.append(representation)
                datapoints.append(dp)

                if (
                    len(text_index_list) < config.batch_size
                    and (len(dataloader) != text_index[0] + 1 and file != parquet_files[-1])
                ):
                    continue

                representation_doc_token = torch.cat(representation_segment_list, dim=0)
                representation_doc_token = representation_doc_token.unsqueeze(dim=1)
                representation_doc_token = representation_doc_token.to(device)
                with torch.cuda.amp.autocast():
                    pred = model(representation_doc_token).tolist()
                if not isinstance(pred, list):
                    pred = [pred]
                hqdfs, lowqdfs = text_select_with_pred(
                    pred,
                    text_index_list,
                    text_segments_list,
                    datapoints,
                    config,
                )
                hq_datapoints.extend(hqdfs)
                lowq_datapoints.extend(lowqdfs)
                text_index_list = []
                representation_segment_list = []
                text_segments_list = []
    pd.concat(hq_datapoints).to_json(
        f"{record_path}/scored-data/cn_hq.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    pd.concat(lowq_datapoints).to_json(
        f"{record_path}/scored-data/cn_lowq.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )


if __name__ == "__main__":
    # config
    with open(
        "/home/zhenghao2022/cc_tools/pred_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)
    cfg = SimpleNamespace(**cfg)

    dump_dir = sys.argv[1]
    if os.path.exists(dump_dir + "/scored-data"):
        import shutil
        print('scored-data exsisted, deleting')
        shutil.rmtree(dump_dir + "/scored-data")

    os.makedirs(dump_dir + "/scored-data")
    process(cfg, dump_dir)
