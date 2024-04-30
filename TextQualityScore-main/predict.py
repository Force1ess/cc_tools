import numpy as np
from tqdm import tqdm
import torch, json, os
from types import SimpleNamespace
from torch.utils.data import DataLoader
from utils.data import DocumentDatasetForPredict
from transformers import BertTokenizer, BertConfig
from network.model_architechure_bert_multi_scale import DocumentBertScoringModel
from pathlib import Path
def text_select_with_pred(file, pred_list, text_index_list, 
                          text_segments_list, info_list, config):
    # 写入文件地址
    file_name = Path(file).stem + ".jsonl"
    filtered_file_name = config.output_path + "/" + os.path.basename(file).replace(".json", "_filtered.jsonl")
    dist_file = config.output_path + "/" + file_name

    # 确定要保留的文本
    text_id_list = list(set(text_index_list))
    text_index_array = np.array(text_index_list)
    pred_bool_array = np.array(pred_list) !=10000000

    for text_id in text_id_list:

        indexes = np.where(text_index_array == text_id)[0]
        cur_text_bool = pred_bool_array[indexes]

        # 获取连续被判定为高质量文本对应的 index
        continuous_ranges = np.where(np.diff(np.concatenate((
            [False], cur_text_bool, [False]))))[0].reshape(-1, 2)

        if len(continuous_ranges) == 0:
            text = "".join([text_segments_list[index] for index in indexes])
            score = np.mean(np.array(pred_list)[np.array(indexes)])
            text_dict = {"text_id": text_id, "text": text, "info": info_list[
                    text_id_list.index(text_id)], "score": score}

            # 将不在连续区间内的文本写入文件 (jsonl格式)
            breakpoint()
            with open(filtered_file_name, "a", encoding="utf-8") as f:
                json.dump(text_dict, f, ensure_ascii=False)
                f.write('\n')

            continue

        index_lists = []
        for start, end in continuous_ranges:
            index_list = [index for index in indexes[start:end]]
            index_lists.append(index_list)
            # text = "".join([text_segments_list[index] for index in index_list])
            # score = np.mean(np.array(pred_list)[np.array(index_list)])
            text_dict = {
                "text_id": text_id,
                "text": [text_segments_list[index] for index in index_list],
                "info": info_list[text_id_list.index(text_id)],
                "score": np.array(pred_list)[np.array(index_list)].tolist(),
            }

            # 将筛选的文本写入文件 (jsonl格式)
            with open(dist_file, "a", encoding="utf-8") as f:
                json.dump(text_dict, f, ensure_ascii=False)
                f.write('\n')

        # 将不在连续区间内的文本写入文件 (jsonl格式)
        for index in indexes:
            flag = False
            # 判定是否在连续区间内
            for index_list in index_lists:
                if index in index_list:
                    flag = True
                    break

            if not flag:
                score = pred_list[index]
                text_dict = {"text_id": text_id, "text": text_segments_list[index], "info": info_list[
                    text_id_list.index(text_id)], "score": score}

                with open(filtered_file_name, "a", encoding="utf-8") as f:
                    json.dump(text_dict, f, ensure_ascii=False)
                    f.write('\n')
def predict(file, model: torch.nn.Module, tokenizer: BertTokenizer, 
            config: SimpleNamespace):
    
    # 创建 data_loader
    model_cfg = BertConfig.from_pretrained(config.bert_model_path)
    num_workers = 1
    dataset = DocumentDatasetForPredict(file, tokenizer, 
                                        model_cfg.max_position_embeddings, 
                                        model_cfg.doc_cfg, model_cfg.segment_cfg,
                                        config)
    
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers, 
        shuffle=False)
    
    # 预测
    world_size =1

    model.eval()

    text_index_list = []
    text_segments_list = []
    representation_segment_list = []
    info_list = []
    with torch.no_grad():
        for text_index, text_segments, representation, info in tqdm(
            dataloader, desc=f"Evaluation", leave=False):

            text_index_list.extend(text_index)
            text_segments_list.extend(text_segments)
            representation_segment_list.append(representation)
            info_list.append(info)

            if len(text_index_list) < world_size * config.batch_size:
                continue
            
            # inference
            representation_doc_token = torch.cat(representation_segment_list, dim=0)  
            # (batch_size, seq_len, hidden_size) -> (batch_size, 1, seq_len, hidden_size)
            representation_doc_token = representation_doc_token.unsqueeze(dim=1)
            with torch.cuda.amp.autocast():
                pred = model(representation_doc_token).tolist()
            if not isinstance(pred, list):
                pred = [pred]

            # 基于 text index, text segments, pred 整合保留的文本
            text_select_with_pred(file, pred, text_index_list, 
                                  text_segments_list, info_list, config)

            text_index_list = []
            text_segments_list = []
            representation_segment_list = []
            info_list = []


def predict_setup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型 (数据并行)
    model = DocumentBertScoringModel(config)
    checkpoint = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model = torch.nn.DataParallel(model)

    # dataloader
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)

    return model, tokenizer, device


def process(config):

    model, tokenizer, device = predict_setup(config)

    # 获取 data_path 下各json文件
    data_path = config.data_path


    parquet_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    for file in parquet_files:
        if not file.endswith(".parquet"):
            continue
        print("开始推理， 文件: " + file)
        predict(file, model, tokenizer, config)


if __name__ == "__main__":
    # config
    with open(
        "/home/zhenghao2022/cc_tools/TextQualityScore-main/configs/pred_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)
    cfg = SimpleNamespace(**cfg)

    process(cfg)
