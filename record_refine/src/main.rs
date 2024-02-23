use clap::{command, Arg};
// TODO SPANREGEX
//use std::env;
// fn main() {
//     env::set_var("ARROW_MEMORY_CAPACITY", "100000000"); // 设置内存限制为 100MB
// }
use parquet::file::serialized_reader::SerializedFileReader;
use rand::Rng;
use rayon::{prelude::*, ThreadPoolBuilder};
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, read_dir, File};
use std::path::PathBuf;

mod trie;
use trie::Trie;
mod datamodel;
use datamodel::*;
mod utils;
use indicatif::ParallelProgressIterator;
use std::sync::mpsc::{self, Sender};
use utils::{consecutive_spans_detect, ngram_lcs, ngram_match};

use crate::utils::SCORE_MODEL;

const TRIE_TRY: usize = 100;

// 每个进程单独开一个发送的好了

fn domain_refine(
    mut df: Vec<RawDataPoint>,
    domain: String,
    min_textlen: usize,
    min_blocklen: usize,
    trie: Option<&Trie>,
    sender: Sender<(Option<(Vec<RefinedDataPoint>, Vec<u64>)>, RefineResult)>,
) {
    let mut del_idxs = HashSet::new();
    let mut sens_count = 0;
    let orig_len_u = df.len();
    let orig_len = orig_len_u as f32;

    df.retain(|dp| consecutive_spans_detect(&dp.text));

    let mut result = RefineResult::default();
    result.domain = domain;
    result.orig_len = orig_len_u;
    result.num_span_repeat = orig_len_u - df.len();

    if let Some(trie_tree) = trie {
        for (i, row) in df.iter().enumerate() {
            if i == TRIE_TRY {
                break;
            }
            let sens = trie_tree.find_match(&row.text);
            if sens.len() > 2 {
                del_idxs.insert(i);
                sens_count += 1;
            }
            result.sens_patterns.extend(sens);
        }

        if sens_count as f32 / min(TRIE_TRY, df.len()) as f32 > 0.1 {
            result.num_sens += df.len();
            result.finish(None, sender);
            return;
        }
        if sens_count != 0 {
            for (i, row) in df.iter().skip(TRIE_TRY).enumerate() {
                if i == TRIE_TRY {
                    break;
                }

                if !trie_tree.query(&row.text, 2) {
                    del_idxs.insert(i);
                    sens_count += 1;
                }
            }
            result.num_sens = del_idxs.len();
        }
    }

    let repeat_time: u16 = max(20, min(500, df.len() / 5)) as u16;
    let mut rng = rand::thread_rng();
    let mut datapoints: Vec<DataPoint> = df.into_iter().map(|dp| dp.into()).collect();

    let mut fail_count: u16 = 0;
    // para dedup
    while fail_count < repeat_time {
        if !del_idxs.is_empty() {
            let mut i = 0;
            datapoints.retain(|_| {
                let keep = !del_idxs.contains(&i);
                i += 1;
                keep
            });
            del_idxs.clear();
        }
        fail_count += 1;

        if datapoints.len() as f32 / orig_len < 0.6 {
            result.num_dedup_para += datapoints.len();
            result.finish(None, sender);
            return;
        }
        if datapoints.len() < 7 {
            break;
        }
        let idx_a = rng.gen_range(0..datapoints.len());
        let idx_b = rng.gen_range(0..datapoints.len());

        if idx_a == idx_b {
            continue;
        }

        let a = &datapoints[idx_a];
        let b = &datapoints[idx_b];
        let paras_a = &a.paras;
        let paras_b = &b.paras;
        let mut small_sus_pattern: HashMap<String, i32> = HashMap::new();
        for i in paras_a {
            if paras_b.contains(i) && i.len() > 2 {
                small_sus_pattern.insert(i.clone(), 0); // 小段完全相同
            }
        }

        if small_sus_pattern.is_empty() {
            continue;
        }
        #[cfg(debug_assertions)]
        {
            println!("small patterns: {:?}", small_sus_pattern);
        }

        for _ in 0..repeat_time / 2 {
            let idx_c: usize = rng.gen_range(0..datapoints.len());
            if idx_c == idx_a || idx_c == idx_b || del_idxs.contains(&idx_c) {
                continue;
            }
            for (pattern, count) in small_sus_pattern.iter_mut() {
                let paras_c = &datapoints[idx_c].paras; // 防止生命周期报错
                if *count < 3 && paras_c.contains(pattern) {
                    *count += 1;
                }
                if *count != 3 {
                    continue;
                }
                *count += 1;
                fail_count = 0;
                for (content_id, content) in datapoints.iter_mut().enumerate() {
                    if del_idxs.contains(&content_id) {
                        continue;
                    }
                    // 不能直接迭代remove，因为迭代器失效
                    content.paras.retain(|para| {
                        if para == pattern {
                            content.textlen -= pattern.len();
                            false
                        } else {
                            true
                        }
                    });
                    if content.textlen < min_textlen
                    //|| (content.textlen as f32) < (content.orig_textlen as f32) * 0.5
                    {
                        del_idxs.insert(content_id);
                    }
                }
                result.dedup_patterns.push(pattern.clone());
            }
        }
        result.num_dedup_para += del_idxs.len();
    }
    #[cfg(debug_assertions)]
    {
        println!("Founded small patterns: {:?}", result.dedup_patterns);
    }
    //block dedup
    fail_count = 0;
    while fail_count < repeat_time {
        fail_count += 1;
        if !del_idxs.is_empty() {
            let mut i = 0;
            // 原地修改，不需要额外内存
            datapoints.retain(|_| {
                let keep = !del_idxs.contains(&i);
                i += 1;
                keep
            });
            del_idxs.clear();
        }
        if datapoints.len() as f32 / orig_len < 0.6 {
            result.num_dedup_block += datapoints.len();
            result.finish(None, sender);
            return;
        }
        if datapoints.len() < 7 {
            break;
        }
        let idx_a = rng.gen_range(0..datapoints.len());
        let idx_b = rng.gen_range(0..datapoints.len());
        if idx_a == idx_b {
            continue;
        }
        let a = &datapoints[idx_a];
        let b = &datapoints[idx_b];
        if let Some(lcs) = ngram_lcs(a.paras.join("\n"), b.paras.join("\n"), min_blocklen) {
            let mut pattern_count = 0;

            for _ in 0..repeat_time {
                let idx_c = rng.gen_range(0..datapoints.len());
                if idx_c == idx_a || idx_c == idx_b || del_idxs.contains(&idx_c) {
                    continue;
                }
                if let Some(_) = ngram_match(&lcs, &datapoints[idx_c].paras, min_blocklen) {
                    pattern_count += 1;
                }
                if pattern_count != 3 {
                    continue;
                }
                fail_count = 0;
                #[cfg(debug_assertions)]
                {
                    println!("Founded pattern: {}", lcs)
                }
                for (content_id, content) in datapoints.iter_mut().enumerate() {
                    if let Some((para_idx, paralen)) =
                        ngram_match(&lcs, &content.paras, min_blocklen)
                    {
                        content.paras.remove(para_idx);
                        content.textlen -= paralen;
                        if content.textlen < min_textlen
                        // || (content.textlen as f32) < (content.orig_textlen as f32) * 0.5
                        {
                            del_idxs.insert(content_id);
                        }
                    }
                }
                result.dedup_patterns.push(lcs);
                break;
            }
        }
        result.num_dedup_block += del_idxs.len();
    }
    #[cfg(debug_assertions)]
    {
        println!(
            "Founded all patterns: {:?} on {} rows",
            result.dedup_patterns, orig_len_u
        );
    }
    result.finish(Some(datapoints), sender);
}
fn record_refine(
    domain_folder: PathBuf,
    domain: &str,
    trie_forest: &HashMap<String, Trie>,
    sender: Sender<(Option<(Vec<RefinedDataPoint>, Vec<u64>)>, RefineResult)>,
) {
    let mut lang_datapoints: HashMap<String, Vec<RawDataPoint>> = HashMap::with_capacity(32);
    let files: Vec<PathBuf> = read_dir(domain_folder)
        .unwrap()
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|entry| entry.path())
        .filter(|entry| -> bool { entry.is_file() && entry.extension().unwrap() == "parquet" })
        .map(|entry| entry.to_path_buf())
        .collect();

    // 不能使用par，否则会导致数据竞争
    let rows = files
        .into_iter()
        .flat_map(|file| {
            SerializedFileReader::new(File::open(file).unwrap())
                .unwrap()
                .into_iter()
        })
        .collect::<Vec<_>>();
    for row in rows {
        //使用新的数据
        let dp: RawDataPoint = row.unwrap().into();
        if !lang_datapoints.contains_key(&dp.language) {
            lang_datapoints.insert(dp.language.clone(), Vec::new());
        }
        lang_datapoints.get_mut(&dp.language).unwrap().push(dp);
    }
    lang_datapoints
        .into_iter()
        .par_bridge()
        .for_each(|(lang, datapoints)| {
            let (mut min_textlen, mut min_blocklen) = (270, 50);
            if lang == "zho_Hant" {
                (min_textlen, min_blocklen) = (150, 30);
            }
            domain_refine(
                datapoints,
                domain.to_owned(),
                min_textlen,
                min_blocklen,
                trie_forest.get(&lang),
                sender.clone(),
            )
        });
}

fn main() {
    let matches = command!()
        .version("1.0")
        .arg(Arg::new("input_dir").required(true))
        .arg(Arg::new("output_dir").required(true))
        .arg(Arg::new("badwords_dir").required(true))
        .arg(Arg::new("qmdodel_path").required(true))
        .arg(Arg::new("work_threads").required(true))
        .get_matches();

    let input_dir = matches.get_one::<String>("input_dir").unwrap().to_string();
    let output_dir = matches.get_one::<String>("output_dir").unwrap().to_string();
    create_dir_all(output_dir.clone()+"/hash").unwrap();
    create_dir_all(output_dir.clone()+"/domain_stats").unwrap();
    assert_eq!(
        OUTPUT_DIR.set(output_dir),
        Ok(())
    );

    let qmdodel_path = matches
        .get_one::<String>("qmdodel_path")
        .unwrap()
        .to_string();
    let mut model = fasttext::FastText::new();
    model
        .load_model(&qmdodel_path)
        .unwrap();
    assert_eq!(
        SCORE_MODEL.set(model).unwrap(),
        ()
    );

    let badwords_dir = matches
        .get_one::<String>("badwords_dir")
        .unwrap()
        .to_string();

    let num_cores = sys_info::cpu_num().unwrap() as usize;
    let work_threads: usize = matches
        .get_one::<String>("work_threads")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(num_cores >= work_threads, true);
    println!("avalable cpus: {}, used {}", num_cores, work_threads);

    ThreadPoolBuilder::new()
        .num_threads(work_threads)
        .build_global()
        .unwrap();
    let mut trie_forest: HashMap<String, Trie> = HashMap::with_capacity(32);
    read_dir(badwords_dir)
        .unwrap()
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|entry| entry.path())
        .filter(|entry| entry.is_file() && entry.extension().unwrap() == "txt")
        .for_each(|entry| {
            let filename = entry.file_stem().unwrap().to_str().unwrap();
            trie_forest.insert(
                filename.to_string(),
                Trie::from_vocab(entry.to_str().unwrap()).unwrap(),
            );
        });
    println!("Loaded trie forest: {} languages", trie_forest.len());
    let domain_folders: Vec<(String, PathBuf)> = read_dir(input_dir)
        .unwrap()
        .into_iter()
        .par_bridge()
        .filter_map(|e| e.ok())
        .map(|entry| entry.path())
        .filter(|entry| entry.is_dir())
        .flat_map(|entry| {
            let alpha = entry.to_path_buf();
            read_dir(alpha)
                .unwrap()
                .into_iter()
                .par_bridge()
                .filter_map(|e| e.ok())
                .map(|entry| entry.path())
                .filter(|entry| entry.is_dir())
                .map(|entry| {
                    (
                        entry.file_name().unwrap().to_str().unwrap().to_owned(),
                        entry,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    println!("Loaded preprocessed data: {} domains", domain_folders.len());
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || io_manager(receiver));
    // foreach无返回值不需要收集
    domain_folders
        .into_par_iter()
        .progress()
        .for_each(|(domain, domain_folder)| {
            record_refine(domain_folder, domain.as_str(), &trie_forest, sender.clone())
        });
}
