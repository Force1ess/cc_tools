use arrow::array::{Float32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::{command, Arg};
use indicatif::ParallelProgressIterator;
use parquet::arrow::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::File;
use url::Url;
use walkdir::WalkDir;
use warc::{WarcHeader, WarcReader};
const INVALID_CHARS: [char; 6] = ['�', 'Ã', '©', '¤', '¶', '¼'];
use once_cell::sync::Lazy;
static LARGE_DOMAINS: Lazy<HashMap<String, u16>> = Lazy::new(|| {
    std::fs::read_to_string("resource/large_domain.txt")
        .expect("Unable to read stopwords file")
        .split('\n')
        .map(|x| {
            let mut line = x.split('\t');
            (
                line.next().unwrap().to_string(),
                (line.next().unwrap().parse::<u32>().unwrap() / 10000) as u16,
            )
        })
        .collect()
});
use zhconv::{zhconv, Variant};
const ZH_VARIANTS: [&str; 2] = ["zho_Hans", "yue_Hant"];
fn garble_check(input: String) -> Option<String> {
    let mut count = 0;
    for ch in input.chars().take(300) {
        if INVALID_CHARS.contains(&ch) {
            count += 1;
            if count > 5 {
                return None;
            }
        }
    }
    Some(input)
}

fn lang_detect(model: &fasttext::FastText, text: &str) -> (String, f32) {
    let preds = model
        .predict(text.replace("\0", "").as_str(), 1, 0f32)
        .unwrap();
    let first = preds.first();
    match first {
        Some(pred) => (pred.label.replace("__label__", ""), pred.prob),
        None => ("unknown".to_string(), 0.0),
    }
}

fn decode_bytes(bytes: &[u8]) -> Option<String> {
    let mut detector = chardetng::EncodingDetector::new();
    detector.feed(bytes, true);

    let (encoding, state) = detector.guess_assess(None, true);
    if !state {
        return None;
    }
    let decoded = encoding.decode_without_bom_handling(bytes).0.to_string();
    garble_check(decoded)
}

async fn write_todisk(
    file_path: PathBuf,
    record_batch: Box<RecordBatch>,
    rc_wprop: WriterProperties,
) {
    let file = File::create(file_path).await.unwrap();

    let mut writer = AsyncArrowWriter::try_new(
        file,
        record_batch.schema(),
        0, //不应该有buffer，因为只写一次
        Some(rc_wprop),
    )
    .unwrap();
    writer.write(&record_batch).await.unwrap();
    writer.close().await.unwrap();
    drop(record_batch);
}
async fn records_saver(
    //receiver: mpsc::Receiver<(Arc<RecordBatch>, String, String)>,
    output_dir: String,
    job_count: u8,
    mut datapoints: HashMap<String, Vec<DataPoint>>, //Vec<DataPoint>,
) {
    let mut limits = libc::rlimit {
        rlim_cur: 0, // 当前（软）限制
        rlim_max: 0, // 最大（硬）限制
    };
    unsafe {
        libc::getrlimit(libc::RLIMIT_NOFILE, &mut limits);
    }
    let fd_limit = (limits.rlim_cur / 3) as usize;

    let schema = Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("uri", DataType::Utf8, false),
        Field::new("date", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("langscore", DataType::Float32, false),
        Field::new("domain", DataType::Utf8, false),
    ]);
    let schema_ref = Arc::new(schema);

    let mut text_builder = StringBuilder::new();
    let mut uri_builder = StringBuilder::new();
    let mut date_builder = StringBuilder::new();
    let mut language_builder = StringBuilder::new();
    let mut domain_builder = StringBuilder::new();
    let mut langscore_builder = Float32Builder::new();

    let w_prop: WriterProperties = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .set_write_batch_size(10240)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
        .set_dictionary_enabled(false)
        .build();

    let mut tasks: Vec<tokio::task::JoinHandle<()>> = vec![];
    let mut file_count = 0;

    for (domain, data_slice) in datapoints.drain() {
        file_count += 1;
        for dp in data_slice {
            text_builder.append_value(dp.text);
            uri_builder.append_value(dp.uri);
            date_builder.append_value(dp.date);
            language_builder.append_value(dp.language);
            langscore_builder.append_value(dp.langscore);
            domain_builder.append_value(dp.domain);
        }

        let record = RecordBatch::try_new(
            schema_ref.clone(),
            vec![
                Arc::new(text_builder.finish()),
                Arc::new(uri_builder.finish()),
                Arc::new(date_builder.finish()),
                Arc::new(language_builder.finish()),
                Arc::new(langscore_builder.finish()),
                Arc::new(domain_builder.finish()),
            ],
        )
        .expect("Failed to create record batch");
        let alpha = match domain.chars().next() {
            Some(c) => c,
            None => {
                continue;
            }
        };
        let dir_path = PathBuf::from_str(output_dir.as_str())
            .unwrap()
            .join(format!("{}/{}", alpha, domain,));

        std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
        let file_path = dir_path.join(format!("{}.parquet", job_count));
        if tasks.len() > fd_limit {
            let _ = tasks.pop().unwrap().await;
        }
        let future = tokio::spawn(write_todisk(file_path, Box::new(record), w_prop.clone()));
        tasks.push(future);
    }

    futures::future::join_all(tasks).await;
    println!("All {} files saved to bucket{}", file_count, job_count);
}
fn get_avail_mem_gb() -> usize {
    match sys_info::mem_info() {
        Ok(mem) => (mem.avail / 1024 / 1024) as usize,
        Err(_) => 0,
    }
}
struct DataPoint {
    text: String,
    domain: String,
    uri: String,
    date: String,
    language: String,
    langscore: f32,
}
// 全大写，全数字，电话号码，邮箱，qq
const ALL_REGEX: &str = r"\n+[A-Z]+\n|\n+[0-9]+\n|\b\+?\d{1,3}\s?\d{4,14}\b|\b[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}\b|\b[Qq]{2}.{1,3}\d{6,10}\b";
// 使用原子组来优化性能
#[cfg(test)]
mod tests {

    use super::*;
    // 测试
    #[test]
    fn garble_workds() {
        let garbledtext = "Ã、�、¤";
        assert_eq!(garble_check(garbledtext.to_string()), None);
    }

    #[test]
    fn regex_works() {
        let regex = Regex::new(ALL_REGEX).unwrap();
        assert!(regex.is_match("qq 2304160042"));
        assert!(regex.is_match("\nSKLDJKLS\n"));
        assert!(regex.is_match("\n1293918273\n"));
        assert!(regex.is_match("\nmyasdasd@mail.com\n"));
        assert!(regex.is_match("\n+86 13800138000\n"));
    }
    #[test]
    fn langid_works() {
        let mut ft_model = fasttext::FastText::new();
        ft_model
            .load_model("resource/models/cc_net-language/lid.176.bin")
            .unwrap();
        println!("{:?}", lang_detect(&ft_model, "くくくくhel卧槽"));
        assert_eq!("zh", lang_detect(&ft_model, "我是中国人").0);
        assert_eq!("zh", lang_detect(&ft_model, "我是中国\0人").0);
    }
    #[test]
    fn utf8_slice_works() {
        // 多种语言
        let text = vec![
            ("你好，我测你🐎", "你好"),
            ("こんにちは、テストさせてください🐎", "こん"),
        ];
        for (text, sliced_text) in text {
            let slice = utf8_slice::till(text, 2);
            assert_eq!(slice, sliced_text);
        }
    }
}

fn wet_process(
    filename: std::path::PathBuf,
    model: Arc<fasttext::FastText>,
    blacklist: Arc<HashSet<String>>,
) -> Vec<DataPoint> {
    let clean_regex = Regex::new(ALL_REGEX).unwrap();
    let mut rng = thread_rng();
    let mut records: Vec<DataPoint> = Vec::new();
    if let Ok(file) = WarcReader::from_path_gzip(filename) {
        for record in file.iter_records() {
            match record {
                Err(err) => {
                    println!("file reading error: {}", err)
                }
                Ok(record) => {
                    if record.warc_type() != &warc::RecordType::Conversion
                        || record.content_length() < 200
                    {
                        continue;
                    }

                    let uri = match record.header(WarcHeader::TargetURI) {
                        Some(uri) => uri.to_string(),
                        None => {
                            continue;
                        }
                    };
                    let domain = match Url::parse(&uri) {
                        Ok(url) => match url.domain() {
                            Some(domain) => {
                                if blacklist.contains(domain) {
                                    continue;
                                }
                                let key = domain.to_string();
                                if LARGE_DOMAINS.contains_key(&key) {
                                    let rand_num =
                                        rng.gen_range(0u16..*LARGE_DOMAINS.get(&key).unwrap());
                                    key + "---" + &rand_num.to_string()
                                } else {
                                    utf8_slice::till(domain, 128).to_string()
                                }
                            }
                            None => {
                                continue;
                            }
                        },
                        Err(_) => {
                            continue;
                        }
                    };
                    let mut body = match decode_bytes(record.body()) {
                        Some(body) => body,
                        None => {
                            continue;
                        }
                    };
                    if body.len() < 200 {
                        continue;
                    }
                    let (mut lang, score) = lang_detect(model.deref(), body.as_str());
                    if lang == "unknown" {
                        continue;
                    }
                    if ZH_VARIANTS.contains(&lang.as_str()) {
                        lang = String::from("zho_Hant");
                        body = zhconv(&body, Variant::ZhCN)
                    }
                    //filter
                    if lang != "zho_Hant" {
                        continue;
                    }
                    // clean
                    body = String::from(clean_regex.replace_all(body.as_str(), ""));

                    let datapoint = DataPoint {
                        text: body,
                        date: record.date().to_string(),
                        uri: uri,
                        domain: domain,
                        language: lang,
                        langscore: score,
                    };
                    records.push(datapoint);
                }
            }
        }
    }
    records
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 获取系统的 CPU 核心数

    let matches = command!()
        .version("1.0")
        .arg(Arg::new("input_dir").required(true))
        .arg(Arg::new("output_dir").required(true))
        .arg(Arg::new("chunksize").required(true))
        .arg(Arg::new("mem_bound").required(true))
        .arg(Arg::new("work_threads").required(true))
        .arg(Arg::new("lid_path").required(true))
        .get_matches();

    let input_dir = matches.get_one::<String>("input_dir").unwrap().to_string();
    let output_dir = matches.get_one::<String>("output_dir").unwrap().to_string();
    let chunksize: usize = matches
        .get_one::<String>("chunksize")
        .unwrap()
        .parse()
        .unwrap();
    let mem_bound: usize = matches
        .get_one::<String>("mem_bound")
        .unwrap()
        .parse()
        .unwrap();
    let lid_path = matches.get_one::<String>("lid_path").unwrap().to_string();

    let num_cores = sys_info::cpu_num().unwrap() as usize;
    let work_threads: usize = matches
        .get_one::<String>("work_threads")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(num_cores >= work_threads, true);
    println!("avalable cpus: {}, used {}", num_cores, work_threads);
    let mut ft_model = fasttext::FastText::new();
    ft_model.load_model(lid_path.as_str()).unwrap();
    let ftm_arc = Arc::new(ft_model);

    let blacklist = std::io::BufReader::new(std::fs::File::open("resource/blacklist")?)
        .lines()
        .filter_map(Result::ok)
        .collect::<HashSet<String>>();
    // 设置 Rayon 线程池的最大线程数为
    let blacklist_arc = Arc::new(blacklist);
    ThreadPoolBuilder::new()
        .num_threads(work_threads)
        .build_global()
        .unwrap();

    let mut tasks: Vec<tokio::task::JoinHandle<()>> = vec![];

    let mut archives: Vec<PathBuf> = WalkDir::new(input_dir.clone())
        .follow_links(true)
        .into_iter()
        .par_bridge()
        .filter_map(|e| e.ok())
        .filter(|entry| {
            entry.file_type().is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("gz")
        })
        .map(|entry| entry.path().to_path_buf())
        .collect();
    archives.sort_unstable();
    println!("Collected {} files in {}", archives.len(), input_dir);
    let mut job_count = 0u8;
    let n_jobs = archives.len() / chunksize;
    for path_slice in archives.chunks(chunksize) {
        let start = Instant::now();
        let grouped_datapoints: HashMap<String, Vec<DataPoint>> = path_slice
            .par_iter()
            .progress()
            .flat_map(|x| wet_process(x.clone(), ftm_arc.clone(), blacklist_arc.clone()))
            .fold(
                || HashMap::new(),
                |mut acc, dp| {
                    acc.entry(dp.domain.clone())
                        .or_insert_with(Vec::new)
                        .push(dp);
                    acc
                },
            )
            .reduce(
                || HashMap::new(),
                |mut acc, hmap| {
                    for (key, value) in hmap {
                        acc.entry(key).or_insert_with(Vec::new).extend(value);
                    }
                    acc
                },
            );

        let free_mem = get_avail_mem_gb();
        println!(
            "Processed {} files, job-{}/{}, free mem: {}, time cost: {} seconds",
            path_slice.len(),
            job_count,
            n_jobs,
            free_mem,
            start.elapsed().as_secs()
        );
        let future = tokio::spawn(records_saver(
            output_dir.clone(),
            job_count,
            grouped_datapoints,
        ));
        tasks.push(future);
        if get_avail_mem_gb() < mem_bound {
            let future = tasks.pop().unwrap();
            println!("Start waiting");
            let start = Instant::now();
            future.await.unwrap();
            println!("Waiting cost {:?}", start.elapsed());
        }
        job_count += 1;
    }
    println!("End processing, wait for saving to disk");
    let start = Instant::now();
    futures::future::join_all(tasks).await;
    println!("Saving finished, cost: {:?}s", start.elapsed());
    Ok(())
}
