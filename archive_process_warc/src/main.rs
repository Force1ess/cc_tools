use arrow::array::{Float32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::{command, Arg};
use indicatif::ParallelProgressIterator;
use lazy_static::lazy_static;
use parquet::arrow::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::SerializedFileReader;
use parquet::record::{Row, RowAccessor};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::File;
use walkdir::WalkDir;
const INVALID_CHARS: [char; 6] = ['�', 'Ã', '©', '¤', '¶', '¼'];
use zhconv::{zhconv, Variant};

async fn write_todisk(
    file_path: PathBuf,
    record_batch: Box<RecordBatch>,
    rc_wprop: WriterProperties,
) {
    let file = File::create(file_path).await.unwrap();

    let mut writer =
        AsyncArrowWriter::try_new(file, record_batch.schema(), Some(rc_wprop)).unwrap();
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
        Field::new("info", DataType::Utf8, false),
    ]);
    let schema_ref = Arc::new(schema);

    let mut text_builder = StringBuilder::new();
    let mut uri_builder = StringBuilder::new();
    let mut date_builder = StringBuilder::new();
    let mut language_builder = StringBuilder::new();
    let mut domain_builder = StringBuilder::new();
    let mut langscore_builder = Float32Builder::new();
    let mut info_builder = StringBuilder::new();

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
            info_builder.append_value(dp.info);
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
                Arc::new(info_builder.finish()),
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
struct DataPoint {
    text: String,
    domain: String,
    uri: String,
    date: String,
    language: String,
    langscore: f32,
    info: String,
}
impl TryFrom<Row> for DataPoint {
    type Error = Box<dyn std::error::Error>; // 你可以根据需要使用具体的错误类型

    fn try_from(row: Row) -> Result<Self, Self::Error> {
        Ok(DataPoint {
            text: row.get_string(0)?.to_owned(),
            uri: row.get_string(1)?.to_owned(),
            domain: row.get_string(2)?.to_owned(),
            info: row
                .get_string(4)
                .map(|s| s.to_owned()) // 将 Option<&String> 转换为 Option<String>
                .unwrap_or_default(), // 在这里提供默认值,
            language: row.get_string(5)?.to_owned(),
            langscore: row.get_double(6)? as f32,
            // 存在很多trafilatura解析不出date的情况
            date: row
                .get_string(3)
                .map(|s| s.to_owned()) // 将 Option<&String> 转换为 Option<String>
                .unwrap_or_default(), // 在这里提供默认值
        })
    }
}
// 全大写，全数字，电话号码，邮箱，qq
const ALL_REGEX: &str = r"\n+[A-Z]+\n|\n+[0-9]+\n|\b\+?\d{1,3}\s?\d{4,14}\b|\b[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}\b|\b[Qq]{2}.{1,3}\d{6,10}\b";
lazy_static! {
    /// This is an example for using doc comment attributes
    static ref CLEAN_REGEX:Regex = Regex::new(ALL_REGEX).unwrap();
}

fn process_file(file: std::path::PathBuf) -> Option<Vec<DataPoint>> {
    let parquet_file = std::fs::File::open(&file).unwrap();
    let reader = SerializedFileReader::new(parquet_file).ok()?;
    Some(
        reader
            .into_iter()
            .filter_map(|row_result| process_row(row_result, &file))
            .collect(),
    )
}

// 处理单个行记录，转换为 DataPoint
fn process_row(
    row_result: Result<parquet::record::Row, parquet::errors::ParquetError>,
    file: &std::path::PathBuf,
) -> Option<DataPoint> {
    match row_result {
        Ok(row) => match DataPoint::try_from(row) {
            Ok(dp) => validate_and_clean(dp),
            Err(e) => {
                eprintln!(
                    "Failed to convert in file {:?}: row to DataPoint: {}",
                    file, e
                );
                None
            }
        },
        Err(e) => {
            eprintln!("Failed in file {:?}: read row {}", file, e);
            None
        }
    }
}

fn validate_and_clean(mut dp: DataPoint) -> Option<DataPoint> {
    if dp.text.len() < 200 {
        return None;
    }
    let mut count = 0;
    for ch in dp.text.chars().take(300) {
        if INVALID_CHARS.contains(&ch) {
            count += 1;
            if count > 5 {
                return None;
            }
        }
    }
    if dp.language == "zh" {
        dp.text = zhconv(&dp.text, Variant::ZhCN);
    }
    dp.text = String::from(CLEAN_REGEX.replace_all(dp.text.as_str(), ""));
    Some(dp)
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 获取系统的 CPU 核心数

    let matches = command!()
        .version("1.0")
        .arg(Arg::new("input_dir").required(true))
        .arg(Arg::new("output_dir").required(true))
        .arg(Arg::new("work_threads").required(true))
        .get_matches();

    let input_dir = matches.get_one::<String>("input_dir").unwrap().to_string();
    let output_dir = matches.get_one::<String>("output_dir").unwrap().to_string();
    let num_cores = sys_info::cpu_num().unwrap() as usize;
    let work_threads: usize = matches
        .get_one::<String>("work_threads")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(num_cores >= work_threads, true);
    println!("avalable cpus: {}, used {}", num_cores, work_threads);
    // 设置 Rayon 线程池的最大线程数为
    ThreadPoolBuilder::new()
        .num_threads(work_threads)
        .build_global()
        .unwrap();

    let mut tasks: Vec<tokio::task::JoinHandle<()>> = vec![];
    let archives: Vec<PathBuf> = WalkDir::new(input_dir.clone())
        .follow_links(true)
        .into_iter()
        .par_bridge()
        .filter_map(|e| e.ok())
        .into_par_iter()
        .filter(|entry| {
            entry.file_type().is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("parquet")
        })
        .map(|entry| entry.path().to_path_buf())
        .collect();
    println!("Collected {} files in {}", archives.len(), input_dir);
    let records: Vec<_> = archives
        .into_par_iter()
        .progress()
        .flat_map(process_file)
        .flatten()
        .collect();
    let mut grouped_datapoints: HashMap<String, Vec<DataPoint>> = HashMap::new();
    records.into_iter().for_each(|dp| {
        let domain = dp.domain.clone();
        if grouped_datapoints.contains_key(&domain) {
            grouped_datapoints.get_mut(&domain).unwrap().push(dp);
        } else {
            grouped_datapoints.insert(domain, vec![dp]);
        }
    });
    let future = tokio::spawn(records_saver(output_dir.clone(), 0, grouped_datapoints));
    tasks.push(future);
    println!("End processing, wait for saving to disk");
    let start = Instant::now();
    futures::future::join_all(tasks).await;
    println!("Saving finished, cost: {:?}s", start.elapsed());
    Ok(())
}
