use arrow::{
    array::{Float32Builder, RecordBatch, StringBuilder, UInt64Builder},
    datatypes::{DataType, Field, Schema},
};
use once_cell::sync::OnceCell;
use parquet::{
    arrow::ArrowWriter,
    file::properties::WriterProperties,
    record::{Row, RowAccessor},
};
use simhash::simhash;
use std::{collections::HashSet, fs::File, io::Write, path::PathBuf, sync::Arc};
pub static OUTPUT_DIR: OnceCell<String> = OnceCell::new();
#[derive(Debug)]
pub struct RawDataPoint {
    pub text: String,
    pub domain: String,
    pub uri: String,
    pub date: String,
    pub language: String,
    pub langscore: f32,
}
impl From<Row> for RawDataPoint {
    fn from(row: Row) -> Self {
        RawDataPoint {
            text: row.get_string(0).unwrap().to_owned(),
            uri: row.get_string(1).unwrap().to_owned(),
            date: row.get_string(2).unwrap().to_owned(),
            language: row.get_string(3).unwrap().to_owned(),
            langscore: row.get_float(4).unwrap().to_owned(),
            domain: row.get_string(5).unwrap().to_owned(),
        }
    }
}
#[derive(Default, Debug)]
pub struct DataPoint {
    pub paras: Vec<String>,
    pub domain: String,
    pub uri: String,
    pub date: String,
    pub language: String,
    pub langscore: f32,
    pub textlen: usize,
    pub orig_textlen: usize,
}
impl From<RawDataPoint> for DataPoint {
    fn from(raw_dp: RawDataPoint) -> Self {
        let paras: Vec<String> = raw_dp.text.split('\n').map(|s| s.to_string()).collect();
        let len: usize = raw_dp.text.len() - paras.len() + 1;
        Self {
            paras: paras,
            uri: raw_dp.uri,
            date: raw_dp.date,
            language: raw_dp.language,
            domain: raw_dp.domain,
            langscore: raw_dp.langscore,
            textlen: len,
            orig_textlen: len,
        }
    }
}
#[derive(Debug, Default)]
pub struct RefinedDataPoint {
    text: String,
    domain: String,
    uri: String,
    date: String,
    language: String,
    lang_score: f32,
    quality_score: f32,
    paralen: f32,
    text_ret_rate: f32,
    textlen: usize,
}
use crate::utils::{check_segments, text_score};

impl From<DataPoint> for RefinedDataPoint {
    fn from(dp: DataPoint) -> Self {
        let num_paras: usize = dp.paras.len();
        let text = dp.paras.join("\n"); // 把所有的段落连接成一个字符串
        Self {
            quality_score: text_score(&text, &dp.language),
            paralen: (text.len() / num_paras) as f32,
            textlen: text.len(),
            text_ret_rate: (text.len() as f32) / dp.orig_textlen as f32,
            text: text,
            domain: dp.domain,
            uri: dp.uri,
            date: dp.date,
            language: dp.language,
            lang_score: dp.langscore,
        }
    }
}
#[derive(Default, Debug)]
pub struct RefineResult {
    pub num_sens: usize,
    pub num_dedup_para: usize,
    pub num_short_paralen: usize,
    pub num_dedup_block: usize,
    pub num_span_repeat: usize,
    pub sens_patterns: Vec<String>,
    pub dedup_patterns: Vec<String>,
    pub orig_len: usize,
    pub domain: String,
    pub language: String,
    avg_lang_score: f32,
    avg_quality_score: f32, //注意，为0代表的是没打分而不是分数真的为0，后续需要处理
    avg_paralen: f32,
    avg_text_ret_rate: f32,
    len: usize,
}
impl RefineResult {
    pub fn finish(
        mut self,
        df: Option<Vec<DataPoint>>,
        sender: std::sync::mpsc::Sender<(Option<(Vec<RefinedDataPoint>, Vec<u64>)>, RefineResult)>,
    ) {
        if let Some(df) = df {
            //unique sens patterns
            self.sens_patterns = self
                .sens_patterns
                .into_iter()
                .collect::<HashSet<String>>()
                .into_iter()
                .collect();
            // convert
            let mut datapoints: Vec<RefinedDataPoint> =
                df.into_iter().map(|dp| dp.into()).collect();

            // discard text contains consec short paras
            datapoints.retain(|dp| check_segments(&dp.text));

            // discard short paragraphs
            let len_u = datapoints.len();
            datapoints.retain(|dp| dp.paralen > 10f32);
            self.len = datapoints.len();
            self.num_short_paralen = len_u - self.len;

            // calc avg paralen
            self.avg_paralen =
                datapoints.iter().map(|dp| dp.paralen).sum::<f32>() / datapoints.len() as f32;

            if (self.len as f32) < (self.orig_len as f32) * 0.4 {
                sender.send((None, self)).unwrap();
                return;
            }
            let hash = datapoints.iter().map(|dp| simhash(&dp.text)).collect();
            self.avg_lang_score =
                datapoints.iter().map(|dp| dp.lang_score).sum::<f32>() / self.len as f32;
            self.avg_quality_score = datapoints.iter().map(|dp| dp.quality_score).sum::<f32>()
                / datapoints
                    .iter()
                    .filter(|dp| dp.quality_score != 0f32)
                    .count() as f32;
            self.avg_text_ret_rate = datapoints.iter().map(|dp| dp.text_ret_rate).sum::<f32>()
                / datapoints.len() as f32;
            match sender.send((Some((datapoints, hash)), self)) {
                Ok(_) => {}
                Err(e) => eprintln!("Failed to send data with datapoints: {}", e),
            }
        } else {
            match sender.send((None, self)) {
                Ok(_) => {}
                Err(e) => eprintln!("Failed to send data without datapoints: {}", e),
            }
        }
    }
}

const MAX_DOMAINS: usize = 10_000; // 100_000;
const MAX_RECORDS: usize = 100_000; //1000_000; // 大约0.4G

pub fn io_manager(
    revicer: std::sync::mpsc::Receiver<(Option<(Vec<RefinedDataPoint>, Vec<u64>)>, RefineResult)>,
) {
    let mut dirty_domains = String::with_capacity(MAX_RECORDS);
    let mut domain_stats = Vec::with_capacity(MAX_DOMAINS);
    let mut hash_list = Vec::with_capacity(MAX_RECORDS);
    let mut num_records = 0;
    let mut datapoints_list = Vec::with_capacity(MAX_RECORDS);
    let mut file_count = 0;
    // let mut dirty_count = 0;
    let mut dirty_file = File::create(format!("{}/dirty_domain.csv", OUTPUT_DIR.get().unwrap(),))
        .expect("Unable to create file");
    let mut writed_records = 0;

    for (data, domain_stat) in revicer {
        if let Some((datapoints, hash)) = data {
            num_records += domain_stat.len;
            datapoints_list.extend(datapoints);
            hash_list.extend(hash);
            domain_stats.push(domain_stat);

            if num_records > MAX_RECORDS {
                save_data(
                    &mut datapoints_list,
                    &mut domain_stats,
                    &mut hash_list,
                    file_count,
                );
                datapoints_list.clear();
                domain_stats.clear();
                hash_list.clear();
                writed_records += num_records;
                num_records = 0;
                file_count += 1;
            }
        } else {
            dirty_domains.push_str(&domain_stat.domain);
            dirty_domains.push_str("\n");
            // dirty_count += 1;
            // if dirty_count == MAX_RECORDS {
            //     dirty_file.write_all(dirty_domains.as_bytes()).unwrap();
            //     dirty_domains.clear();
            //     dirty_count = 0;
            // }
        }
    }
    save_data(
        &mut datapoints_list,
        &mut domain_stats,
        &mut hash_list,
        file_count,
    );
    dirty_file.write_all(dirty_domains.as_bytes()).unwrap();
    writed_records += num_records;
    println!(
        "Process finished: {} records saved in {} files",
        writed_records, file_count
    )
}
fn save_data(
    datapoints: &mut Vec<RefinedDataPoint>,
    domain_stats: &mut Vec<RefineResult>,
    hash_list: &mut Vec<u64>,
    file_count: usize,
) {
    let mut hash_vector = Vec::with_capacity(hash_list.len());
    for (i, hash) in hash_list.iter().enumerate() {
        hash_vector.push((file_count, i, *hash));
    }
    hash_vector.sort_unstable_by_key(|x| x.2);
    let mut hash_file = File::create(format!(
        "{}/hash/{:04}.csv",
        OUTPUT_DIR.get().unwrap(),
        file_count
    ))
    .expect("Unable to create file");
    let _ = hash_file.write_all(
        hash_vector
            .into_iter()
            .map(|x| format!("{},{},{}\n", x.0, x.1, x.2))
            .collect::<String>()
            .as_bytes(),
    );
    save_domain_data(datapoints, file_count);
    save_domain_stats(domain_stats, file_count);
}
fn save_domain_stats(domain_stats: &Vec<RefineResult>, file_count: usize) {
    let schema = Schema::new(vec![
        Field::new("num_sens", DataType::UInt64, true),
        Field::new("sens_patterns", DataType::Utf8, true),
        Field::new("num_dedup_para", DataType::UInt64, true),
        Field::new("num_dedup_block", DataType::UInt64, true),
        Field::new("dedup_patterns", DataType::Utf8, true),
        Field::new("avg_lang_score", DataType::Float32, true),
        Field::new("avg_quality_score", DataType::Float32, true),
        Field::new("avg_paralen", DataType::Float32, true),
        Field::new("avg_text_ret_rate", DataType::Float32, true),
        Field::new("len", DataType::UInt64, true),
        Field::new("orig_len", DataType::UInt64, true),
        Field::new("domain", DataType::Utf8, true),
        Field::new("file_index", DataType::UInt64, true),
        Field::new("language", DataType::Utf8, true),
    ]);

    let schema_ref = Arc::new(schema);

    let domain_stats_len = domain_stats.len();

    let mut num_sens_builder = UInt64Builder::with_capacity(domain_stats_len);
    let mut sens_patterns_builder = StringBuilder::new();
    let mut num_dedup_para_builder = UInt64Builder::with_capacity(domain_stats_len);
    let mut num_dedup_block_builder = UInt64Builder::with_capacity(domain_stats_len);
    let mut dedup_patterns_builder = StringBuilder::new();
    let mut avg_lang_score_builder = Float32Builder::with_capacity(domain_stats_len);
    let mut avg_quality_score_builder = Float32Builder::with_capacity(domain_stats_len);
    let mut avg_paralen_builder = Float32Builder::with_capacity(domain_stats_len);
    let mut avg_text_ret_rate_builder = Float32Builder::with_capacity(domain_stats_len);
    let mut len_builder = UInt64Builder::with_capacity(domain_stats_len);
    let mut orig_len_builder = UInt64Builder::with_capacity(domain_stats_len);
    let mut domain_builder = StringBuilder::new();
    let mut file_index_builder = UInt64Builder::with_capacity(domain_stats_len);
    let mut language_builder = StringBuilder::new();

    for ds in domain_stats {
        num_sens_builder.append_value(ds.num_sens as u64);
        sens_patterns_builder.append_value(&ds.sens_patterns.join(","));
        num_dedup_para_builder.append_value(ds.num_dedup_para as u64);
        num_dedup_block_builder.append_value(ds.num_dedup_block as u64);
        dedup_patterns_builder.append_value(&ds.dedup_patterns.join(","));
        avg_lang_score_builder.append_value(ds.avg_lang_score);
        avg_quality_score_builder.append_value(ds.avg_quality_score);
        avg_paralen_builder.append_value(ds.avg_paralen);
        avg_text_ret_rate_builder.append_value(ds.avg_text_ret_rate);
        len_builder.append_value(ds.len as u64);
        orig_len_builder.append_value(ds.orig_len as u64);
        domain_builder.append_value(&ds.domain);
        file_index_builder.append_value(file_count as u64);
        language_builder.append_value(&ds.language);
    }

    let record_batch = RecordBatch::try_new(
        schema_ref.clone(),
        vec![
            Arc::new(num_sens_builder.finish()),
            Arc::new(sens_patterns_builder.finish()),
            Arc::new(num_dedup_para_builder.finish()),
            Arc::new(num_dedup_block_builder.finish()),
            Arc::new(dedup_patterns_builder.finish()),
            Arc::new(avg_lang_score_builder.finish()),
            Arc::new(avg_quality_score_builder.finish()),
            Arc::new(avg_paralen_builder.finish()),
            Arc::new(avg_text_ret_rate_builder.finish()),
            Arc::new(len_builder.finish()),
            Arc::new(orig_len_builder.finish()),
            Arc::new(domain_builder.finish()),
            Arc::new(file_index_builder.finish()),
            Arc::new(language_builder.finish()),
        ],
    )
    .unwrap();

    let file_path = PathBuf::from(format!(
        "{}/domain_stats/{:04}.parquet",
        OUTPUT_DIR.get().unwrap(),
        file_count
    ));
    let file = File::create(&file_path).unwrap();
    let wprops = WriterProperties::builder().build();

    let mut writer = ArrowWriter::try_new(file, schema_ref, Some(wprops)).unwrap();
    writer.write(&record_batch).unwrap();
    writer.close().unwrap();
}
fn save_domain_data(datapoints: &Vec<RefinedDataPoint>, file_count: usize) {
    let schema = Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("domain", DataType::Utf8, false),
        Field::new("uri", DataType::Utf8, false),
        Field::new("date", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("lang_score", DataType::Float32, false),
        Field::new("quality_score", DataType::Float32, false),
        Field::new("paralen", DataType::Float32, false),
        Field::new("text_ret_rate", DataType::Float32, false),
        Field::new("textlen", DataType::UInt64, false),
    ]);

    let schema_ref = Arc::new(schema);

    let mut text_builder = StringBuilder::new();
    let mut domain_builder = StringBuilder::new();
    let mut uri_builder = StringBuilder::new();
    let mut date_builder = StringBuilder::new();
    let mut language_builder = StringBuilder::new();
    let mut lang_score_builder = Float32Builder::new();
    let mut quality_score_builder = Float32Builder::new();
    let mut paralen_builder = Float32Builder::new();
    let mut text_ret_rate_builder = Float32Builder::new();
    let mut textlen_builder = UInt64Builder::new();

    for dp in datapoints {
        text_builder.append_value(&dp.text);
        domain_builder.append_value(&dp.domain);
        uri_builder.append_value(&dp.uri);
        date_builder.append_value(&dp.date);
        language_builder.append_value(&dp.language);
        lang_score_builder.append_value(dp.lang_score);
        quality_score_builder.append_value(dp.quality_score);
        paralen_builder.append_value(dp.paralen);
        text_ret_rate_builder.append_value(dp.text_ret_rate);
        textlen_builder.append_value(dp.textlen as u64);
    }

    let record_batch = RecordBatch::try_new(
        schema_ref.clone(),
        vec![
            Arc::new(text_builder.finish()),
            Arc::new(domain_builder.finish()),
            Arc::new(uri_builder.finish()),
            Arc::new(date_builder.finish()),
            Arc::new(language_builder.finish()),
            Arc::new(lang_score_builder.finish()),
            Arc::new(quality_score_builder.finish()),
            Arc::new(paralen_builder.finish()),
            Arc::new(text_ret_rate_builder.finish()),
            Arc::new(textlen_builder.finish()),
        ],
    )
    .unwrap();

    let file_path = PathBuf::from(format!(
        "{}/data/{:04}.parquet",
        OUTPUT_DIR.get().unwrap(),
        file_count
    ));
    let file = File::create(&file_path).unwrap();
    let wprops = WriterProperties::builder().build();

    let mut writer = ArrowWriter::try_new(file, schema_ref, Some(wprops)).unwrap();
    writer.write(&record_batch).unwrap();
    writer.close().unwrap();
}
