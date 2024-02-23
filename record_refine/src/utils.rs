use jieba_rs::Jieba;
use once_cell::sync::{Lazy, OnceCell};
use regex::Regex;
static SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\p{P}\p{Z}]+").unwrap());
static STOPWORDS: Lazy<Vec<String>> = Lazy::new(|| {
    std::fs::read_to_string("resource/cn_stopwords.txt")
        .expect("Unable to read stopwords file")
        .split('\n')
        .map(|x| x.to_string())
        .collect()
});
pub static SCORE_MODEL: OnceCell<fasttext::FastText> = OnceCell::new();
static JIEBA: Lazy<Jieba> = Lazy::new(|| Jieba::new());
pub fn consecutive_spans_detect(text: &str) -> bool {
    let words = SPLIT_RE.split(text).collect::<Vec<_>>();
    let mut queue = [""; 3];
    if words.len() < 3 {
        return true;
    }
    queue[0] = words[0];
    queue[1] = words[1];
    for word_idx in 2..words.len() - 1 {
        queue[word_idx % 3] = words[word_idx];
        if queue[0] == queue[1] && queue[1] == queue[2] {
            return false;
        }
    }
    true
}
pub fn text_score(text: &str, lang: &str) -> f32 {
    if lang != "zho_Hant" {
        return 0f32;
    }
    let mut segs = JIEBA.cut(text, false);
    segs.retain(|x| x.len() > 1 && !STOPWORDS.contains(&x.to_string()));
    let cut_text = segs.join(" ");
    let preds = SCORE_MODEL.get().unwrap().predict(&cut_text, 1, 0.0).unwrap();
    let pred = preds.first().unwrap();
    if pred.label == "__label__dirty" {
        -pred.prob
    } else {
        pred.prob
    }
}

// 不再使用闭包方式
// pub fn get_text_scorer() -> Box<dyn Fn(&str, &str) -> f32+Send+Sync> {
//     // let mut score_model = fasttext::FastText::new();
//     // score_model
//     //     .load_model("resource/models/oasis-fasttext/model.bin")
//     //     .unwrap();

//     // let stopwords: Vec<String> = std::fs::read_to_string("resource/cn_stopwords.txt")
//     //     .expect("Unable to read stopwords file")
//     //     .split('\n')
//     //     .map(|x| x.to_string())
//     //     .collect();
//     // let jieba = Jieba::new();

//     fn build(text: &str) -> String {
//         let segs = JIEBA.cut(text, false);
//         let segs: Vec<&str> = segs
//             .into_iter()
//             .filter(|x| x.len() > 1 && !STOPWORDS.contains(&x.to_string()))
//             .collect();
//         segs.join(" ")
//     }

//     Box::new(move |text: &str, lang:&str| {
//         if lang!= "zho_Hant" {
//             return 0f32;
//         }
//         let text = build(text);
//         let preds = SCORE_MODEL.predict(text.as_str(), 1, 0.0).unwrap();
//         let pred = preds.first().unwrap();
//         if pred.label == "__label__dirty" {
//             -pred.prob
//         } else {
//             pred.prob
//         }
//     })
// }

pub fn generate_ngrams(s: &str, n: usize, step: usize) -> Vec<&str> {
    if n > s.chars().count() || step > s.chars().count() {
        return vec![];
    }

    let char_indices: Vec<(usize, char)> = s.char_indices().collect();
    let mut ngrams = vec![];

    for i in (0..char_indices.len() - n + 1).step_by(step) {
        let start_index = char_indices[i].0;
        let end_index = if i + n == char_indices.len() {
            s.len()
        } else {
            char_indices[i + n].0
        };
        ngrams.push(&s[start_index..end_index]);
    }
    ngrams
}

pub fn ngram_lcs(pattern: String, text: String, min_n: usize) -> Option<String> {
    let mut pattern = pattern;
    let mut text = text;
    let mut pattern_len = pattern.len();
    if pattern_len > text.len() {
        std::mem::swap(&mut pattern, &mut text);
        pattern_len = pattern.len();
    }
    if pattern_len < min_n {
        return None;
    }
    for (n, step) in [(100, 30), (min_n, 10)] {
        let ngrams = generate_ngrams(pattern.as_str(), n, step);
        for ngram in ngrams {
            if text.contains(ngram) && !ngram.contains("\n") {
                return Some(ngram.to_owned());
            }
        }
    }
    None
}

pub fn ngram_match(ngram: &str, text: &[String], min_len: usize) -> Option<(usize, usize)> {
    for (idx, para) in text.iter().enumerate() {
        if para.len() < min_len {
            continue;
        }
        if para.contains(ngram) {
            #[cfg(debug_assertions)]
            {
                println!("ngram match: {} in para {}", para, ngram);
            }
            return Some((idx, para.chars().count()));
        }
    }
    None
}

mod test {
    #[allow(unused_imports)]
    use super::*;
    #[test]
    fn test_consecutive_spans_detect() {
        let text = "今天心情真不错，今天天气真好。";
        assert_eq!(consecutive_spans_detect(text), true);
        let text = "今天心情真不错!今天心情真不错.今天心情真不错，今天天气真好。\n今天心情真不错，今天天气真好。";
        assert_eq!(consecutive_spans_detect(text), false);
    }
    #[test]
    fn test_score() {
        let text = "友情链接表	今天有什么球赛百丽宫影城华夏天机网90ko金赞娱乐场老虎机游戏在线玩外国中文网站大全767666.com风云足球节目天上人间影院风云足球直播表六合彩官方网站皇冠网上投注澳门金沙彩票投注站申请喜彩网雁荡山棋牌jj斗地主官方淘宝皇冠店铺大全澳门网络博彩欧洲杯官网六合彩免费资料澳门彩票官网香港马会六合彩体球比分网指定开心斗地主单机版下载博彩网信誉娱乐城迅盈网球比分永利国际时时彩导航澳门球盘国际老虎日顶尖高手主论坛线上赌博网站奇迹娱乐城bet007篮球中国长城网博彩网金世豪娱乐时时彩qq群爱博邮件群发系统凤凰全讯网博e百彩讯网虎扑足球中国竞彩网站网上真钱斗地主";
        println!("{}", text_score(text, "zho_Hant"));
        assert!(text_score(text, "zho_Hant") < -0.5f32);
        assert_eq!(text_score(text, "en"), 0f32);
    }
    #[test]
    fn test_jieba() {
        let text = "今天心情真不错";
        let jieba = Jieba::new();
        let segs = jieba.cut(text, false);
        let segs: Vec<&str> = segs.into_iter().filter(|x| x.len() > 1).collect();
        assert_eq!(segs, vec!["今天", "心情", "真不错"]);
    }
    // #[test]
    // fn speed_test() {
    //     let data: Vec<Vec<String>> = vec![vec!["今天心情真不错".to_string(); 100]; 100];

    //     // Version 1: Using rayon's par_iter and for loop
    //     let version1 = std::time::Instant::now();
    //     let _: Vec<Vec<u64>> = data
    //         .par_iter()
    //         .map(|row| row.iter().map(|item| simhash(item)).collect())
    //         .collect();

    //     println!("Version 1: {:?}\n", version1.elapsed());

    //     // Version 2: Using rayon's par_iter to construct DataFrame and then apply
    // let df: DataFrame = data.into_par_iter()
    //     .map(|row| {
    //         let series = Series::new("", row);
    //         let result = series.str().unwrap().apply(&simhash);
    //         result.into_series()
    //     })
    //     .collect();

    // println!("{:?}", df);
    // }
    #[test]
    fn test_generate_ngrams() {
        let s = "今天心情真不错";
        let n = 3;
        let step = 2;
        let expected = vec!["今天心", "心情真", "真不错"];
        let result = generate_ngrams(s, n, step);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ngram_lcs() {
        let pattern = "奥斯卡大奖阿拉斯加lasted了卡简单了卡家的辣口水鸡拉迪说了卡升级到了开始到了卡技术到了卡数据来看就打算".to_string();
        let text = "奥斯卡大奖阿拉斯加lasted了卡简单了卡家的辣口水鸡拉迪说了卡升级到了开始到了卡技术到了卡数".to_string();
        let min_n = 10;
        let expected = Some("奥斯卡大奖阿拉斯加l".to_string());
        let result = ngram_lcs(pattern, text, min_n);
        println!("pattern: {:?}", result,);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ngram_match() {
        let ngram = "心情真不错";
        let text = vec![
            "心情真好".to_string(),
            "今天心情真不错".to_string(),
            "今天天气真好".to_string(),
        ];
        let min_len = 3;
        let expected = Some((1, 7));
        let result = ngram_match(ngram, &text, min_len);
        println!("{:?}", result);
        assert_eq!(result, expected);
    }
}
