use once_cell::sync::{Lazy, OnceCell};
use regex::Regex;
static SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\p{P}\p{Z}]+").unwrap());
pub static LANGDET_MODEL: OnceCell<fasttext::FastText> = OnceCell::new();

pub fn lang_detect(text: &str) -> (String, f32) {
    let preds = LANGDET_MODEL
        .get()
        .unwrap()
        .predict(text.replace("\0", "").as_str(), 1, 0f32)
        .unwrap();
    let first = preds.first();
    match first {
        Some(pred) => (pred.label.replace("__label__", ""), pred.prob),
        None => ("unknown".to_string(), 0.0),
    }
}

pub fn check_segments(input: &str) -> bool {
    let segments = input.split('\n');

    let mut short_segments_count = 0;
    for segment in segments {
        if segment.trim().len() < 20 {
            short_segments_count += 1;
            if short_segments_count >= 5 {
                return false;
            }
        } else {
            short_segments_count = 0;
        }
    }
    true
}

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
            return Some((idx, para.len())); // 不可使用chars.count
        }
    }
    None
}

mod test {

    #[allow(unused_imports)]
    use super::*;
    #[test]
    fn test_lang_detect() {
        let mut ft_model = fasttext::FastText::new();
        ft_model
            .load_model("resource/models/cc_net-language/nllb-model.bin")
            .unwrap();
        LANGDET_MODEL.set(ft_model).unwrap();
        println!("{:?}", lang_detect(r#"Corona Extra – Mexican Lager\nSet delivery address to see local pricingWhy be basic when you could be Extra? Corona Extra is a staple at everything from summer beach parties to everyday occasions. This pilsner style Mexican beer has starts sweet and finishes citrusy, with hints of lemon and ginger. Best served with a fresh wedge of lime, and sipped under an umbrella on the beach.\nMore By Corona\n4.86\n14 Reviews\n- 3 months agoJohn A. - Verified buyer""\n- 4 months agoJohn A. - Verified buyer""\n- 4 months agoAdrian R. - Verified buyer""\n- 10 months agolinda E. - Verified buyer\n- 1 year agoArthur M. - Verified buyer\n- 1 year agoArthur M. - Verified buyer\n- 1 year agoAraceli D. - Verified buyer\n- 1 year agoJose G. - Verified buyer\n- 2 years ago\nGreat tasteAwesome taste all aroundJonathan . - Verified buyer\n- 2 years ago\nAwesome beerGreat tastingJonathan . - Verified buyer\n- 2 years ago\nLove itIts just good specially aith lime tajin or just lime and saltJennifer O. - Verified buyer\n- 2 years ago\nRegular tasteI am not sure what to say it’s the same taste everywhereNour C. - Verified buyer\n- 3 years ago\nYumFast n EasyTyler R. - Verified buyer\n- 3 years ago\nFeels Like The BeachFizzy, light, good with lime + tajinLauren E. - Verified buyer"#));
    }
    #[test]
    fn test_consecutive_spans_detect() {
        let text = "今天心情真不错，今天天气真好。";
        assert_eq!(consecutive_spans_detect(text), true);
        let text = "今天心情真不错!今天心情真不错.今天心情真不错，今天天气真好。\n今天心情真不错，今天天气真好。";
        assert_eq!(consecutive_spans_detect(text), false);
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
        let expected = Some((1, "今天心情真不错".to_string().len()));
        let result = ngram_match(ngram, &text, min_len);
        println!("{:?}", result);
        assert_eq!(result, expected);
    }
}
