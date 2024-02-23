use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

#[derive(Debug)]
pub struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_word: bool,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            is_word: false,
        }
    }
}

#[derive(Debug)]
pub struct Trie {
    root: TrieNode,
    pub source:String,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            root: TrieNode::new(),
            source: String::new(),
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut current = &mut self.root;
        for letter in word.chars() {
            current = current.children.entry(letter).or_insert(TrieNode::new());
        }
        current.is_word = true;
    }

    pub fn from_vocab(vocabs_path: &str) -> Result<Self, std::io::Error> {
        let mut trie = Trie::new();
        trie.source = vocabs_path.to_string();
        let file = File::open(vocabs_path)?;
        let reader = io::BufReader::new(file);
        for line in reader.lines() {
            let word = line?;
            trie.insert(&word);
        }
        return Ok(trie);
    }

    pub fn find_match(&self, word: &str) -> Vec<String> {
        let mut res = vec![];
        let chars: Vec<char> = word.chars().collect();
        for i in 0..chars.len() {
            let c = chars[i];
            let mut node = self.root.children.get(&c);
            if node.is_some() {
                for j in i + 1..chars.len() {
                    let _c = chars[j];
                    node = node.unwrap().children.get(&_c);
                    if node.is_some() {
                        if node.unwrap().is_word {
                            res.push(chars[i..=j].iter().collect::<String>());
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        res
    }

    pub fn query<S: AsRef<str>>(&self, text: S, threshold: usize) -> bool {
        let lower_string = text.as_ref().to_lowercase();
        let mut times = 0;
        let mut node = &self.root;
        for char in lower_string.chars() {
            match node.children.get(&char) {
                Some(next_node) => {
                    node = next_node;
                    if node.is_word {
                        times += 1;
                        if times > threshold {
                            return false;
                        }
                    }
                }
                None => {
                    node = &self.root; // reset to root node
                }
            }
        }
        true
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie() {
        let mut trie = Trie::new();
        trie.insert("你好");
        trie.insert("world");
        trie.insert("rust");
        trie.insert("rustacean");

        // Test query
        assert_eq!(trie.query("你好", 1), true);
        assert_eq!(trie.query("world", 1), true);

        // Test exceeding threshold
        assert_eq!(trie.query("你好！RUSTACEAN", 1), false);
        let trie = Trie::from_vocab("./resource/badwords/zho_Hant.txt").unwrap();
        assert_eq!(trie.query("操你妈 你妈逼，你很幽默吗 法轮功", 2), false);
    }
}
