pub const MAX_WORD_BYTES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Whitespace,
    Alnum,
    Other,
}

#[inline]
pub fn is_alnum(ch: char) -> bool {
    ch.is_alphanumeric()
}

pub fn char_class(text: &[char], index: usize) -> Class {
    let ch = text[index];
    if ch.is_whitespace() {
        return Class::Whitespace;
    }
    if is_alnum(ch) {
        return Class::Alnum;
    }
    if ch == '-'
        && index > 0
        && index + 1 < text.len()
        && is_alnum(text[index - 1])
        && is_alnum(text[index + 1])
    {
        return Class::Alnum;
    }
    Class::Other
}

pub fn split_digit_run(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() <= 3 || !chars.iter().all(|ch| ch.is_ascii_digit()) {
        return vec![word.to_string()];
    }
    let head = chars.len() % 3;
    let mut groups: Vec<String> = Vec::new();
    if head > 0 {
        groups.push(chars[..head].iter().collect());
    }
    let mut start = head;
    while start < chars.len() {
        groups.push(chars[start..start + 3].iter().collect());
        start += 3;
    }
    groups
}

pub fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut words: Vec<String> = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let group = char_class(&chars, index);
        let mut end = index + 1;
        while end < chars.len() && char_class(&chars, end) == group {
            end += 1;
        }
        let run: String = chars[index..end].iter().collect();
        match group {
            Class::Alnum => words.extend(split_digit_run(&run)),
            _ => words.push(run),
        }
        index = end;
    }
    words
}

pub fn word_chunks(word: &str, max_bytes: usize) -> Vec<Vec<u8>> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    for ch in word.chars() {
        let length = ch.len_utf8();
        if !current.is_empty() && current.len() + length > max_bytes {
            chunks.push(std::mem::take(&mut current));
        }
        let mut buffer = [0u8; 4];
        current.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}
