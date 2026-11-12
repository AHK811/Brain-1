use crate::pre_tokenize::pre_tokenize;
use crate::vocab::Vocab;
use anyhow::{ensure, Result};
use std::collections::HashMap;

pub const PAD_LABEL: &[u8] = b"<pad>";
pub const EOS_LABEL: &[u8] = b"<eos>";

#[derive(Debug, Clone)]
pub struct TrainStats {
    pub word_types: usize,
    pub corpus_word_instances: usize,
    pub merges: usize,
    pub vocab_size: usize,
    pub stopped_early: bool,
    pub largest_pair_count: usize,
}

pub fn train_bpe(texts: &[String], target_vocab: usize) -> Result<(Vocab, TrainStats)> {
    ensure!(
        target_vocab > 258,
        "vocab must exceed 256 byte tokens plus two specials"
    );
    let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut instances = 0usize;
    for text in texts {
        for word in pre_tokenize(text) {
            *counts.entry(word.into_bytes()).or_insert(0) += 1;
            instances += 1;
        }
    }
    let mut types: Vec<(Vec<u8>, usize)> = counts.into_iter().collect();
    types.sort_by(|a, b| a.0.cmp(&b.0));
    let mut syms: Vec<Vec<u32>> = types
        .iter()
        .map(|(bytes, _)| bytes.iter().map(|b| *b as u32).collect())
        .collect();
    let word_counts: Vec<usize> = types.iter().map(|(_, count)| *count).collect();

    let mut tokens: Vec<Vec<u8>> = (0u16..256).map(|value| vec![value as u8]).collect();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut pair_total: HashMap<(u32, u32), usize> = HashMap::new();
    let mut pair_types: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();

    for (index, symbols) in syms.iter().enumerate() {
        for pair in adjacent(symbols) {
            *pair_total.entry(pair).or_insert(0) += word_counts[index];
            let list = pair_types.entry(pair).or_default();
            match list.iter().position(|(existing, _)| *existing == index) {
                Some(position) => list[position].1 += 1,
                None => list.push((index, 1)),
            }
        }
    }

    let max_merges = target_vocab - 258;
    let mut largest = 0usize;
    while merges.len() < max_merges {
        let mut best: Option<((u32, u32), usize, usize)> = None;
        for (pair, total) in pair_total.iter() {
            if *total <= 1 {
                continue;
            }
            let first = pair_types
                .get(pair)
                .and_then(|list| {
                    list.iter()
                        .filter(|(_, occurrences)| *occurrences > 0)
                        .map(|(index, _)| *index)
                        .min()
                })
                .unwrap_or(usize::MAX);
            let candidate = (*pair, *total, first);
            match best {
                None => best = Some(candidate),
                Some((best_pair, best_total, best_first)) => {
                    let better = (*total > best_total)
                        || (*total == best_total && first < best_first)
                        || (*total == best_total && first == best_first && *pair < best_pair);
                    if better {
                        best = Some(candidate);
                    }
                }
            }
        }
        let Some(((left, right), total, _)) = best else {
            break;
        };
        largest = largest.max(total);
        let new_id = (256 + merges.len()) as u32;
        let mut combined = tokens[left as usize].clone();
        combined.extend_from_slice(&tokens[right as usize].clone());
        tokens.push(combined);
        merges.push((left, right));

        let affected: Vec<(usize, usize)> = pair_types.remove(&(left, right)).unwrap_or_default();
        for (index, _occurrences) in affected {
            let before = adjacent(&syms[index]);
            for pair in before {
                if let Some(list) = pair_types.get_mut(&pair) {
                    if let Some(position) = list.iter().position(|(existing, _)| *existing == index)
                    {
                        list[position].1 -= 1;
                        if list[position].1 == 0 {
                            list.remove(position);
                        }
                    }
                }
                if let Some(entry) = pair_total.get_mut(&pair) {
                    *entry = entry.saturating_sub(word_counts[index]);
                    if *entry == 0 {
                        pair_total.remove(&pair);
                    }
                }
            }
            merge_pair(&mut syms[index], left, right, new_id);
            let after = adjacent(&syms[index]);
            for pair in after {
                *pair_total.entry(pair).or_insert(0) += word_counts[index];
                let list = pair_types.entry(pair).or_default();
                match list.iter().position(|(existing, _)| *existing == index) {
                    Some(position) => list[position].1 += 1,
                    None => list.push((index, 1)),
                }
            }
        }
    }

    tokens.push(PAD_LABEL.to_vec());
    tokens.push(EOS_LABEL.to_vec());
    let stopped_early = merges.len() < max_merges;
    let merge_count = merges.len();
    let vocab = Vocab::new(tokens, merges)?;
    let stats = TrainStats {
        word_types: types.len(),
        corpus_word_instances: instances,
        merges: merge_count,
        vocab_size: vocab.vocab_size(),
        stopped_early,
        largest_pair_count: largest,
    };
    Ok((vocab, stats))
}

fn adjacent(symbols: &[u32]) -> Vec<(u32, u32)> {
    symbols
        .windows(2)
        .map(|window| (window[0], window[1]))
        .collect()
}

fn merge_pair(symbols: &mut Vec<u32>, left: u32, right: u32, replacement: u32) {
    let mut out: Vec<u32> = Vec::with_capacity(symbols.len());
    let mut index = 0usize;
    while index < symbols.len() {
        if index + 1 < symbols.len() && symbols[index] == left && symbols[index + 1] == right {
            out.push(replacement);
            index += 2;
        } else {
            out.push(symbols[index]);
            index += 1;
        }
    }
    std::mem::swap(symbols, &mut out);
}
