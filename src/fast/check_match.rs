use fd_bs58::constants::{
    BINARY_SZ_32, ENC_TABLE_32, INTERMEDIATE_SZ_32, R1_DIV, RAW58_SZ_32,
};
use num_bigint::BigUint;
use num_traits::One;

pub const MAX_PATTERN_LEN: usize = 44;
pub const MAX_PATTERNS: usize = 64;

const ALPHABET: &[u8; 58] =
    b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const ALPHABET_CI: &[u8; 58] =
    b"123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

const fn identity_lut() -> [u8; 58] {
    let mut lut = [0u8; 58];
    let mut i = 0;
    while i < 58 {
        lut[i] = i as u8;
        i += 1;
    }
    lut
}

const fn build_match_lut(alphabet: &[u8; 58]) -> [u8; 58] {
    let mut lut = [0u8; 58];
    let mut i = 0;
    while i < 58 {
        lut[i] = i as u8;
        let mut j = 0;
        while j < i {
            if alphabet[j] == alphabet[i] {
                lut[i] = j as u8;
                break;
            }
            j += 1;
        }
        i += 1;
    }
    lut
}

static MATCH_LUT_CS: [u8; 58] = identity_lut();
static MATCH_LUT_CI: [u8; 58] = build_match_lut(ALPHABET_CI);

#[inline]
pub fn match_lut(case_insensitive: bool) -> &'static [u8; 58] {
    if case_insensitive {
        &MATCH_LUT_CI
    } else {
        &MATCH_LUT_CS
    }
}

/// Longest suffix tail checked by numeric value alone. \(58^8\) fits
/// in a `u64`; longer suffixes are filtered on their last 8 chars, a
/// necessary condition for the full suffix to match.
const SUFFIX_FILTER_LEN: usize = 8;
/// Proven bound, not a tuning knob: every canonical digit has at
/// most 2 raw preimages under either `match_lut` (CS: 1), and the
/// filter tail is at most 8 digits, so at most \(2^8 = 256\) residues
/// can ever be built. Anything larger means a bug; fall back then.
const SUFFIX_FILTER_MAX_RESIDUES: usize = 256;

/// Accepted values of a pubkey modulo \(58^{\text{len}}\).
///
/// Proof of no false negatives: if the base58 encoding ends with a
/// suffix, its length is at least the suffix length and its low
/// `len` base58 digits are exactly that suffix (or its last `len`
/// chars). Leading `'1'`s are digit 0 on the high side and cannot
/// affect those low digits; an encoding shorter than the suffix
/// cannot end with it. Hence a matching pubkey's numeric residue is
/// always in `residues`, and rejection is safe. Encodings shorter
/// than the suffix also cannot match, so rejecting them is safe too.
/// Passers are still verified by the full encode in `check_match_32`.
struct SuffixFilter {
    len: u8,
    modulus: u64,
    /// Sorted ascending (built most-significant digit first).
    residues: Vec<u64>,
    /// Bit `d` set iff raw digit `d` can be the suffix's last digit.
    last_digit_mask: u64,
}

struct PatternIdx {
    prefix_ranges: Option<Vec<([u8; 32], [u8; 32])>>,
    prefix_idx: [u8; MAX_PATTERN_LEN],
    prefix_len: u8,
    suffix_idx: [u8; MAX_PATTERN_LEN],
    suffix_len: u8,
    suffix_filter: Option<SuffixFilter>,
}

/// Build the residue set for the last `SUFFIX_FILTER_LEN` suffix
/// digits. Each canonical index expands to every raw digit mapping
/// to it under `match_lut`, i.e. exactly the digits `check_match_32`
/// accepts at that position, so CI and CS share this construction.
/// Returns `None` (fall back to encode) on unmappable pattern chars
/// or if the expansion grows past `SUFFIX_FILTER_MAX_RESIDUES`.
fn build_suffix_filter(
    suffix_idx: &[u8; MAX_PATTERN_LEN],
    suffix_len: u8,
    match_lut: &[u8; 58],
) -> Option<SuffixFilter> {
    if suffix_len == 0 {
        return None;
    }
    let len = (suffix_len as usize).min(SUFFIX_FILTER_LEN);
    let start = suffix_len as usize - len;
    let mut residues = vec![0u64];
    for &canonical in &suffix_idx[start..suffix_len as usize] {
        let mut next = Vec::new();
        for &r in &residues {
            for d in 0..58u8 {
                if match_lut[d as usize] == canonical {
                    next.push(r * 58 + u64::from(d));
                }
            }
        }
        if next.is_empty() || next.len() > SUFFIX_FILTER_MAX_RESIDUES {
            return None;
        }
        residues = next;
    }
    let mut last_digit_mask = 0u64;
    let last = suffix_idx[suffix_len as usize - 1];
    for d in 0..58u8 {
        if match_lut[d as usize] == last {
            last_digit_mask |= 1u64 << d;
        }
    }
    Some(SuffixFilter {
        len: len as u8,
        modulus: 58u64.pow(len as u32),
        residues,
        last_digit_mask,
    })
}

/// `N mod 58` in four `u64` chunks: cheap first gate (rejects all
/// but ~1/58, or 2/58 CI, before the weighted residue below).
#[inline]
fn mod58_gate(bytes: &[u8; 32]) -> u64 {
    const POW64_MOD58: u64 = ((1u128 << 64) % 58) as u64;
    let mut r = 0u64;
    for chunk in bytes.chunks_exact(8) {
        let c = u64::from_be_bytes(chunk.try_into().unwrap());
        r = (r * POW64_MOD58 + c % 58) % 58;
    }
    r
}

/// `N mod modulus` via per-byte weights `256^(31-i) mod modulus`.
/// Accumulator bound: `255 * 32 * (modulus-1) < 2^60` for
/// `modulus <= 58^8 < 2^47`, so one final `u64` modulo suffices.
#[inline]
fn residue_weighted(
    bytes: &[u8; 32],
    modulus: u64,
    weights: &[u64; 32],
) -> u64 {
    debug_assert!(modulus > 1);
    let mut s = 0u64;
    for i in 0..32 {
        s += u64::from(bytes[i]) * weights[i];
    }
    s % modulus
}

fn suffix_weights(modulus: u64) -> [u64; 32] {
    let mut w = [0u64; 32];
    if modulus <= 1 {
        return w;
    }
    let m = u128::from(modulus);
    let mut cur = 1u64;
    for i in (0..32).rev() {
        w[i] = cur;
        cur = ((u128::from(cur) * 256) % m) as u64;
    }
    w
}

/// One or more prefix/suffix pairs. A pubkey matches if it matches any pair.
///
/// GPU blob layout (little-endian), shared with CUDA / OpenCL:
/// ```text
/// u32 n
/// u8  max_prefix_len
/// u8  max_suffix_len
/// n times: u8 plen, plen bytes, u8 slen, slen bytes
/// ```
/// Bytes are canonical raw_base58 indices, not ASCII.
pub struct MatchTargets {
    patterns: Vec<PatternIdx>,
    match_lut: &'static [u8; 58],
    /// \(58^{\text{max filter len}}\); 1 when no pattern has a filter.
    /// Residues for smaller moduli divide out of a residue mod this.
    suffix_filter_max_modulus: u64,
    suffix_weights: [u64; 32],
}

impl MatchTargets {
    pub fn new(
        patterns: &[(&str, &str)],
        case_insensitive: bool,
    ) -> Self {
        assert!(
            patterns.len() <= MAX_PATTERNS,
            "at most {MAX_PATTERNS} patterns"
        );
        let alphabet = if case_insensitive {
            ALPHABET_CI
        } else {
            ALPHABET
        };
        let lut = match_lut(case_insensitive);

        let mut out = Vec::with_capacity(patterns.len());
        for &(prefix, suffix) in patterns {
            debug_assert!(prefix.len() <= MAX_PATTERN_LEN);
            debug_assert!(suffix.len() <= MAX_PATTERN_LEN);
            let mut prefix_idx = [0u8; MAX_PATTERN_LEN];
            for (i, &b) in prefix.as_bytes().iter().enumerate() {
                prefix_idx[i] = char_to_canonical(b, alphabet, lut);
            }
            let mut suffix_idx = [0u8; MAX_PATTERN_LEN];
            for (i, &b) in suffix.as_bytes().iter().enumerate() {
                suffix_idx[i] = char_to_canonical(b, alphabet, lut);
            }
            let suffix_len = suffix.len() as u8;
            out.push(PatternIdx {
                prefix_ranges: prefix_ranges(prefix, case_insensitive),
                prefix_idx,
                prefix_len: prefix.len() as u8,
                suffix_filter: build_suffix_filter(
                    &suffix_idx,
                    suffix_len,
                    lut,
                ),
                suffix_idx,
                suffix_len,
            });
        }

        let suffix_filter_max_modulus = out
            .iter()
            .filter_map(|p| p.suffix_filter.as_ref())
            .map(|f| f.modulus)
            .max()
            .unwrap_or(1);
        Self {
            patterns: out,
            match_lut: lut,
            suffix_weights: suffix_weights(suffix_filter_max_modulus),
            suffix_filter_max_modulus,
        }
    }

    /// Bit `i` is set when pattern `i` matches. Empty set → no match.
    #[cfg(test)]
    pub fn match_mask(&self, bytes: &[u8; 32]) -> u64 {
        self.match_mask_active(bytes, u64::MAX)
    }

    /// Bit `i` is set when pattern `i` matches and that kind is in `active`.
    #[inline]
    pub fn match_mask_active(
        &self,
        bytes: &[u8; 32],
        active: u64,
    ) -> u64 {
        let mut mask = 0u64;
        let mut encode_active = 0u64;
        // Residue mod `suffix_filter_max_modulus`, computed on the
        // first suffix-bearing candidate that has a filter; residues
        // for smaller filter moduli divide out of it (`58^a | 58^b`).
        let mut suffix_residue: Option<u64> = None;
        let mut last_digit: Option<u64> = None;
        for (pi, p) in self.patterns.iter().enumerate() {
            if active & (1u64 << pi) == 0 {
                continue;
            }
            if let Some(ranges) = &p.prefix_ranges {
                if !ranges
                    .iter()
                    .any(|(lo, hi)| bytes >= lo && bytes <= hi)
                {
                    continue;
                }
                if p.suffix_len == 0 {
                    mask |= 1u64 << pi;
                    continue;
                }
            }
            if p.suffix_len > 0 {
                if let Some(filter) = &p.suffix_filter {
                    if !self.suffix_filter_passes(
                        filter,
                        bytes,
                        &mut suffix_residue,
                        &mut last_digit,
                    ) {
                        continue;
                    }
                }
            }
            encode_active |= 1u64 << pi;
        }
        if encode_active != 0 {
            mask |= check_match_32(
                bytes,
                &self.patterns,
                self.match_lut,
                encode_active,
            );
        }
        mask
    }

    /// Suffix residue gates, kept out of the hot prefix loop's
    /// frame (`#[cold]`): called only for suffix-bearing patterns
    /// that survived any prefix range check.
    #[cold]
    fn suffix_filter_passes(
        &self,
        filter: &SuffixFilter,
        bytes: &[u8; 32],
        suffix_residue: &mut Option<u64>,
        last_digit: &mut Option<u64>,
    ) -> bool {
        debug_assert!(
            (1..=SUFFIX_FILTER_LEN as u8).contains(&filter.len)
        );
        // Gate 1: last base58 digit (`N mod 58`).
        let digit =
            *last_digit.get_or_insert_with(|| mod58_gate(bytes));
        if filter.last_digit_mask >> digit & 1 == 0 {
            return false;
        }
        // Gate 2 (len > 1): full residue; len == 1 is already
        // decided exactly by gate 1's mask.
        if filter.len > 1 {
            let big = *suffix_residue.get_or_insert_with(|| {
                residue_weighted(
                    bytes,
                    self.suffix_filter_max_modulus,
                    &self.suffix_weights,
                )
            });
            let residue = big % filter.modulus;
            if filter
                .residues
                .binary_search(&residue)
                .is_err()
            {
                return false;
            }
        }
        true
    }

    /// Canonical-index blob consumed by `gpu_grind_init` / `gpu_keypair_init`.
    #[cfg(any(test, feature = "gpu"))]
    pub fn gpu_blob(&self) -> Vec<u8> {
        let max_prefix_len = self
            .patterns
            .iter()
            .map(|p| p.prefix_len)
            .max()
            .unwrap_or(0);
        let max_suffix_len = self
            .patterns
            .iter()
            .map(|p| p.suffix_len)
            .max()
            .unwrap_or(0);
        let mut v = Vec::with_capacity(
            6 + self
                .patterns
                .iter()
                .map(|p| {
                    2 + p.prefix_len as usize + p.suffix_len as usize
                })
                .sum::<usize>(),
        );
        v.extend_from_slice(
            &(self.patterns.len() as u32).to_le_bytes(),
        );
        v.push(max_prefix_len);
        v.push(max_suffix_len);
        for p in &self.patterns {
            v.push(p.prefix_len);
            v.extend_from_slice(&p.prefix_idx[..p.prefix_len as usize]);
            v.push(p.suffix_len);
            v.extend_from_slice(&p.suffix_idx[..p.suffix_len as usize]);
        }
        v
    }
}

#[inline]
fn char_to_canonical(
    c: u8,
    alphabet: &[u8; 58],
    match_lut: &[u8; 58],
) -> u8 {
    for (k, &ch) in alphabet.iter().enumerate() {
        if ch == c {
            return match_lut[k];
        }
    }
    255
}

// Exact prefixes can be checked as byte ranges without base58
// conversion. Keep the encode matcher for leading '1' and
// case-insensitive searches.
fn prefix_ranges(
    prefix: &str,
    ci: bool,
) -> Option<Vec<([u8; 32], [u8; 32])>> {
    if ci || prefix.is_empty() || prefix.starts_with('1') {
        return None;
    }
    let mut value = BigUint::from(0u8);
    for b in prefix.bytes() {
        value *= 58u8;
        value += ALPHABET.iter().position(|&c| c == b)?;
    }
    let minimum = BigUint::one() << 248usize;
    let maximum = (BigUint::one() << 256usize) - 1u8;
    let to_bytes = |n: &BigUint| {
        let v = n.to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - v.len()..].copy_from_slice(&v);
        out
    };
    let mut ranges = Vec::new();
    let mut scale = BigUint::one();
    for _ in prefix.len()..=MAX_PATTERN_LEN {
        let lo = (&value * &scale).max(minimum.clone());
        let hi = ((&value + 1u8) * &scale - 1u8).min(maximum.clone());
        if lo <= hi {
            ranges.push((to_bytes(&lo), to_bytes(&hi)));
        }
        scale *= 58u8;
    }
    Some(ranges)
}

#[inline]
fn emit_limb(
    intermediate: &[u64; INTERMEDIATE_SZ_32],
    raw_base58: &mut [u8; RAW58_SZ_32],
    limb: usize,
) {
    let v = intermediate[limb] as u32;
    raw_base58[5 * limb + 4] = (v % 58) as u8;
    raw_base58[5 * limb + 3] = ((v / 58) % 58) as u8;
    raw_base58[5 * limb + 2] = ((v / 3364) % 58) as u8;
    raw_base58[5 * limb + 1] = ((v / 195_112) % 58) as u8;
    raw_base58[5 * limb] = (v / 11_316_496) as u8;
}

#[inline]
fn ensure_limb(
    intermediate: &[u64; INTERMEDIATE_SZ_32],
    raw_base58: &mut [u8; RAW58_SZ_32],
    limbs_done: &mut usize,
    need: usize,
) {
    while *limbs_done <= need {
        emit_limb(intermediate, raw_base58, *limbs_done);
        *limbs_done += 1;
    }
}

#[inline]
fn check_match_32(
    bytes: &[u8; 32],
    patterns: &[PatternIdx],
    match_lut: &[u8; 58],
    active: u64,
) -> u64 {
    let mut in_leading_0s = 0usize;
    while in_leading_0s < 32 && bytes[in_leading_0s] == 0 {
        in_leading_0s += 1;
    }

    let mut binary = [0u32; BINARY_SZ_32];
    for i in 0..BINARY_SZ_32 {
        let o = i * 4;
        binary[i] = u32::from_be_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
        ]);
    }

    let mut intermediate = [0u64; INTERMEDIATE_SZ_32];
    for i in 0..BINARY_SZ_32 {
        for j in 0..INTERMEDIATE_SZ_32 - 1 {
            intermediate[j + 1] +=
                u64::from(binary[i]) * ENC_TABLE_32[i][j];
        }
    }

    for i in (1..INTERMEDIATE_SZ_32).rev() {
        intermediate[i - 1] += intermediate[i] / R1_DIV;
        intermediate[i] %= R1_DIV;
    }

    let mut raw_base58 = [0u8; RAW58_SZ_32];
    let mut limbs_done = 0usize;

    ensure_limb(&intermediate, &mut raw_base58, &mut limbs_done, 0);

    let mut raw_leading_0s = 0usize;
    while raw_leading_0s < RAW58_SZ_32 {
        ensure_limb(
            &intermediate,
            &mut raw_base58,
            &mut limbs_done,
            raw_leading_0s / 5,
        );
        if raw_base58[raw_leading_0s] != 0 {
            break;
        }
        raw_leading_0s += 1;
    }

    let skip = raw_leading_0s - in_leading_0s;
    let encoded_length = RAW58_SZ_32 - skip;

    if patterns.is_empty() {
        return 1;
    }

    // Same per-char early-reject walk as the original single-pattern
    // matcher; extra patterns re-use already-emitted limbs.
    let mut mask = 0u64;
    for (pi, p) in patterns.iter().enumerate() {
        if active & (1u64 << pi) == 0 {
            continue;
        }
        let mut ok = true;
        for i in 0..p.prefix_len as usize {
            let rb_idx = skip + i;
            ensure_limb(
                &intermediate,
                &mut raw_base58,
                &mut limbs_done,
                rb_idx / 5,
            );
            if match_lut[raw_base58[rb_idx] as usize] != p.prefix_idx[i]
            {
                ok = false;
                break;
            }
        }
        if ok && p.suffix_len > 0 {
            let suffix_len = p.suffix_len as usize;
            let tail_start = skip + encoded_length - suffix_len;
            let last_limb = (skip + encoded_length - 1) / 5;
            ensure_limb(
                &intermediate,
                &mut raw_base58,
                &mut limbs_done,
                last_limb,
            );
            for i in 0..suffix_len {
                if match_lut[raw_base58[tail_start + i] as usize]
                    != p.suffix_idx[i]
                {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            mask |= 1u64 << pi;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::pubkey_of_seed;
    use sha2::{Digest, Sha512};

    fn encode_matches(
        bytes: &[u8; 32],
        prefix: &str,
        suffix: &str,
        ci: bool,
    ) -> bool {
        let s = fd_bs58::encode_32(*bytes);
        if ci {
            let lc: String =
                s.chars()
                    .map(|c| {
                        if c == 'L' {
                            c
                        } else {
                            c.to_ascii_lowercase()
                        }
                    })
                    .collect();
            (prefix.is_empty() || lc.starts_with(prefix))
                && (suffix.is_empty() || lc.ends_with(suffix))
        } else {
            (prefix.is_empty() || s.starts_with(prefix))
                && (suffix.is_empty() || s.ends_with(suffix))
        }
    }

    #[test]
    fn agrees_with_full_encode_random() {
        let target = MatchTargets::new(&[("zz", "LM")], false);
        for i in 0u32..5000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.match_mask(&bytes) != 0,
                encode_matches(&bytes, "zz", "LM", false),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_with_full_encode_ci() {
        let target = MatchTargets::new(&[("mithriL", "")], true);
        for i in 0u32..2000 {
            let h: [u8; 64] =
                Sha512::digest((i ^ 0xdeadbeef).to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.match_mask(&bytes) != 0,
                encode_matches(&bytes, "mithriL", "", true),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_on_known_keys() {
        let keys = [
            "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr",
            "11111111111111111111111111111111",
            "1zfbgASTPZHoQ5DhqS5f2bnJk88rxMi137DmZowDztN",
        ];
        for key in keys {
            let bytes = fd_bs58::decode_32(key).unwrap();
            let prefix = &key[..key.len().min(3)];
            let suffix = &key[key.len().saturating_sub(2)..];
            let target = MatchTargets::new(&[(prefix, suffix)], false);
            assert!(target.match_mask(&bytes) != 0, "{key}");
        }
    }

    #[test]
    fn or_of_two_patterns() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let miss =
            MatchTargets::new(&[("zzz", ""), ("aaa", "qq")], false);
        assert_eq!(miss.match_mask(&bytes), 0);
        let hit =
            MatchTargets::new(&[("zzz", ""), ("XkC", "KAr")], false);
        assert_eq!(hit.match_mask(&bytes), 0b10);
        let pref_only =
            MatchTargets::new(&[("nope", "nope"), ("XkC", "")], false);
        assert_eq!(pref_only.match_mask(&bytes), 0b10);
    }

    #[test]
    fn prefix_ranges_skip_leading_one_and_ci() {
        assert!(prefix_ranges("Harmonic", false).is_some());
        assert!(prefix_ranges("1abc", false).is_none());
        assert!(prefix_ranges("Harmonic", true).is_none());
        assert!(prefix_ranges("", false).is_none());
    }

    #[test]
    fn prefix_only_range_agrees_with_encode() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let hit = MatchTargets::new(&[("XkC", "")], false);
        assert_eq!(hit.match_mask(&bytes), 0b1);
        let miss = MatchTargets::new(&[("XkD", "")], false);
        assert_eq!(miss.match_mask(&bytes), 0);
    }

    fn sampled_pubkeys(count: u32, salt: u32) -> Vec<[u8; 32]> {
        (0..count)
            .map(|i| {
                let h: [u8; 64] =
                    Sha512::digest((i ^ salt).to_le_bytes()).into();
                let seed: [u8; 32] = h[..32].try_into().unwrap();
                pubkey_of_seed(&seed)
            })
            .collect()
    }

    fn boundary_keys() -> Vec<[u8; 32]> {
        let mut one = [0u8; 32];
        one[31] = 1;
        let mut fifty_seven = [0u8; 32];
        fifty_seven[31] = 57;
        let mut two_digit = [0u8; 32];
        two_digit[30] = 1;
        let mut leading_zero_ff = [0xFFu8; 32];
        leading_zero_ff[0] = 0;
        leading_zero_ff[1] = 0;
        vec![
            [0u8; 32],
            [0xFFu8; 32],
            [1u8; 32],
            one,
            fifty_seven,
            two_digit,
            leading_zero_ff,
            fd_bs58::decode_32(
                "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr",
            )
            .unwrap(),
            fd_bs58::decode_32(
                "1zfbgASTPZHoQ5DhqS5f2bnJk88rxMi137DmZowDztN",
            )
            .unwrap(),
        ]
    }

    fn assert_agrees(
        patterns: &[(&str, &str)],
        ci: bool,
        keys: &[[u8; 32]],
    ) {
        let target = MatchTargets::new(patterns, ci);
        for bytes in keys {
            let mut expected = 0u64;
            for (i, &(prefix, suffix)) in patterns.iter().enumerate() {
                if encode_matches(bytes, prefix, suffix, ci) {
                    expected |= 1u64 << i;
                }
            }
            assert_eq!(
                target.match_mask(bytes),
                expected,
                "ci={ci} patterns={patterns:?} bytes={bytes:?}"
            );
        }
    }

    #[test]
    fn suffix_filter_shape() {
        let cs = MatchTargets::new(&[("", "KAr"), ("abc", "")], false);
        let f = cs.patterns[0]
            .suffix_filter
            .as_ref()
            .unwrap();
        assert_eq!(f.len, 3);
        assert_eq!(f.modulus, 58u64.pow(3));
        assert_eq!(f.residues.len(), 1);
        assert!(cs.patterns[1].suffix_filter.is_none());

        // CI: `L` has one raw variant, `m` expands to `M`/`m`, so
        // the caller-canonical suffix "Lm" yields two residues.
        let ci = MatchTargets::new(&[("", "Lm")], true);
        let f = ci.patterns[0]
            .suffix_filter
            .as_ref()
            .unwrap();
        assert_eq!(f.len, 2);
        assert_eq!(f.residues.len(), 2);
        assert!(f
            .residues
            .windows(2)
            .all(|w| w[0] < w[1]));

        // Longer suffixes filter on their trailing 8 chars.
        let long = MatchTargets::new(&[("", "123456789ABC")], false);
        let f = long.patterns[0]
            .suffix_filter
            .as_ref()
            .unwrap();
        assert_eq!(f.len, 8);
        assert_eq!(f.modulus, 58u64.pow(8));
    }

    #[test]
    fn suffix_filter_suffix_only_cs_random() {
        let patterns: &[(&str, &str)] = &[
            ("", "1"),
            ("", "9"),
            ("", "zz"),
            ("", "KAr"),
            ("", "aBc9"),
            ("", "111"),
            ("", "xYz9q"),
            ("", "abcdefgh"),
            ("", "123456789"),
        ];
        assert_agrees(patterns, false, &sampled_pubkeys(2000, 0x5eed));
    }

    #[test]
    fn suffix_filter_suffix_only_ci_random() {
        // Caller-canonical CI form: lowercased except `L`.
        let patterns: &[(&str, &str)] = &[
            ("", "a"),
            ("", "zz"),
            ("", "Lm"),
            ("", "ab1"),
            ("", "mithriL"),
            ("", "z9x8y7"),
            ("", "111"),
        ];
        assert_agrees(patterns, true, &sampled_pubkeys(2000, 0xc1));
    }

    #[test]
    fn suffix_filter_mixed_prefix_suffix_random() {
        let cs: &[(&str, &str)] = &[
            ("XkC", "KAr"),
            ("Xk", "Ar"),
            ("zz", "KAr"),
            ("Har", "nic"),
            ("", "KAr"),
            ("XkC", ""),
        ];
        assert_agrees(cs, false, &sampled_pubkeys(1500, 0xbeef));

        let ci: &[(&str, &str)] =
            &[("xk", "kar"), ("", "Lm"), ("mithriL", ""), ("zz", "zz")];
        assert_agrees(ci, true, &sampled_pubkeys(1500, 0xf00d));
    }

    #[test]
    fn suffix_filter_boundary_keys() {
        let keys = boundary_keys();
        let cs: &[(&str, &str)] = &[
            ("", "1"),
            ("", "111"),
            ("", "11111111"),
            ("", "111111111"),
            ("", "111111111111"),
            ("", "2"),
            ("", "12"),
            ("", "1112"),
            ("", "zz"),
            ("111", "111"),
        ];
        assert_agrees(cs, false, &keys);

        let ci: &[(&str, &str)] = &[
            ("", "1"),
            ("", "11111111"),
            ("", "ab"),
            ("", "Lm"),
            ("", "zz"),
        ];
        assert_agrees(ci, true, &keys);
    }

    #[test]
    fn suffix_filter_long_suffix_hits_known_key() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        for len in [9usize, 10, 12] {
            let suffix = &key[key.len() - len..];
            let target = MatchTargets::new(&[("", suffix)], false);
            assert_eq!(target.match_mask(&bytes), 1, "suffix {suffix}");
            let target =
                MatchTargets::new(&[(&key[..3], suffix)], false);
            assert_eq!(target.match_mask(&bytes), 1, "suffix {suffix}");
        }
    }

    // Lead differential fuzz: raw 32-byte inputs (the matcher is a
    // total function on bytes; real pubkeys are a subset), plus
    // structured leading-zero and interval-boundary corpora, all
    // checked mask-for-mask against full `fd_bs58` encode + string
    // match (`encode_matches`, the in-file naive oracle).
    #[test]
    fn differential_fuzz_raw_and_boundary() {
        let mut state = 0x243f6a8885a308d3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut keys: Vec<[u8; 32]> = Vec::new();
        for _ in 0..25_000 {
            let mut k = [0u8; 32];
            for c in k.chunks_mut(8) {
                c.copy_from_slice(&next().to_le_bytes()[..c.len()]);
            }
            keys.push(k);
        }
        // Adversarial leading-zero counts (encoded leading '1's).
        for z in 0..32usize {
            for _ in 0..16 {
                let mut k = keys[next() as usize % keys.len()];
                k[..z].fill(0);
                if z < 32 && k[z] == 0 {
                    k[z] = 1;
                }
                keys.push(k);
            }
        }
        // Small-integer boundaries around powers of 58 and 256.
        let mut small: Vec<u128> = vec![0, 1, 2, 56, 57, 58, 59, 116];
        for e in 1..8u32 {
            let p = 58u128.pow(e);
            small.extend([
                p - 1,
                p,
                p + 1,
                2 * p - 1,
                2 * p,
                2 * p + 1,
            ]);
        }
        for v in small {
            let mut k = [0u8; 32];
            k[16..].copy_from_slice(&v.to_be_bytes());
            keys.push(k);
            let mut kf = [0xFFu8; 32];
            kf[16..].copy_from_slice(&(u128::MAX - v).to_be_bytes());
            keys.push(kf);
        }
        keys.extend(boundary_keys());
        // Prefix-range interval edges: lo-1, lo, hi, hi+1 per range.
        for pref in ["Harmonic", "zz", "Z"] {
            for (lo, hi) in prefix_ranges(pref, false).unwrap() {
                for cand in [lo, hi] {
                    let n = BigUint::from_bytes_be(&cand);
                    for d in [-1i64, 0, 1] {
                        let m = if d < 0 {
                            &n - BigUint::from(1u8)
                        } else if d > 0 {
                            &n + BigUint::from(1u8)
                        } else {
                            n.clone()
                        };
                        let v = m.to_bytes_be();
                        if v.len() <= 32 {
                            let mut k = [0u8; 32];
                            k[32 - v.len()..].copy_from_slice(&v);
                            keys.push(k);
                        }
                    }
                }
            }
        }
        let cs: &[(&str, &str)] = &[
            ("", "pump"),
            ("", "LM"),
            ("", "1"),
            ("", "111"),
            ("zz", "LM"),
            ("Harmonic", ""),
            ("", "123456789"),
            ("XkC", "KAr"),
            ("1abc", ""),
            ("111", "111"),
        ];
        assert_agrees(cs, false, &keys);
        let ci: &[(&str, &str)] = &[
            ("", "pump"),
            ("", "Lm"),
            ("mithriL", ""),
            ("", "L"),
            ("", "zz"),
            ("xk", "kar"),
        ];
        assert_agrees(ci, true, &keys);
    }

    /// Exhaustive over every `N < 58^3` (as a 32-byte key, so the
    /// encoding is mostly leading '1's): zero disagreements allowed.
    #[test]
    fn suffix_filter_exhaustive_small_values() {
        let patterns: &[(&str, &str)] =
            &[("", "LM"), ("", "111"), ("", "12"), ("", "pump")];
        let target = MatchTargets::new(patterns, false);
        for n in 0..58u32.pow(3) {
            let mut k = [0u8; 32];
            k[28..].copy_from_slice(&n.to_be_bytes());
            let mut expected = 0u64;
            for (i, &(prefix, suffix)) in patterns.iter().enumerate() {
                if encode_matches(&k, prefix, suffix, false) {
                    expected |= 1u64 << i;
                }
            }
            assert_eq!(target.match_mask(&k), expected, "n={n}");
        }
    }

    #[test]
    fn gpu_blob_roundtrip_header() {
        let t = MatchTargets::new(&[("ab", "xy"), ("Z", "")], false);
        let blob = t.gpu_blob();
        assert_eq!(&blob[..4], &2u32.to_le_bytes());
        assert_eq!(blob[4], 2); // max prefix
        assert_eq!(blob[5], 2); // max suffix
        assert_eq!(blob[6], 2); // plen of first
    }
}
