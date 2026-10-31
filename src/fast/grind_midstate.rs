//! Fast SHA-256 evaluation for `Pubkey::create_with_seed` grinding, plus
//! an exact byte-level prefilter for base58 prefix/suffix patterns.
//!
//! This module is intentionally self-contained: it does not depend on how
//! `crate::fast` wires it into the grind loop.

use num_bigint::BigUint;
use num_traits::One;
use sha2::digest::generic_array::GenericArray;

/// Standard SHA-256 initial chaining value (FIPS 180-4 §5.3.3).
const SHA256_IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const BASE58_ALPHABET: &[u8; 58] =
    b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Longest possible base58 encoding of a 32-byte value.
const MAX_ENCODED_LEN: usize = 44;

/// Precomputed SHA-256 block templates for one `(base, owner)` pair.
///
/// Block-layout proof
/// ------------------
/// `Pubkey::create_with_seed(base, seed, owner)` is
/// `SHA256(base ‖ seed ‖ owner)` where `base` and `owner` are 32 bytes
/// each and `seed` is 16 bytes, so the message `M` is exactly 80 bytes
/// (640 bits). SHA-256 padding appends a `0x80` byte, zero bytes, and the
/// 64-bit big-endian bit length, bringing the padded message to exactly
/// 128 bytes = two 64-byte blocks:
///
/// ```text
/// block 1 = M[0..64]   = base (32) ‖ seed (16) ‖ owner[0..16]  (16)
/// block 2 = M[64..80]  = owner[16..32] (16)
///         ‖ 0x80 ‖ 0x00 × 39 ‖ 640u64.to_be_bytes()
/// ```
///
/// `compress256` applies only the SHA-256 compression function — no
/// padding, no length encoding — so feeding it exactly these two padded
/// blocks starting from the standard IV and serialising the resulting
/// state as big-endian bytes yields precisely `SHA256(M)`.
///
/// Neither block is constant across attempts in general: block 2 is
/// fully constant for a fixed `owner`, but block 1 contains the varying
/// `seed` at bytes `32..48`, so a base-only "midstate" saves nothing
/// (32 bytes of `base` never fill a block on their own anyway). The win
/// here is therefore not fewer compressions — it is always exactly two —
/// but removing the generic `Digest` buffering / clone / finalize
/// overhead per attempt and specialising the constant second block.
#[derive(Clone, Debug)]
pub struct GrindMidstate {
    block1_template: [u8; 64],
    block2: [u8; 64],
}

impl GrindMidstate {
    pub fn new(base: &[u8; 32], owner: &[u8; 32]) -> Self {
        let mut block1_template = [0u8; 64];
        block1_template[..32].copy_from_slice(base);
        // Bytes 32..48 are a placeholder, overwritten with the seed on
        // every call to `pubkey_bytes`.
        block1_template[48..].copy_from_slice(&owner[..16]);

        let mut block2 = [0u8; 64];
        block2[..16].copy_from_slice(&owner[16..]);
        block2[16] = 0x80;
        // Bytes 17..56 stay zero (padding).
        block2[56..].copy_from_slice(&640u64.to_be_bytes());

        Self {
            block1_template,
            block2,
        }
    }

    /// `SHA256(base ‖ seed ‖ owner)` for the `seed` given, i.e. the bytes
    /// of `Pubkey::create_with_seed(base, seed_str, owner)`.
    #[inline]
    pub fn pubkey_bytes(&self, seed: &[u8; 16]) -> [u8; 32] {
        let mut block1 = self.block1_template;
        block1[32..48].copy_from_slice(seed);

        let mut state = SHA256_IV;
        sha2::compress256(
            &mut state,
            &[
                GenericArray::from(block1),
                GenericArray::from(self.block2),
            ],
        );

        let mut out = [0u8; 32];
        for (i, word) in state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// Exact byte ranges for a base58 prefix, mirroring
/// `check_match::prefix_ranges`.
///
/// Returns `None` (no usable filter) when `ci` is set, the prefix is
/// empty, the prefix starts with `'1'`, or the prefix contains a
/// non-base58 character. A `'1'`-leading prefix is excluded because
/// leading `'1'`s encode leading zero *bytes*, not the numeric value, so
/// the value-range argument below does not apply to them.
///
/// Otherwise: a 32-byte value encodes (ignoring leading `'1'`s, which a
/// non-`'1'` prefix cannot match anyway) as a base58 string of some
/// length `L` in `prefix.len()..=44` whose first `prefix.len()` digits
/// have value `v`. Such a value lies in
/// `[v * 58^(L-plen), (v+1) * 58^(L-plen) - 1]`. Encodings of 32-byte
/// values are at least `2^248` in the range sense used by the existing
/// matcher, hence the lower clamp; the upper clamp is `2^256 - 1`.
fn prefix_ranges(
    prefix: &str,
    case_insensitive: bool,
) -> Option<Vec<([u8; 32], [u8; 32])>> {
    if case_insensitive || prefix.is_empty() || prefix.starts_with('1')
    {
        return None;
    }
    let mut value = BigUint::from(0u8);
    for b in prefix.bytes() {
        value *= 58u8;
        value += BASE58_ALPHABET
            .iter()
            .position(|&c| c == b)?;
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
    for _ in prefix.len()..=MAX_ENCODED_LEN {
        let lo = (&value * &scale).max(minimum.clone());
        let hi = ((&value + 1u8) * &scale - 1u8).min(maximum.clone());
        if lo <= hi {
            ranges.push((to_bytes(&lo), to_bytes(&hi)));
        }
        scale *= 58u8;
    }
    Some(ranges)
}

/// Residue filter for a base58 suffix: a value passes iff
/// `value mod modulus == value_residue`, where `modulus = 58^k` and
/// `value_residue` is the trailing `k` suffix characters read as a
/// base58 number.
#[derive(Clone, Copy, Debug)]
struct SuffixResidue {
    modulus: u64,
    value: u64,
}

/// How many trailing suffix characters the residue filter uses. Ten
/// digits give a `58^10 ≈ 4.3 × 10^17` modulus, which still fits
/// comfortably in a `u64` while rejecting all but ~1 in 4 × 10^17
/// non-matching suffixes.
const SUFFIX_RESIDUE_DIGITS: usize = 10;

fn suffix_residue(
    suffix: &str,
    case_insensitive: bool,
) -> Option<SuffixResidue> {
    if case_insensitive || suffix.is_empty() {
        return None;
    }
    let bytes = suffix.as_bytes();
    let k = bytes.len().min(SUFFIX_RESIDUE_DIGITS);
    let mut value = 0u64;
    let mut modulus = 1u64;
    for &b in &bytes[bytes.len() - k..] {
        value = value * 58
            + BASE58_ALPHABET
                .iter()
                .position(|&c| c == b)? as u64;
        modulus *= 58;
    }
    Some(SuffixResidue { modulus, value })
}

#[derive(Clone, Debug)]
struct PatternFilter {
    prefix_ranges: Option<Vec<([u8; 32], [u8; 32])>>,
    suffix: Option<SuffixResidue>,
}

/// Grind-local exact rejection filter over raw 32-byte pubkey bytes.
///
/// `could_match` never false-rejects: if the base58 string of `bytes`
/// starts with a pattern's prefix and ends with its suffix (under the
/// case-sensitivity the filter was built with), the pattern always
/// survives. Survivors — and every candidate for patterns the filter
/// cannot reason about — must still go through the authoritative string
/// match; this filter only skips candidates cheaply.
///
/// Exactness of the two sub-filters:
///
/// * Prefix: see [`prefix_ranges`]. It is only built for case-sensitive,
///   non-empty prefixes not starting with `'1'`; anything else leaves
///   the pattern with no prefix filter.
/// * Suffix: let `k = min(suffix_len, 10)`, `modulus = 58^k`, and let
///   `v` be the last `k` suffix characters interpreted in base58. If an
///   encoded string `S` ends with the suffix, then `S` has at least `k`
///   characters and its last `k` characters equal those characters.
///   Reading `S` as a base58 number, the leading `'1'` padding that
///   encodes leading zero bytes contributes digit value 0, so the
///   numeric value `N` of the 32 bytes satisfies
///   `N mod 58^k = (last k base58 digits of S) = v`: the low `k` base58
///   digits of `N` are exactly the last `k` characters whenever there
///   are at least `k` of them, including the case where `N` has fewer
///   than `k` significant digits and the remaining positions are
///   `'1'` (zero) padding. Hence a residue mismatch proves the suffix
///   cannot match. Case-insensitive and empty suffixes get no filter.
///
/// A pattern with no filters at all always survives (conservative).
#[derive(Clone, Debug)]
pub struct BytePrefilter {
    patterns: Vec<PatternFilter>,
    /// Largest suffix modulus across patterns, or 0 if none has one.
    /// The residue of the candidate bytes is computed once modulo this
    /// value and reduced modulo each pattern's own modulus.
    max_modulus: u64,
}

impl BytePrefilter {
    pub fn new(
        patterns: &[(&str, &str)],
        case_insensitive: bool,
    ) -> Self {
        let mut max_modulus = 0u64;
        let mut out = Vec::with_capacity(patterns.len());
        for &(prefix, suffix) in patterns {
            let suffix = suffix_residue(suffix, case_insensitive);
            if let Some(s) = suffix {
                max_modulus = max_modulus.max(s.modulus);
            }
            out.push(PatternFilter {
                prefix_ranges: prefix_ranges(prefix, case_insensitive),
                suffix,
            });
        }
        Self {
            patterns: out,
            max_modulus,
        }
    }

    /// True if any pattern might match `bytes`; false only when every
    /// pattern is exactly ruled out. Authoritative matching must still
    /// run on `true`.
    #[inline]
    pub fn could_match(&self, bytes: &[u8; 32]) -> bool {
        // Residue of `bytes` modulo `max_modulus`, computed lazily on
        // first need (only patterns with a suffix filter need it).
        let mut residue_max: Option<u64> = None;

        for pattern in &self.patterns {
            if let Some(ranges) = &pattern.prefix_ranges {
                if !ranges
                    .iter()
                    .any(|(lo, hi)| bytes >= lo && bytes <= hi)
                {
                    continue;
                }
            }
            if let Some(suffix) = &pattern.suffix {
                let r = *residue_max.get_or_insert_with(|| {
                    // self.max_modulus > 0 whenever any suffix exists.
                    // Process 8 bytes at a time in u128: r < 58^10 < 2^59,
                    // so r * 2^64 + chunk < 2^123 never overflows u128.
                    // (A u64 `r * 256` would wrap for modulus 58^10 and
                    // could false-reject a true suffix match.)
                    let modulus = u128::from(self.max_modulus);
                    let mut r: u128 = 0;
                    for chunk in bytes.chunks_exact(8) {
                        let v = u64::from_be_bytes(
                            chunk.try_into().unwrap(),
                        );
                        r = ((r << 64) + u128::from(v)) % modulus;
                    }
                    r as u64
                });
                if r % suffix.modulus != suffix.value {
                    continue;
                }
            }
            // Neither filter rejected this pattern.
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use solana_pubkey::Pubkey;

    fn reference_digest(
        base: &[u8; 32],
        seed: &[u8; 16],
        owner: &[u8; 32],
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(base);
        hasher.update(seed);
        hasher.update(owner);
        hasher.finalize().into()
    }

    #[test]
    fn pubkey_bytes_matches_create_with_seed_and_digest() {
        let mut patterned_base = [0u8; 32];
        let mut patterned_owner = [0u8; 32];
        for i in 0..32 {
            patterned_base[i] = (i as u8)
                .wrapping_mul(37)
                .wrapping_add(11);
            patterned_owner[i] = (i as u8)
                .wrapping_mul(91)
                .wrapping_add(7);
        }
        let bases_owners: [([u8; 32], [u8; 32]); 4] = [
            ([0u8; 32], [0u8; 32]),
            ([0xffu8; 32], [0xffu8; 32]),
            ([0x42u8; 32], [0x24u8; 32]),
            (patterned_base, patterned_owner),
        ];
        let ascii_seeds: [[u8; 16]; 3] = [
            *b"0123456789ABCDEF",
            *b"aaaaaaaaaaaaaaaa",
            *b"SeedSeedSeedSeed",
        ];
        let arbitrary_seeds: [[u8; 16]; 4] = [
            [0u8; 16],
            [0xffu8; 16],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [
                0xde, 0xad, 0xbe, 0xef, 0x00, 0x80, 0xff, 0x7f, 0x01,
                0xfe, 0x02, 0xfd, 0x03, 0xfc, 0x04, 0xfb,
            ],
        ];

        for (base, owner) in &bases_owners {
            let midstate = GrindMidstate::new(base, owner);
            for seed in &ascii_seeds {
                let ours = midstate.pubkey_bytes(seed);
                assert_eq!(ours, reference_digest(base, seed, owner));
                let seed_str = std::str::from_utf8(seed).unwrap();
                let expected = Pubkey::create_with_seed(
                    &Pubkey::new_from_array(*base),
                    seed_str,
                    &Pubkey::new_from_array(*owner),
                )
                .unwrap()
                .to_bytes();
                assert_eq!(ours, expected, "seed {seed_str}");
            }
            for seed in &arbitrary_seeds {
                assert_eq!(
                    midstate.pubkey_bytes(seed),
                    reference_digest(base, seed, owner),
                    "arbitrary seed {seed:?}"
                );
            }
        }
    }

    fn string_matches(s: &str, prefix: &str, suffix: &str) -> bool {
        (prefix.is_empty() || s.starts_with(prefix))
            && (suffix.is_empty() || s.ends_with(suffix))
    }

    #[test]
    fn prefilter_never_false_rejects() {
        // (prefix, suffix, case_insensitive)
        let cases: [(&str, &str, bool); 6] = [
            ("ZZZ", "", false),            // prefix only
            ("", "xYz", false),            // suffix only
            ("AB", "yz", false),           // both
            ("abC", "De", true),           // ci: conservative
            ("1abc", "", false),           // leading '1': conservative
            ("", "ABCDEFGHJKLMNP", false), // suffix longer than 10
        ];
        let filters: Vec<BytePrefilter> = cases
            .iter()
            .map(|&(p, s, ci)| BytePrefilter::new(&[(p, s)], ci))
            .collect();

        for i in 0u32..20_000 {
            let bytes: [u8; 32] =
                Sha256::digest(i.to_le_bytes()).into();
            let encoded = fd_bs58::encode_32(bytes);
            for (idx, &(prefix, suffix, ci)) in cases.iter().enumerate()
            {
                // For the ci case any reasonable notion of a string match
                // implies survival (the filter keeps no ci filters at
                // all), so test it against the case-sensitive reading.
                let _ = ci;
                if string_matches(&encoded, prefix, suffix) {
                    assert!(
                        filters[idx].could_match(&bytes),
                        "false reject: case {idx} ({prefix:?}, {suffix:?}) \
                         key {encoded} counter {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn prefilter_long_suffix_residue_no_overflow_false_reject() {
        // Regression: with a 10-digit suffix the modulus is 58^10 > 2^56,
        // so a u64 `r * 256 + b` residue loop wraps and can false-reject.
        // Build the integer whose canonical base58 form is exactly the
        // suffix, as a 32-byte big-endian value; its fd encoding ends
        // with that suffix (leading zero bytes become leading '1's).
        let suffix = "ABCDEFGHJK";
        let mut value = BigUint::from(0u8);
        for b in suffix.bytes() {
            let digit = BASE58_ALPHABET
                .iter()
                .position(|&c| c == b)
                .unwrap();
            value = value * 58u8 + digit as u8;
        }
        let v = value.to_bytes_be();
        let mut bytes = [0u8; 32];
        bytes[32 - v.len()..].copy_from_slice(&v);
        let encoded = fd_bs58::encode_32(bytes);
        assert!(encoded.ends_with(suffix), "{encoded}");
        let filter = BytePrefilter::new(&[("", suffix)], false);
        assert!(
            filter.could_match(&bytes),
            "false reject of {encoded}"
        );
        // And a different long suffix is rejected by the residue.
        let miss = BytePrefilter::new(&[("", "ABCDEFGHJL")], false);
        assert!(!miss.could_match(&bytes));
    }

    #[test]
    fn prefilter_conservative_patterns_always_survive() {
        let ci = BytePrefilter::new(&[("abC", "De")], true);
        let leading_one = BytePrefilter::new(&[("1abc", "")], false);
        let empty = BytePrefilter::new(&[("", "")], false);
        let bytes = [0xabu8; 32];
        assert!(ci.could_match(&bytes));
        assert!(leading_one.could_match(&bytes));
        assert!(empty.could_match(&bytes));
    }

    #[test]
    fn prefilter_rejects_most_non_matches_for_prefix() {
        let filter = BytePrefilter::new(&[("ZZZ", "")], false);
        let total = 20_000u32;
        let mut rejected = 0u32;
        for i in 0..total {
            let bytes: [u8; 32] =
                Sha256::digest(i.to_le_bytes()).into();
            if !filter.could_match(&bytes) {
                rejected += 1;
            }
        }
        assert!(
            rejected * 100 > 90 * total,
            "only rejected {rejected}/{total} for prefix ZZZ"
        );
    }
}
