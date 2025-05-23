use crate::Hash;
use blake3::Hasher as Blake3State;

const BLOCK_HASH_DOMAIN: &[u8; 32] = b"BlockHash\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

#[derive(Clone)]
pub(super) struct PowHasher(Blake3State);

#[derive(Clone, Copy)]
pub(super) struct HeavyHasher;

#[derive(Clone)]
pub struct HeaderHasher(Blake3State);

impl PowHasher {
    #[inline]
    pub(super) fn new(pre_pow_hash: Hash, timestamp: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&pre_pow_hash.to_le_bytes());
        hasher.update(&timestamp.to_le_bytes());
        hasher.update(&[0u8; 32]);
        Self(hasher)
    }

    #[inline(always)]
    pub(super) fn finalize_with_nonce(&mut self, nonce: u64) -> Hash {
        let mut hasher = self.0.clone(); // Clone the internal blake3::Hasher
        hasher.update(&nonce.to_le_bytes());
        let mut hash_bytes = [0u8; 32];
        hasher.finalize_xof().fill(&mut hash_bytes);
        let mut words = [0u64; 4];
        for (i, chunk) in hash_bytes.chunks(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Hash::new(words)
    }
}

impl HeavyHasher {
    #[inline(always)]
    pub(super) fn hash(in_hash: Hash) -> Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&in_hash.to_le_bytes());
        let mut hash_bytes = [0u8; 32];
        hasher.finalize_xof().fill(&mut hash_bytes);
        let mut words = [0u64; 4];
        for (i, chunk) in hash_bytes.chunks(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Hash::new(words)
    }
}

impl HeaderHasher {
    #[inline(always)]
    pub fn new() -> Self {
        Self(blake3::Hasher::new_keyed(BLOCK_HASH_DOMAIN))
    }

    pub fn write<A: AsRef<[u8]>>(&mut self, data: A) {
        self.0.update(data.as_ref());
    }

    #[inline(always)]
    pub fn finalize(self) -> Hash {
        let hash = self.0.finalize();
        let bytes = hash.as_bytes();
        let mut words = [0u64; 4];
        for (i, chunk) in bytes.chunks(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Hash::new(words)
    }
}

pub trait Hasher {
    fn update<A: AsRef<[u8]>>(&mut self, data: A) -> &mut Self;
}

impl Hasher for HeaderHasher {
    fn update<A: AsRef<[u8]>>(&mut self, data: A) -> &mut Self {
        self.write(data);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{HeavyHasher, PowHasher};
    use crate::Hash;

    #[test]
    fn test_pow_hash() {
        let timestamp: u64 = 1715521488610;
        let nonce: u64 = 11171827086635415026;
        let pre_pow_hash = Hash::from_le_bytes([
            99, 231, 29, 85, 153, 225, 235, 207, 36, 237, 3, 55, 106, 21, 221, 122, 28, 51, 249, 76, 190, 128, 153, 244, 189, 104, 26,
            178, 170, 4, 177, 103,
        ]);
        let mut hasher = PowHasher::new(pre_pow_hash, timestamp);
        let hash1 = hasher.finalize_with_nonce(nonce);

        let mut hasher = blake3::Hasher::new();
        hasher
            .update(&pre_pow_hash.to_le_bytes())
            .update(&timestamp.to_le_bytes())
            .update(&[0u8; 32])
            .update(&nonce.to_le_bytes());
        let mut hash_bytes = [0u8; 32];
        hasher.finalize_xof().fill(&mut hash_bytes);
        let mut words = [0u64; 4];
        for (i, chunk) in hash_bytes.chunks(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        let hash2 = Hash::new(words);
        assert_eq!(hash2, hash1);
    }

    #[test]
    fn test_heavy_hash() {
        let val = Hash::from_le_bytes([42; 32]);
        let hash1 = HeavyHasher::hash(val);

        let mut hasher = blake3::Hasher::new();
        hasher.update(&val.to_le_bytes());
        let mut hash_bytes = [0u8; 32];
        hasher.finalize_xof().fill(&mut hash_bytes);
        let mut words = [0u64; 4];
        for (i, chunk) in hash_bytes.chunks(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        let hash2 = Hash::new(words);
        assert_eq!(hash2, hash1);
    }

}

#[cfg(all(test, feature = "bench"))]
mod benches {
    extern crate test;

    use self::test::{black_box, Bencher};
    use super::{HeavyHasher, PowHasher};
    use crate::Hash;

    #[bench]
    pub fn bench_pow_hash(bh: &mut Bencher) {
        let timestamp: u64 = 1715521488610;
        let mut nonce: u64 = 11171827086635415026;
        let pre_pow_hash = Hash::from_bytes([42; 32]);
        let mut hasher = PowHasher::new(pre_pow_hash, timestamp);

        bh.iter(|| {
            for _ in 0..100 {
                black_box(&mut hasher);
                black_box(&mut nonce);
                black_box(hasher.finalize_with_nonce(nonce));
            }
        });
    }

    #[bench]
    pub fn bench_heavy_hash(bh: &mut Bencher) {
        let mut data = Hash::from_bytes([42; 32]);
        bh.iter(|| {
            for _ in 0..100 {
                black_box(&mut data);
                black_box(HeavyHasher::hash(data));
            }
        });
    }
}