use log::info;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::thread;
use time::{OffsetDateTime, UtcOffset};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::format_description;
pub use crate::pow::hasher::HeaderHasher;
use crate::{
    pow::{
        hasher::{Hasher, PowHash},
        mem_hash::mem_hash,
    },
    proto::{RpcBlock, RpcBlockHeader},
    target::{self, Uint256},
    Error, Hash,
};
use vecno_miner::Worker;

mod hasher;
mod mem_hash;

static SHARE_LOGGER_RUNNING: AtomicBool = AtomicBool::new(false);
pub static SHARE_TRACKER: Lazy<Arc<ShareTracker>> = Lazy::new(|| Arc::new(ShareTracker::new()));

#[derive(Clone)]
pub struct ShareTracker {
    shares: Arc<Mutex<Vec<Instant>>>,
}

impl ShareTracker {
    pub fn new() -> Self {
        let tracker = ShareTracker {
            shares: Arc::new(Mutex::new(Vec::new())),
        };
        tracker.start_cleanup();
        tracker
    }

    pub fn record_share(&self) {
        let mut shares = self.shares.lock().unwrap();
        shares.push(Instant::now());
    }

    fn start_cleanup(&self) {
        let shares = Arc::clone(&self.shares);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(10));
            let now = Instant::now();
            let ten_secs_ago = now - Duration::from_secs(10);
            let mut shares = shares.lock().unwrap();
            shares.retain(|&t| t >= ten_secs_ago);
        });
    }

    pub fn start_reporter(&self) {
        if SHARE_LOGGER_RUNNING.swap(true, Ordering::SeqCst) {
            return;
        }

        let shares = Arc::clone(&self.shares);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(10));
                let count = {
                    let now = Instant::now();
                    let ten_secs_ago = now - Duration::from_secs(10);
                    let shares = shares.lock().unwrap();
                    shares.iter().filter(|&t| *t >= ten_secs_ago).count()
                };

                if count > 0 {
                    info!("Successfully found {} shares in the past 10 seconds", count);
                }
            }
        });
    }
}

#[derive(Clone)]
pub enum BlockSeed {
    FullBlock(Box<RpcBlock>),
    PartialBlock {
        id: String,
        header_hash: [u64; 4],
        timestamp: u64,
        nonce: u64,
        target: Uint256,
        nonce_mask: u64,
        nonce_fixed: u64,
        hash: Option<String>,
    },
}

impl BlockSeed {
    pub fn timestamp(&self) -> u64 {
        match self {
            BlockSeed::FullBlock(block) => block.header.as_ref().map(|h| h.timestamp).unwrap_or(0).try_into().unwrap(),
            BlockSeed::PartialBlock { timestamp, .. } => *timestamp,
        }
    }

    pub fn report_block(&self) {
        let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");
        let local_offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);

        match self {
            BlockSeed::FullBlock(block) => {
                let block_hash = block.block_hash().expect("Block hash should be available");
                let header = block.header.as_ref().expect("header must exist");

                let local_time = if header.timestamp > 0 {
                    let utc = OffsetDateTime::from_unix_timestamp((header.timestamp / 1000) as i64)
                        .unwrap_or(OffsetDateTime::UNIX_EPOCH);
                    utc.to_offset(local_offset)
                } else {
                    OffsetDateTime::now_local()
                        .unwrap_or_else(|_| OffsetDateTime::now_utc().to_offset(local_offset))
                };

                info!(
                    "Block found: {:x} (Timestamp: {})",
                    block_hash,
                    local_time.format(&format).unwrap_or("unknown".to_string())
                );
            }

            BlockSeed::PartialBlock { hash, timestamp, .. } => {
                if let Some(real_hash) = hash {
                    let local_time = if *timestamp > 0 {
                        let utc = OffsetDateTime::from_unix_timestamp(*timestamp as i64)
                            .unwrap_or(OffsetDateTime::now_utc());
                        utc.to_offset(local_offset)
                    } else {
                        OffsetDateTime::now_local()
                            .unwrap_or_else(|_| OffsetDateTime::now_utc().to_offset(local_offset))
                    };

                    info!(
                        "Block found: {} (Timestamp: {})",
                        real_hash,
                        local_time.format(&format).unwrap_or("unknown".to_string())
                    );
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct State {
    pub id: usize,
    pub target: Uint256,
    pub pow_hash_header: [u8; 72],
    block: Arc<BlockSeed>,
    hasher: PowHash,
    pub nonce_mask: u64,
    pub nonce_fixed: u64,
    pub timestamp: u64,
}

impl State {
    #[inline]
    pub fn new(id: usize, block_seed: BlockSeed) -> Result<Self, Error> {
        let pre_pow_hash;
        let header_timestamp: u64;
        let header_target;
        let nonce_mask: u64;
        let nonce_fixed: u64;

        match &block_seed {
            BlockSeed::FullBlock(block) => {
                let header = block.header.as_ref().ok_or("Header is missing")?;
                header_target = target::u256_from_compact_target(header.bits);
                let mut hasher = HeaderHasher::new();
                serialize_header(&mut hasher, header, true);
                pre_pow_hash = hasher.finalize();
                header_timestamp = header.timestamp as u64;
                nonce_mask = 0xffffffffffffffffu64;
                nonce_fixed = 0;
            }
            BlockSeed::PartialBlock {
                header_hash,
                timestamp,
                target,
                nonce_fixed: fixed,
                nonce_mask: mask,
                ..
            } => {
                pre_pow_hash = Hash::new(*header_hash);
                header_timestamp = *timestamp;
                header_target = *target;
                nonce_mask = *mask;
                nonce_fixed = *fixed;
            }
        }

        let hasher = PowHash::new(pre_pow_hash, header_timestamp);
        let mut pow_hash_header = [0u8; 72];
        pow_hash_header.copy_from_slice(
            [
                pre_pow_hash.to_le_bytes().as_slice(),
                header_timestamp.to_le_bytes().as_slice(),
                [0u8; 32].as_slice(),
            ]
            .concat()
            .as_slice(),
        );

        Ok(Self {
            id,
            target: header_target,
            pow_hash_header,
            block: Arc::new(block_seed),
            hasher,
            nonce_mask,
            nonce_fixed,
            timestamp: header_timestamp,
        })
    }

    #[inline(always)]
    pub fn calculate_pow(&self, nonce: u64) -> Uint256 {
        let block_hash = self.hasher.clone().finalize_with_nonce(nonce);
        let hash = mem_hash(block_hash, self.timestamp, nonce);
        Uint256::from_le_bytes(hash.as_bytes())
    }

    #[inline(always)]
    pub fn check_pow(&mut self, nonce: u64) -> bool {
        let pow = self.calculate_pow(nonce);
        pow <= self.target
    }

    #[inline(always)]
    pub fn generate_block_if_pow(
        &mut self,
        nonce: u64,
        _share_tracker: &Arc<ShareTracker>,
    ) -> Option<BlockSeed> {
        self.check_pow(nonce).then(|| {
            let mut block_seed = (*self.block).clone();
            match block_seed {
                BlockSeed::FullBlock(ref mut block) => {
                    let header = block.header.as_mut().expect("Header exists on creation");
                    header.nonce = nonce;
                }
                BlockSeed::PartialBlock {
                    nonce: ref mut header_nonce,
                    ..
                } => {
                    *header_nonce = nonce;
                }
            }
            block_seed
        })
    }

    pub fn load_to_gpu(&self, gpu_work: &mut dyn Worker) {
        gpu_work.load_block_constants(&self.pow_hash_header, &self.target.0);
    }

    #[inline(always)]
    pub fn pow_gpu(&self, gpu_work: &mut dyn Worker) {
        gpu_work.calculate_hash(None, self.nonce_mask, self.nonce_fixed, self.timestamp);
    }
}

#[cfg(not(any(target_pointer_width = "64", target_pointer_width = "32")))]
compile_error!("Supporting only 32/64 bits");

#[inline(always)]
pub fn serialize_header<H: Hasher>(hasher: &mut H, header: &RpcBlockHeader, for_pre_pow: bool) {
    let (nonce, timestamp) = if for_pre_pow { (0, 0) } else { (header.nonce, header.timestamp) };
    let num_parents = header.parents.len();
    let version: u16 = header.version.try_into().unwrap();
    hasher.update(version.to_le_bytes()).update((num_parents as u64).to_le_bytes());

    let mut hash = [0u8; 32];
    for parent in &header.parents {
        hasher.update((parent.parent_hashes.len() as u64).to_le_bytes());
        for hash_string in &parent.parent_hashes {
            decode_to_slice(hash_string, &mut hash).unwrap();
            hasher.update(hash);
        }
    }
    decode_to_slice(&header.hash_merkle_root, &mut hash).unwrap();
    hasher.update(hash);

    decode_to_slice(&header.accepted_id_merkle_root, &mut hash).unwrap();
    hasher.update(hash);
    decode_to_slice(&header.utxo_commitment, &mut hash).unwrap();
    hasher.update(hash);

    hasher
        .update(timestamp.to_le_bytes())
        .update(header.bits.to_le_bytes())
        .update(nonce.to_le_bytes())
        .update(header.daa_score.to_le_bytes())
        .update(header.blue_score.to_le_bytes());

    let blue_work_len = (header.blue_work.len() + 1) / 2;
    if header.blue_work.len() % 2 == 0 {
        decode_to_slice(&header.blue_work, &mut hash[..blue_work_len]).unwrap();
    } else {
        let mut blue_work = String::with_capacity(header.blue_work.len() + 1);
        blue_work.push('0');
        blue_work.push_str(&header.blue_work);
        decode_to_slice(&blue_work, &mut hash[..blue_work_len]).unwrap();
    }

    hasher.update((blue_work_len as u64).to_le_bytes()).update(&hash[..blue_work_len]);

    decode_to_slice(&header.pruning_point, &mut hash).unwrap();
    hasher.update(hash);
}

#[derive(Debug)]
enum FromHexError {
    OddLength,
    InvalidStringLength,
    InvalidHexCharacter { c: char, index: usize },
}

impl std::fmt::Display for FromHexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FromHexError::OddLength => write!(f, "hex string has odd length"),
            FromHexError::InvalidStringLength => write!(f, "hex string length does not match output buffer"),
            FromHexError::InvalidHexCharacter { c, index } => {
                write!(f, "invalid hex character '{}' at index {}", c, index)
            }
        }
    }
}

impl std::error::Error for FromHexError {}

#[inline(always)]
fn decode_to_slice<T: AsRef<[u8]>>(data: T, out: &mut [u8]) -> Result<(), FromHexError> {
    let data = data.as_ref();
    if data.len() % 2 != 0 {
        return Err(FromHexError::OddLength);
    }
    if data.len() / 2 != out.len() {
        return Err(FromHexError::InvalidStringLength);
    }

    for (i, byte) in out.iter_mut().enumerate() {
        *byte = val(data[2 * i], 2 * i)? << 4 | val(data[2 * i + 1], 2 * i + 1)?;
    }

    #[inline(always)]
    fn val(c: u8, idx: usize) -> Result<u8, FromHexError> {
        match c {
            b'A'..=b'F' => Ok(c - b'A' + 10),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'0'..=b'9' => Ok(c - b'0'),
            _ => Err(FromHexError::InvalidHexCharacter {
                c: c as char,
                index: idx,
            }),
        }
    }

    Ok(())
}