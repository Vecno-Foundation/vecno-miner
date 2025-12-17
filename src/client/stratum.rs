use futures::prelude::*;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use rand::{rngs::ThreadRng, Rng};
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_util::codec::Framed;
use dashmap::{DashMap, DashSet};
use tokio::sync::mpsc::{self, Sender};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::Mutex;
use num_traits::float::FloatCore;
use time::{OffsetDateTime, UtcOffset};
use crate::format_description;
mod stratum_codec;

use crate::client::stratum::stratum_codec::{
    ErrorCode, MiningNotify, MiningSubmit, NewLineJsonCodecError, StratumLine,
    MiningSubscribe, SetExtranonce, StratumCommand, StratumError, StratumLinePayload, StratumResult,
};
use crate::client::Client;
use crate::pow::{BlockSeed, BlockSeed::PartialBlock, SHARE_TRACKER};
use crate::{miner::MinerManager, Error, Uint256};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use log::{debug, info, warn, error};
use once_cell::sync::{Lazy, OnceCell};
use stratum_codec::NewLineJsonCodec;

const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208);

type BlockHandle = JoinHandle<Result<(), Error>>;

static ACCEPTED_BLOCK_HASHES: Lazy<Arc<DashSet<String>>> = Lazy::new(|| Arc::new(DashSet::new()));

#[derive(Default)]
pub struct ShareStats {
    pub accepted: AtomicU64,
    pub stale: AtomicU64,
    pub low_diff: AtomicU64,
    pub duplicate: AtomicU64,
    pub shares_pending: Arc<DashMap<u32, (String, BlockSeed)>>,
}

static SHARE_STATS: OnceCell<Arc<ShareStats>> = OnceCell::new();

impl Display for ShareStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let pending_len = self.shares_pending.len();
        write!(
            f,
            "Shares: {}{}{}{} Pending: {}",
            match self.accepted.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Accepted: {} ", v),
            },
            match self.stale.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Stale: {} ", v),
            },
            match self.low_diff.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Low difficulty: {} ", v),
            },
            match self.duplicate.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Duplicate: {} ", v),
            },
            pending_len
        )
    }
}

pub struct StratumHandler {
    send_channel: Sender<StratumLine>,
    stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>> + Send>>,
    miner_address: String,
    #[allow(dead_code)]
    mine_when_not_synced: bool,
    worker: String,
    password: Option<String>,
    block_template_ctr: Arc<AtomicU16>,
    target_pool: Uint256,
    nonce_mask: u64,
    nonce_fixed: u64,
    extranonce: Option<String>,
    last_stratum_id: Arc<AtomicU32>,
    shares_stats: Arc<ShareStats>,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,
    duplicate_count: Arc<AtomicU64>,
}

#[async_trait(?Send)]
impl Client for StratumHandler {
    async fn register(&mut self) -> Result<(), Error> {
        let mut id = Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst));
        let address_name = format!("{}.{}", self.miner_address, self.worker);
        if self.worker == "default_worker" {
            warn!("Using default worker name for {}. Consider specifying a unique worker name (e.g., pool_address/worker1) to avoid nonce collisions.", address_name);
        }
        if let Err(e) = self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(
                    MiningSubscribe::MiningSubscribeOptions((
                        address_name.clone(),
                        format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                    )),
                )),
                jsonrpc: None,
            })
            .await
        {
            warn!("Failed to send subscribe command for worker {}: {}", self.worker, e);
            return Err(e.into());
        }
        id = Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst));

        debug!("Sending authorize request for worker {} with address_name: {}", self.worker, address_name);
        if let Err(e) = self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
                    address_name,
                    self.password.clone().unwrap_or("x".to_string()),
                ))),
                jsonrpc: None,
            })
            .await
        {
            warn!("Failed to send authorize command for worker {}: {}", self.worker, e);
            return Err(e.into());
        }

        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        info!("Waiting for stratum messages for worker {}", self.worker);
        loop {
            match self.stream.try_next().await {
                Ok(Some(msg)) => {
                    debug!("Received StratumLine: {:?}", msg);
                    self.handle_message(msg, miner).await?
                },
                Ok(None) => {
                    warn!("Stratum connection closed for worker {}", self.worker);
                    return Err("Stratum connection closed".into());
                }
                Err(e) => {
                    warn!("Stratum stream error for worker {}: {}", self.worker, e);
                    return Err(e.into());
                }
            }
        }
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl ShareStats {
    pub fn new() -> Self {
        ShareStats {
            accepted: AtomicU64::new(0),
            stale: AtomicU64::new(0),
            low_diff: AtomicU64::new(0),
            duplicate: AtomicU64::new(0),
            shares_pending: Arc::new(DashMap::new()),
        }
    }
}

impl StratumHandler {
    pub async fn connect(
        address: String,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        password: Option<String>,
    ) -> Result<Box<Self>, Error> {
        let (connect_addr, worker) = address
            .rsplit_once('/')
            .map(|(addr, w)| (addr.to_string(), w.to_string()))
            .unwrap_or((address.clone(), String::from("default_worker")));
        let max_retries = 5;
        let mut retries = 0;

        loop {
            info!(
                "Connecting to {} (worker: {}, password provided: {}, attempt {}/{})",
                connect_addr, worker, password.is_some(), retries + 1, max_retries
            );
            match TcpStream::connect(&connect_addr).await {
                Ok(socket) => {
                    let client = Framed::new(socket, NewLineJsonCodec::new());
                    let (send_channel, recv) = mpsc::channel::<StratumLine>(100);
                    let (sink, stream) = client.split();
                    let worker_clone = worker.clone();
                    tokio::spawn(async move {
                        if let Err(e) = ReceiverStream::new(recv).map(Ok).forward(sink).await {
                            warn!("Stratum sink closed for worker {}: {}", worker_clone, e);
                        }
                    });

                    let share_state = SHARE_STATS.get_or_init(|| Arc::new(ShareStats::new())).clone();
                    let last_stratum_id = Arc::new(AtomicU32::new(0));
                    let (block_channel, block_handle) = Self::create_block_channel(
                        send_channel.clone(),
                        miner_address.clone(),
                        worker.clone(),
                        last_stratum_id.clone(),
                        share_state.clone(),
                    );

                    let handler = Box::new(Self {
                        stream: Box::pin(stream),
                        send_channel,
                        miner_address,
                        mine_when_not_synced,
                        worker,
                        password,
                        block_template_ctr: block_template_ctr
                            .unwrap_or_else(|| Arc::new(AtomicU16::new(ThreadRng::default().random_range(0..10_000)))),
                        target_pool: Default::default(),
                        nonce_mask: 0,
                        nonce_fixed: 0,
                        extranonce: None,
                        last_stratum_id,
                        shares_stats: share_state,
                        block_channel,
                        block_handle,
                        duplicate_count: Arc::new(AtomicU64::new(0)),
                    });

                    {
                        let hashes = ACCEPTED_BLOCK_HASHES.clone();
                        tokio::spawn(async move {
                            loop {
                                tokio::time::sleep(Duration::from_secs(600)).await;
                                hashes.clear();
                            }
                        });
                    }

                    return Ok(handler);
                }
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(format!("Failed to connect after {} retries: {}", max_retries, e).into());
                    }
                    warn!("Connection failed for worker {}: {}. Retrying in {} seconds...", worker, e, 5 * 2u64.pow(retries as u32));
                    tokio::time::sleep(Duration::from_secs(5 * 2u64.pow(retries as u32))).await;
                }
            }
        }
    }

    fn create_block_channel(
        send_channel: Sender<StratumLine>,
        miner_address: String,
        worker: String,
        last_stratum_id: Arc<AtomicU32>,
        share_stats: Arc<ShareStats>,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(100);
        let submitted_nonces = Arc::new(Mutex::new(HashMap::new()));
        let reported_nonces = Arc::new(Mutex::new(HashMap::<String, HashSet<u64>>::new()));

        let handle = tokio::spawn(async move {
            let mut stream = ReceiverStream::new(recv);

            while let Some(block_seed) = stream.next().await {
                let (nonce, job_id, _timestamp) = match &block_seed {
                    BlockSeed::PartialBlock { nonce, id, timestamp, .. } => (nonce, id, timestamp),
                    BlockSeed::FullBlock(_) => unreachable!(),
                };

                {
                    let mut reported = reported_nonces.lock().await;
                    let entry = reported.entry(job_id.clone()).or_insert_with(HashSet::new);
                    if !entry.insert(*nonce) {
                        continue;
                    }
                }

                let msg_id = last_stratum_id.fetch_add(1, Ordering::SeqCst);
                {
                    let mut nonces_guard = submitted_nonces.lock().await;
                    let nonces = nonces_guard.entry(job_id.clone()).or_insert(HashSet::new());
                    if !nonces.insert(*nonce) {
                        continue;
                    }

                    share_stats.shares_pending.insert(msg_id, (job_id.clone(), block_seed.clone()));
                    SHARE_TRACKER.record_share();
                }

                let nonce_str = format!("{:016x}", nonce);
                let address_name = format!("{}.{}", miner_address, worker);
                let line = StratumLine {
                    id: Some(msg_id),
                    payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                        MiningSubmit::MiningSubmitShort((
                            address_name,
                            job_id.into(),
                            nonce_str,
                        )),
                    )),
                    jsonrpc: None,
                };
                if let Err(e) = send_channel.send(line).await {
                    warn!("Failed to send share for worker {}: {}", worker, e);
                    share_stats.shares_pending.remove(&msg_id);
                    return Err(e.into());
                }
            }
            Ok(())
        });
        (send, handle)
    }

    async fn handle_message(&mut self, msg: StratumLine, miner: &mut MinerManager) -> Result<(), Error> {
        debug!("Raw StratumLine message: {:?}", msg);
        match msg.payload {
            StratumLinePayload::StratumError { error: StratumError(code, error_msg, _) } => {
                if let Some(id) = msg.id {
                    let job_id_str = self.shares_stats.shares_pending.remove(&id)
                        .and_then(|(_, (job_id, _))| Some(job_id))
                        .unwrap_or_else(|| {
                            warn!("No pending share for id={:?} for worker {}", id, self.worker);
                            String::new()
                        });

                    match code {
                        ErrorCode::Unknown => {
                            warn!("Got error code {}: {} for worker {}", code, error_msg, self.worker);
                            if !job_id_str.is_empty() {
                                warn!("Failed share (Job id: {}) for worker {}: {}", job_id_str, self.worker, error_msg);
                            }
                            Ok(())
                        }
                        ErrorCode::JobNotFound => {
                            self.shares_stats.stale.fetch_add(1, Ordering::SeqCst);
                            info!("Stale share detected (Job id: {}) for worker {}", job_id_str, self.worker);
                            Ok(())
                        }
                        ErrorCode::DuplicateShare => {
                            self.shares_stats.duplicate.fetch_add(1, Ordering::SeqCst);
                            let count = self.duplicate_count.fetch_add(1, Ordering::SeqCst) + 1;
                            info!("Duplicate share detected (Job id: {}) for worker {}, count={}", job_id_str, self.worker, count);
                            if count > 50 {
                                warn!("Excessive duplicate shares ({}) for worker {}. Disconnecting to request new extranonce.", count, self.worker);
                                return Err("Excessive duplicate shares, disconnecting".into());
                            }
                            Ok(())
                        }
                        ErrorCode::LowDifficultyShare => {
                            self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                            info!("Low difficulty share detected (Job id: {}) for worker {}", job_id_str, self.worker);
                            Ok(())
                        }
                        ErrorCode::Unauthorized => {
                            warn!("Got error code {}: {} for worker {}", code, error_msg, self.worker);
                            Err(error_msg.into())
                        }
                        ErrorCode::NotSubscribed => {
                            warn!("Got error code {}: {} for worker {}", code, error_msg, self.worker);
                            Err(error_msg.into())
                        }
                    }
                } else {
                    warn!("Error response with no ID: code={}, message={} for worker {}", code, error_msg, self.worker);
                    Ok(())
                }
            }
            StratumLinePayload::StratumResult { result } => {
                if let Some(id) = msg.id {
                    match result {
                        StratumResult::ShareResult { accepted: true, block_hash: pool_block_hash } => {
                            if let Some((job_id_str, mut block_seed)) = self.shares_stats.shares_pending.remove(&id).map(|(_, v)| v) {
                                self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                self.duplicate_count.store(0, Ordering::SeqCst);
                                SHARE_TRACKER.record_share();

                                if let Some(ref pool_hash) = pool_block_hash {
                                    if let BlockSeed::PartialBlock { hash: ref mut seed_hash, .. } = block_seed {
                                        *seed_hash = Some(pool_hash.clone());
                                    }
                                    if ACCEPTED_BLOCK_HASHES.insert(pool_hash.clone()) {
                                        let local_offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
                                        let utc = OffsetDateTime::from_unix_timestamp(block_seed.timestamp() as i64)
                                            .unwrap_or(OffsetDateTime::now_utc());
                                        let local_time = utc.to_offset(local_offset);
                                        let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");

                                        info!(
                                            "Block found: {} (Job: {}, Timestamp: {})",
                                            pool_hash,
                                            job_id_str,
                                            local_time.format(&format).unwrap_or("unknown".to_string())
                                        );
                                    }
                                }
                            }
                            Ok(())
                        }

                        StratumResult::ShareResult { accepted: false, .. } => {
                            if let Some((job_id_str, _)) = self.shares_stats.shares_pending.remove(&id).map(|(_, v)| v) {
                                info!("Share rejected (Job: {}) for worker {}", job_id_str, self.worker);
                            }
                            Ok(())
                        }

                        StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                            if let Some((job_id_str, block_seed)) = self.shares_stats.shares_pending.remove(&id).map(|(_, v)| v) {
                                self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                self.duplicate_count.store(0, Ordering::SeqCst);
                                SHARE_TRACKER.record_share();

                                if let BlockSeed::PartialBlock { hash: Some(h), timestamp, .. } = &block_seed {
                                    if ACCEPTED_BLOCK_HASHES.insert(h.clone()) {
                                        let local_offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
                                        let utc = OffsetDateTime::from_unix_timestamp(*timestamp as i64)
                                            .unwrap_or(OffsetDateTime::now_utc());
                                        let local_time = utc.to_offset(local_offset);
                                        let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");

                                        info!(
                                            "Block found: {} (Job: {}, Timestamp: {})",
                                            h,
                                            job_id_str,
                                            local_time.format(&format).unwrap_or("unknown".to_string())
                                        );
                                    }
                                }
                            }
                            Ok(())
                        }

                        StratumResult::Plain(Some(false)) | StratumResult::Eth((false, _)) => {
                            if let Some((job_id_str, _)) = self.shares_stats.shares_pending.remove(&id).map(|(_, v)| v) {
                                info!("Share rejected (Job: {}) for worker {}", job_id_str, self.worker);
                            }
                            Ok(())
                        }

                        StratumResult::Subscribe((subscriptions, extranonce, nonce_size)) => {
                            info!("Subscribed: subscriptions={:?}, extranonce={}, nonce_size={} for worker {}", subscriptions, extranonce, nonce_size, self.worker);
                            self.set_extranonce(&extranonce, &nonce_size)
                        }
                        StratumResult::SubscribeEth((true, protocol)) => {
                            info!("Subscribed to {} for worker {}", protocol, self.worker);
                            Ok(())
                        }
                        _ => {
                            warn!("Unexpected result: {:?} for worker {}", result, self.worker);
                            Ok(())
                        }
                    }
                } else {
                    warn!("Result response with no ID: result={:?} for worker {}", result, self.worker);
                    Ok(())
                }
            }
            StratumLinePayload::StratumCommand(command) => match command {
                StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((extranonce, nonce_size))) => {
                    info!("Set extranonce: {} with size {} for {}", extranonce, nonce_size, self.worker);
                    self.set_extranonce(&extranonce, &nonce_size)
                }
                StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlainEth(extranonce)) => {
                    let nonce_size = extranonce.len() as u32 / 2;
                    info!("Set extranonce (Eth): {} with size {} for {}", extranonce, nonce_size, self.worker);
                    self.set_extranonce(&extranonce, &nonce_size)
                }
                StratumCommand::MiningSetDifficulty((difficulty,)) => {
                    debug!("Received difficulty: {} for worker {}", difficulty, self.worker);
                    self.set_difficulty(&difficulty)
                }
                StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                    debug!(
                        "Processing MiningNotifyShort: job_id={}, header_hash={:016x}{:016x}{:016x}{:016x}, timestamp={}",
                        id,
                        header_hash.get(0).copied().unwrap_or(0),
                        header_hash.get(1).copied().unwrap_or(0),
                        header_hash.get(2).copied().unwrap_or(0),
                        header_hash.get(3).copied().unwrap_or(0),
                        timestamp
                    );
                    if header_hash.len() != 4 {
                        error!("Invalid header_hash length: {}, expected 4 for worker {}", header_hash.len(), self.worker);
                        return Err(format!("Invalid header_hash length: {}, expected 4", header_hash.len()).into());
                    }
                    self.block_template_ctr
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                        .unwrap();
                    miner
                        .process_block(Some(PartialBlock {
                            id,
                            header_hash,
                            timestamp,
                            nonce: self.nonce_fixed,
                            target: self.target_pool,
                            nonce_mask: self.nonce_mask,
                            nonce_fixed: self.nonce_fixed,
                            hash: None,
                        }))
                        .await
                }
                StratumCommand::MiningNotify(MiningNotify::MiningNotifySimple((id, hash))) => {
                    info!("Received mining.notify (simple): job_id={}, hash={} for worker {}", id, hash, self.worker);
                    self.block_template_ctr
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                        .unwrap();
                    if hash.len() != 80 {
                        return Err(format!("Invalid hash length: {}, expected 80 hex chars", hash.len()).into());
                    }
                    let hash_part = &hash[..64];
                    let timestamp_part = &hash[64..];
                    let hash_bytes = hex::decode(hash_part).map_err(|e| format!("Failed to decode hash: {}", e))?;
                    if hash_bytes.len() != 32 {
                        return Err(format!("Invalid hash length: {}, expected 32 bytes", hash_bytes.len()).into());
                    }
                    let mut bytes = [0u8; 32];
                    bytes.copy_from_slice(&hash_bytes);
                    let mut header_hash = [0u64; 4];
                    for i in 0..4 {
                        header_hash[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
                    }
                    let timestamp = u64::from_str_radix(timestamp_part, 16)
                        .map_err(|e| format!("Failed to parse timestamp: {}", e))?;
                    debug!(
                        "Processing job: job_id={}, header_hash={:016x}{:016x}{:016x}{:016x}, timestamp={}",
                        id, header_hash[0], header_hash[1], header_hash[2], header_hash[3], timestamp
                    );
                    miner
                        .process_block(Some(PartialBlock {
                            id,
                            header_hash,
                            timestamp,
                            nonce: self.nonce_fixed,
                            target: self.target_pool,
                            nonce_mask: self.nonce_mask,
                            nonce_fixed: self.nonce_fixed,
                            hash: None,
                        }))
                        .await
                }
                StratumCommand::MiningNotify(MiningNotify::MiningNotifyGeneric(params)) => {
                    warn!("Received unsupported mining.notify format for worker {}: {:?}", self.worker, params);
                    Ok(())
                }
                _ => {
                    warn!("Unexpected command: {:?} for worker {}", command, self.worker);
                    Ok(())
                }
            }
        }
    }

    fn set_difficulty(&mut self, difficulty: &f32) -> Result<(), Error> {
        if *difficulty <= 0.0 {
            warn!("Received invalid difficulty {} for worker {}, setting to default", difficulty, self.worker);
            self.target_pool = Uint256::default();
            return Ok(());
        }
        if *difficulty > 1_000_000_000.0 {
            warn!("Difficulty {} too large for worker {}, capping at 1e9", difficulty, self.worker);
            return Err("Difficulty too large".into());
        }
        let mut buf = [0u64, 0u64, 0u64, 0u64];
        let (mantissa, exponent, _) = difficulty.recip().integer_decode();
        let new_mantissa = mantissa * DIFFICULTY_1_TARGET.0;
        let new_exponent = (DIFFICULTY_1_TARGET.1 + exponent) as u64;
        let start = (new_exponent / 64) as usize;
        let remainder = new_exponent % 64;
        if start >= buf.len() {
            return Err("Difficulty exponent too large".into());
        }
        buf[start] = new_mantissa << remainder;
        if start < 3 {
            buf[start + 1] = new_mantissa >> (64 - remainder);
        } else if new_mantissa.leading_zeros() < remainder as u32 {
            return Err("Target is too big".into());
        }
        self.target_pool = Uint256::new(buf);
        debug!("Set difficulty: {} -> target_pool={:?} for worker {}", difficulty, self.target_pool, self.worker);
        Ok(())
    }

    fn set_extranonce(&mut self, extranonce: &str, nonce_size: &u32) -> Result<(), Error> {
        self.extranonce = Some(extranonce.to_string());
        self.nonce_fixed = u64::from_str_radix(extranonce, 16)?;
        let effective_nonce_size = if *nonce_size == 0 { extranonce.len() as u32 / 2 } else { *nonce_size };
        self.nonce_mask = if effective_nonce_size >= 8 { u64::MAX } else { (1 << (effective_nonce_size * 8)) - 1 };
        if self.nonce_fixed & !self.nonce_mask != 0 {
            warn!("Extranonce {} exceeds nonce_size {} for worker {}", extranonce, effective_nonce_size, self.worker);
            return Err("Extranonce exceeds nonce size".into());
        }
        debug!("Extranonce set: nonce_fixed={:016x}, nonce_mask={:016x} for worker {}", self.nonce_fixed, self.nonce_mask, self.worker);
        Ok(())
    }
}

impl Drop for StratumHandler {
    fn drop(&mut self) {
        self.block_handle.abort()
    }
}