use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::vecnod_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockTemplateRequestMessage, GetInfoRequestMessage, VecnodMessage, NotifyNewBlockTemplateRequestMessage,
};
use crate::{miner::MinerManager, Error};
use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{rng, RngCore};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::{self, error::SendError, Sender};
use tokio::task::JoinHandle;
use tokio::time::{interval, Duration};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};
use std::sync::atomic::AtomicU64;

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"));
type BlockHandle = JoinHandle<Result<(), PollSendError<VecnodMessage>>>;

#[allow(dead_code)]
pub struct VecnodHandler {
    client: RpcClient<TonicChannel>,
    pub send_channel: Sender<VecnodMessage>,
    stream: Streaming<VecnodMessage>,
    miner_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,
    successful_blocks: Arc<AtomicU64>,
}

#[async_trait(?Send)]
impl Client for VecnodHandler {
    async fn register(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        let successful_blocks = Arc::clone(&self.successful_blocks);
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                let count = successful_blocks.swap(0, Ordering::SeqCst);
                if count > 0 {
                    info!("Successfully mined {} blocks in the past 10 seconds", count);
                }
            }
        });

        while let Some(msg) = self.stream.message().await? {
            match msg.payload {
                Some(payload) => self.handle_message(payload, miner).await?,
                None => warn!("vecno message payload is empty"),
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl VecnodHandler {
    pub async fn connect<D>(
        address: D,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
    ) -> Result<Box<Self>, Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<Error>,
    {
        let mut client = RpcClient::connect(address).await?;
        let (send_channel, recv) = mpsc::channel(2);
        send_channel.send(GetInfoRequestMessage {}.into()).await?;
        let stream = client.message_stream(ReceiverStream::new(recv)).await?.into_inner();
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone());
        Ok(Box::new(Self {
            client,
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((rng().next_u64() % 10_000u64) as u16))),
            block_channel,
            block_handle,
            successful_blocks: Arc::new(AtomicU64::new(0)),
        }))
    }

    fn create_block_channel(send_channel: Sender<VecnodMessage>) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(1);
        (
            send,
            tokio::spawn(async move {
                ReceiverStream::new(recv)
                    .map(|block_seed| match block_seed {
                        FullBlock(block) => VecnodMessage::submit_block(*block),
                        PartialBlock { .. } => unreachable!("All blocks sent here should have arrived from here"),
                    })
                    .map(Ok)
                    .forward(PollSender::new(send_channel))
                    .await
            }),
        )
    }

    async fn client_send(&self, msg: impl Into<VecnodMessage>) -> Result<(), SendError<VecnodMessage>> {
        self.send_channel.send(msg.into()).await
    }

    async fn client_get_block_template(&mut self) -> Result<(), SendError<VecnodMessage>> {
        self.block_template_ctr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000)).unwrap();
        self.client_send(GetBlockTemplateRequestMessage { 
            pay_address: self.miner_address.clone(), 
            extra_data: EXTRA_DATA.into() 
        }).await
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            Payload::BlockAddedNotification(_) => self.client_get_block_template().await?,
            Payload::NewBlockTemplateNotification(_) => self.client_get_block_template().await?,
            Payload::GetBlockTemplateResponse(template) => match (template.block, template.is_synced, template.error) {
                (Some(b), true, None) => miner.process_block(Some(FullBlock(Box::new(b)))).await?,
                (Some(b), false, None) if self.mine_when_not_synced => {
                    miner.process_block(Some(FullBlock(Box::new(b)))).await?
                }
                (_, false, None) => miner.process_block(None).await?,
                (_, _, Some(e)) => {
                    return Err(format!("GetTemplate returned with an error: {:?}", e).into());
                }
                (None, true, None) => error!("No block and No Error!"),
            },
            Payload::SubmitBlockResponse(res) => match res.error {
                None => {
                    self.successful_blocks.fetch_add(1, Ordering::SeqCst);
                }
                Some(e) => warn!("Failed submitting block: {:?}", e),
            },
            Payload::GetBlockResponse(msg) => {
                if let Some(e) = msg.error {
                    return Err(e.message.into());
                } else {
                    info!("Get block response: {:?}", msg);
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("VecnodMessage version: {}", info.server_version);
                self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?;
                self.client_get_block_template().await?;
            }
            Payload::NotifyNewBlockTemplateResponse(res) => match res.error {
                None => info!("Registered for new template notifications"),
                Some(e) => error!("Failed registering for new template notifications: {:?}", e),
            },
            Payload::NotifyBlockAddedResponse(res) => match res.error {
                None => info!("Registered for block notifications (upgrade your Vecnod for better experience)"),
                Some(e) => error!("Failed registering for block notifications: {:?}", e),
            },
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}

impl Drop for VecnodHandler {
    fn drop(&mut self) {
        self.block_handle.abort();
    }
}