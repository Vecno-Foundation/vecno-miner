use bytes::BytesMut;
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error as StdError;
use std::fmt::{self, Display, Formatter};
use std::io;
use tokio_util::codec::{Decoder, Encoder, LinesCodec, LinesCodecError};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorCode {
    Unknown = 20,
    JobNotFound = 21,
    DuplicateShare = 22,
    LowDifficultyShare = 23,
    Unauthorized = 24,
    NotSubscribed = 25,
}

impl Display for ErrorCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::Unknown => write!(f, "Unknown"),
            ErrorCode::JobNotFound => write!(f, "JobNotFound"),
            ErrorCode::DuplicateShare => write!(f, "DuplicateShare"),
            ErrorCode::LowDifficultyShare => write!(f, "LowDifficultyShare"),
            ErrorCode::Unauthorized => write!(f, "Unauthorized"),
            ErrorCode::NotSubscribed => write!(f, "NotSubscribed"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StratumError(pub ErrorCode, pub String, #[serde(default)] pub Option<Value>);

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningNotify {
    MiningNotifyShort((String, [u64; 4], u64)),
    MiningNotifySimple((String, String)),
    MiningNotifyGeneric(Value),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubmit {
    MiningSubmitShort((String, String, String)),
    MiningSubmitLong((String, String, String, String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SetExtranonce {
    SetExtranoncePlain((String, u32)),
    SetExtranoncePlainEth(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubscribe {
    MiningSubscribeDefault((String,)),
    MiningSubscribeOptions((String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum StratumResult {
    Plain(Option<bool>),
    Eth((bool, String)),
    Subscribe((Vec<(String, String)>, String, u32)),
    SubscribeEth((bool, String)),
    // NEW: Support extended share result format
    #[serde(rename_all = "snake_case")]
    ShareResult {
        accepted: bool,
        #[serde(rename = "block_hash")]
        block_hash: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum StratumLinePayload {
    StratumCommand(StratumCommand),
    StratumResult { result: StratumResult },
    StratumError { error: StratumError },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "method", content = "params")]
pub enum StratumCommand {
    #[serde(rename = "mining.set_extranonce", alias = "set_extranonce")]
    SetExtranonce(SetExtranonce),
    #[serde(rename = "mining.set_difficulty")]
    MiningSetDifficulty((f32,)),
    #[serde(rename = "mining.notify")]
    MiningNotify(MiningNotify),
    #[serde(rename = "mining.subscribe")]
    Subscribe(MiningSubscribe),
    #[serde(rename = "mining.authorize")]
    Authorize((String, String)),
    #[serde(rename = "mining.submit")]
    MiningSubmit(MiningSubmit),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StratumLine {
    pub id: Option<u32>,
    #[serde(flatten)]
    pub payload: StratumLinePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jsonrpc: Option<String>,
}

#[derive(Debug)]
pub enum NewLineJsonCodecError {
    JsonParseError(String),
    #[allow(dead_code)]
    JsonEncodeError,
    LineSplitError,
    LineEncodeError,
    Io(()),
    InvalidHeaderHash(String),
}

impl fmt::Display for NewLineJsonCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonParseError(s) => write!(f, "JSON parse error: {}", s),
            Self::JsonEncodeError => write!(f, "JSON encode error"),
            Self::LineSplitError => write!(f, "Line split error"),
            Self::LineEncodeError => write!(f, "Line encode error"),
            Self::Io(_) => write!(f, "IO error"),
            Self::InvalidHeaderHash(s) => write!(f, "Invalid header_hash: {}", s),
        }
    }
}

impl StdError for NewLineJsonCodecError {}

impl From<io::Error> for NewLineJsonCodecError {
    fn from(_e: io::Error) -> Self {
        NewLineJsonCodecError::Io(())
    }
}

impl From<serde_json::Error> for NewLineJsonCodecError {
    fn from(e: serde_json::Error) -> Self {
        NewLineJsonCodecError::JsonParseError(e.to_string())
    }
}

impl From<LinesCodecError> for NewLineJsonCodecError {
    fn from(_e: LinesCodecError) -> Self {
        NewLineJsonCodecError::LineSplitError
    }
}

pub struct NewLineJsonCodec {
    lines_codec: LinesCodec,
}

impl NewLineJsonCodec {
    pub fn new() -> Self {
        Self { lines_codec: LinesCodec::new() }
    }
}

impl Decoder for NewLineJsonCodec {
    type Item = StratumLine;
    type Error = NewLineJsonCodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode(src)? {
            Some(s) => {
                debug!("Attempting to parse StratumLine: {}", s);
                let raw: Value = serde_json::from_str(&s)?;
                let id = raw.get("id").and_then(|id| id.as_u64().map(|i| i as u32));
                let jsonrpc = raw.get("jsonrpc").and_then(|j| j.as_str().map(String::from));

                if let Some(error) = raw.get("error") {
                    if let Some(error_array) = error.as_array() {
                        if error_array.len() >= 2 {
                            let code = error_array[0].as_u64().unwrap_or(20) as u8;
                            let message = error_array[1].as_str().unwrap_or("Unknown error").to_string();
                            let extra = error_array.get(2).cloned();
                            let error_code = match code {
                                20 => ErrorCode::Unknown,
                                21 => ErrorCode::JobNotFound,
                                22 => ErrorCode::DuplicateShare,
                                23 => ErrorCode::LowDifficultyShare,
                                24 => ErrorCode::Unauthorized,
                                25 => ErrorCode::NotSubscribed,
                                _ => ErrorCode::Unknown,
                            };
                            let line = StratumLine {
                                id,
                                payload: StratumLinePayload::StratumError {
                                    error: StratumError(error_code, message, extra),
                                },
                                jsonrpc,
                            };
                            debug!("Successfully parsed error StratumLine: {:?}", line);
                            return Ok(Some(line));
                        }
                    }
                }

                if raw.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                    if let Some(params) = raw.get("params").and_then(|p| p.as_array()) {
                        if params.len() >= 3 && params[1].is_array() {
                            let header_hash = params[1].as_array().unwrap();
                            if header_hash.len() != 4 || !header_hash.iter().all(|v| v.is_u64()) {
                                debug!("Invalid header_hash in mining.notify: {:?}", params[1]);
                                return Err(NewLineJsonCodecError::InvalidHeaderHash(format!(
                                    "Expected array of 4 u64 values, got {:?}", params[1]
                                )));
                            }
                        }
                    }
                }

                let line: StratumLine = match serde_json::from_str(&s) {
                    Ok(line) => line,
                    Err(e) => {
                        if raw.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                            let line = StratumLine {
                                id,
                                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningNotify(
                                    MiningNotify::MiningNotifyGeneric(raw["params"].clone()),
                                )),
                                jsonrpc,
                            };
                            debug!("Parsed as generic mining.notify: {:?}", line);
                            return Ok(Some(line));
                        }
                        return Err(NewLineJsonCodecError::JsonParseError(format!(
                            "Failed to parse StratumLine: {}. Raw JSON: {}",
                            e, s
                        )));
                    }
                };
                debug!("Successfully parsed StratumLine: {:?}", line);
                Ok(Some(line))
            }
            None => Ok(None),
        }
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode_eof(buf)? {
            Some(s) => {
                debug!("Attempting to parse StratumLine (EOF): {}", s);
                let raw: Value = serde_json::from_str(&s)?;
                let id = raw.get("id").and_then(|id| id.as_u64().map(|i| i as u32));
                let jsonrpc = raw.get("jsonrpc").and_then(|j| j.as_str().map(String::from));

                if let Some(error) = raw.get("error") {
                    if let Some(error_array) = error.as_array() {
                        if error_array.len() >= 2 {
                            let code = error_array[0].as_u64().unwrap_or(20) as u8;
                            let message = error_array[1].as_str().unwrap_or("Unknown error").to_string();
                            let extra = error_array.get(2).cloned();
                            let error_code = match code {
                                20 => ErrorCode::Unknown,
                                21 => ErrorCode::JobNotFound,
                                22 => ErrorCode::DuplicateShare,
                                23 => ErrorCode::LowDifficultyShare,
                                24 => ErrorCode::Unauthorized,
                                25 => ErrorCode::NotSubscribed,
                                _ => ErrorCode::Unknown,
                            };
                            let line = StratumLine {
                                id,
                                payload: StratumLinePayload::StratumError {
                                    error: StratumError(error_code, message, extra),
                                },
                                jsonrpc,
                            };
                            debug!("Successfully parsed error StratumLine (EOF): {:?}", line);
                            return Ok(Some(line));
                        }
                    }
                }

                if raw.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                    if let Some(params) = raw.get("params").and_then(|p| p.as_array()) {
                        if params.len() >= 3 && params[1].is_array() {
                            let header_hash = params[1].as_array().unwrap();
                            if header_hash.len() != 4 || !header_hash.iter().all(|v| v.is_u64()) {
                                debug!("Invalid header_hash in mining.notify (EOF): {:?}", params[1]);
                                return Err(NewLineJsonCodecError::InvalidHeaderHash(format!(
                                    "Expected array of 4 u64 values, got {:?}", params[1]
                                )));
                            }
                        }
                    }
                }

                let line: StratumLine = match serde_json::from_str(&s) {
                    Ok(line) => line,
                    Err(e) => {
                        if raw.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                            let line = StratumLine {
                                id,
                                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningNotify(
                                    MiningNotify::MiningNotifyGeneric(raw["params"].clone()),
                                )),
                                jsonrpc,
                            };
                            debug!("Parsed as generic mining.notify: {:?}", line);
                            return Ok(Some(line));
                        }
                        return Err(NewLineJsonCodecError::JsonParseError(format!(
                            "Failed to parse StratumLine: {}. Raw JSON: {}",
                            e, s
                        )));
                    }
                };
                debug!("Successfully parsed StratumLine (EOF): {:?}", line);
                Ok(Some(line))
            }
            None => Ok(None),
        }
    }
}

impl Encoder<StratumLine> for NewLineJsonCodec {
    type Error = NewLineJsonCodecError;

    fn encode(&mut self, item: StratumLine, dst: &mut BytesMut) -> Result<(), Self::Error> {
        let json = serde_json::to_string(&item)?;
        debug!("Encoding StratumLine: {}", json);
        self.lines_codec.encode(json, dst).map_err(|_| NewLineJsonCodecError::LineEncodeError)
    }
}

impl Default for NewLineJsonCodec {
    fn default() -> Self {
        Self::new()
    }
}