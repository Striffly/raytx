use std::{env, sync::Arc, time::Duration};

use anyhow::{anyhow, Result};
use jito_json_rpc_client::jsonrpc_client::rpc_client::RpcClient as JitoRpcClient;
use solana_client::{
    rpc_client::RpcClient,
    rpc_request::RpcError,
    client_error::ClientError,
    client_error::ClientErrorKind,
};
use solana_sdk::{
    instruction::Instruction,
    signature::Keypair,
    signer::Signer,
    system_transaction,
    transaction::{Transaction, TransactionError, VersionedTransaction},
    commitment_config::CommitmentConfig,
    signature::Signature,
};
use spl_token::ui_amount_to_amount;
use std::str::FromStr;
use tokio::time::{sleep, Instant};
use tracing::{error, info, warn};

use crate::{jito::{self, get_tip_account, get_tip_value, wait_for_bundle_confirmation}, get_commitment_config};

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub commitment: CommitmentConfig,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            commitment: CommitmentConfig::confirmed(),
        }
    }
}

fn is_retryable_error(error: &ClientError) -> bool {
    match error.kind() {
        // RPC errors that can be retried
        ClientErrorKind::RpcError(rpc_err) => matches!(
            rpc_err,
            RpcError::RpcResponseError { code, .. } if matches!(
                code,
                -32002 | // Too many requests (rate limit)
                -32004 | // Request timeout
                -32005 | // Node is behind
                -32009   // Node is unhealthy
            )
        ),

        // Transaction errors that indicate we can retry
        ClientErrorKind::TransactionError(tx_err) => matches!(
            tx_err,
            TransactionError::BlockhashNotFound | // Expired blockhash, retry with new one
            TransactionError::WouldExceedMaxBlockCostLimit |
            TransactionError::WouldExceedMaxAccountCostLimit |
            TransactionError::WouldExceedMaxVoteCostLimit |
            TransactionError::AccountInUse |
            TransactionError::AccountLoadedTwice
        ),

        // Network errors (generic) that justify a retry
        ClientErrorKind::Io(_) => true,  // Retry for I/O errors (network issues, temporary failures)

        // Reqwest errors related to network issues or timeout
        ClientErrorKind::Reqwest(_) => true,  // Retry on network or timeout issues

        // Custom error containing "timeout"
        ClientErrorKind::Custom(err) if err.contains("timeout") => true, // Retry on timeout

        // Other errors are not retryable
        _ => false,
    }
}

// prioritization fee = UNIT_PRICE * UNIT_LIMIT
fn get_unit_price() -> u64 {
    env::var("UNIT_PRICE")
        .ok()
        .and_then(|v| u64::from_str(&v).ok())
        .unwrap_or(20000)
}

fn get_unit_limit() -> u32 {
    env::var("UNIT_LIMIT")
        .ok()
        .and_then(|v| u32::from_str(&v).ok())
        .unwrap_or(200_000)
}

pub async fn send_transaction_with_retry(
    client: &RpcClient,
    keypair: &Keypair,
    instructions: &[Instruction],
    transaction: &mut Transaction,
    config: RetryConfig,
) -> Result<(Signature, u32)> {
    let mut attempt = 0;
    let mut last_error = None;

    while attempt < config.max_retries {
        attempt += 1;
        info!("Attempt {} to send transaction", attempt);

        match send_and_confirm_transaction(client, transaction, &config).await {
            Ok(signature) => {
                info!("Transaction successful on attempt {}", attempt);
                return Ok((signature, attempt));
            }
            Err(err) => {
                warn!("Attempt {} failed: {}", attempt, err);
                
                if let Some(client_error) = err.downcast_ref::<ClientError>() {
                    if !is_retryable_error(client_error) {
                        break;
                    }

                    match client_error.kind() {
                        ClientErrorKind::TransactionError(TransactionError::BlockhashNotFound) => {
                            match get_new_transaction(client, instructions, keypair) {
                                Ok(new_tx) => {
                                    *transaction = new_tx;
                                    info!("Updated transaction with new blockhash");
                                    continue;
                                }
                                Err(e) => {
                                    error!("Failed to update transaction with new blockhash: {}", e);
                                    break;
                                }
                            }
                        },
                        ClientErrorKind::RpcError(RpcError::RpcResponseError { code, .. }) if matches!(
                            code,
                            -32002
                        ) => {
                            sleep(config.base_delay * 2).await;
                            continue;
                        },
                        _ => {
                            sleep(config.base_delay * attempt as u32).await;
                        }
                    }
                } else {
                    break;
                }
                
                last_error = Some(err);
            }
        }
    }

    Err(anyhow!(
        "Transaction failed after {} attempts. Last error: {:?}",
        attempt,
        last_error.unwrap_or_else(|| anyhow!("Unknown error"))
    ))
}

async fn send_and_confirm_transaction(
    client: &RpcClient,
    transaction: &Transaction,
    config: &RetryConfig,
) -> Result<Signature> {
    let signature = client
        .send_transaction_with_config(
            transaction,
            solana_client::rpc_config::RpcSendTransactionConfig {
                skip_preflight: env::var("SKIP_PREFLIGHT").unwrap_or_else(|_| "true".to_string()).parse().unwrap_or(true),
                preflight_commitment: None,
                encoding: None,
                max_retries: None,
                min_context_slot: None,
            },
        )?;

    match client.confirm_transaction_with_commitment(&signature, config.commitment)
    {
        Ok(confirmation) => {
            if confirmation.value {
                Ok(signature)
            } else {
                Err(anyhow!("Transaction was not confirmed"))
            }
        }
        Err(err) => Err(anyhow!("Failed to confirm transaction: {}", err)),
    }
}

pub fn get_new_transaction(
    client: &RpcClient,
    instructions: &[Instruction],
    keypair: &Keypair,
) -> Result<Transaction> {
    let recent_blockhash = client.get_latest_blockhash()?;
    let txn = Transaction::new_signed_with_payer(
        &instructions,
        Some(&keypair.pubkey()),
        &vec![&*keypair],
        recent_blockhash,
    );
    Ok(txn)
}

pub async fn new_signed_and_send(
    client: &RpcClient,
    keypair: &Keypair,
    mut instructions: Vec<Instruction>,
    use_jito: bool
) -> Result<Vec<String>> {
    let unit_limit = get_unit_limit();
    let unit_price = get_unit_price();
    // If not using Jito, manually set the compute unit price and limit
    if !use_jito {
        let modify_compute_units =
            solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_limit(
                unit_limit,
            );
        let add_priority_fee =
            solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_price(
                unit_price,
            );
        instructions.insert(0, modify_compute_units);
        instructions.insert(1, add_priority_fee);
    }
    // Send initial transaction
    let txn = get_new_transaction(client, &instructions, keypair)?;
    let recent_blockhash = client.get_latest_blockhash()?;

    if env::var("TX_SIMULATE").ok() == Some("true".to_string()) {
        let simulate_result = client.simulate_transaction(&txn)?;
        if let Some(logs) = simulate_result.value.logs {
            for log in logs {
                info!("{}", log);
            }
        }
        return match simulate_result.value.err {
            Some(err) => Err(anyhow!("{}", err)),
            None => Ok(vec![]),
        };
    }

    let start_time = Instant::now();
    let mut txs = vec![];
    if use_jito {
        // Using Jito block engine
        let tip_account = get_tip_account().await?;
        // Jito tip, the upper limit is 0.1 SOL
        let mut tip = get_tip_value().await?;
        tip = tip.min(0.1);
        let tip_lamports = ui_amount_to_amount(tip, spl_token::native_mint::DECIMALS);
        info!(
            "tip account: {}, tip(sol): {}, lamports: {}",
            tip_account, tip, tip_lamports
        );

        let jito_client = Arc::new(JitoRpcClient::new(format!(
            "{}/api/v1/bundles",
            jito::BLOCK_ENGINE_URL.to_string()
        )));
        // Create bundle with transaction and tip
        let mut bundle: Vec<VersionedTransaction> = vec![];
        bundle.push(VersionedTransaction::from(txn));
        bundle.push(VersionedTransaction::from(system_transaction::transfer(
            &keypair,
            &tip_account,
            tip_lamports,
            recent_blockhash,
        )));
        let bundle_id = jito_client.send_bundle(&bundle).await?;
        info!("bundle_id: {}", bundle_id);

        txs = wait_for_bundle_confirmation(
            move |id: String| {
                let client = Arc::clone(&jito_client);
                async move {
                    let response = client.get_bundle_statuses(&[id]).await;
                    let statuses = response.inspect_err(|err| {
                        error!("Error fetching bundle status: {:?}", err);
                    })?;
                    Ok(statuses.value)
                }
            },
            bundle_id,
            Duration::from_millis(1000),
            Duration::from_secs(10),
        )
        .await?;
    } else {
        let config = RetryConfig {
            max_retries: env::var("MAX_RETRIES").unwrap_or_else(|_| "3".to_string()).parse().unwrap_or(3),
            base_delay: Duration::from_millis(500),
            commitment: get_commitment_config()?,
        };
        
        let mut txn = get_new_transaction(client, &instructions, keypair)?;
        let (signature, attempts) = send_transaction_with_retry(
            client, 
            keypair, 
            &instructions, 
            &mut txn,
            config
        ).await?;
        info!(
            "Transaction successful after {} attempts. Signature: {}",
            attempts, signature
        );
        txs.push(signature.to_string());
    }

    info!("tx elapsed: {:?}", start_time.elapsed());
    Ok(txs)
}
