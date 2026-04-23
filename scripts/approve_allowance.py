"""
One-time script: approve the Polymarket CTF Exchange to spend USDC from the proxy wallet.

This submits an on-chain transaction on Polygon — requires a small amount of MATIC for gas.
Run once, then restart the bot normally.
"""

import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

load_dotenv()

key = os.getenv("POLYMARKET_PRIVATE_KEY")
funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")
sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

if not key:
    sys.exit("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")

print(f"Key:     {key[:10]}...")
print(f"Funder:  {funder}")
print(f"Sig type: {sig_type}")
print()

client = ClobClient(
    host="https://clob.polymarket.com",
    key=key,
    chain_id=137,
    signature_type=sig_type,
    funder=funder,
)

print("Deriving API credentials...")
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)
print(f"API key: {creds.api_key[:8]}...")
print()

# Check current allowance first
print("Checking current USDC allowance...")
params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
result = client.get_balance_allowance(params=params)
print(f"Current state: {result}")
print()

# Submit approval transaction
print("Submitting approval transaction (requires MATIC for gas)...")
try:
    tx = client.update_balance_allowance(params=params)
    print(f"Approval tx submitted: {tx}")
    print()
    print("Done. Wait ~15 seconds for the transaction to confirm, then restart the bot.")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
