#!/usr/bin/env python3
"""
poc_sliding_window_feasibility.py — KL-NOVEL-BDB-IV-REUSE-DETECTED
Aggressive Overlapping-Window Collision Scanner

Companion to poc_iv_reuse_exploit.py and poc_iv_reuse_feasibility.py.

The standard aligned-scan treats the wallet.dat as a flat stream of 16-byte
blocks aligned to 16-byte boundaries (offset 0, 16, 32, …).  This misses
ciphertext collisions that occur at non-aligned offsets due to BDB record
packing, page boundaries, or record-header variability.

This scanner slides a 16-byte window one byte at a time across the entire
file, finding ALL repeated 16-byte patterns regardless of alignment.  It
then cross-references every collision with the mkey/ckey encrypted blobs
to identify exploitable IV-reuse that the page-aligned scan missed.

Offline operation — no bitcoind required.
Python 3 standard library only — no external dependencies.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
BLOCK_SIZE = 16
FINDING_ID = "KL-NOVEL-BDB-IV-REUSE-DETECTED"

N_SECP256K1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

CKEY_MARKER = b'\x05\x63\x6b\x65\x79'   # \x05ckey
MKEY_MARKER = b'\x04mkey'

BDB_STRUCTURAL_PREFIXES = [
    b'\x00' * 16,
    b'\x61\x15\x06\x00',  # BDB magic (little-endian)
    b'\x00\x06\x15\x61',  # BDB magic (big-endian)
]


# ═══════════════════════════════════════════════════════════════════
# BDB wallet parser (reused from poc_iv_reuse_feasibility.py)
# ═══════════════════════════════════════════════════════════════════
def _read_compact_size(data: bytes, off: int) -> Tuple[int, int]:
    if off >= len(data):
        return 0, 0
    b = data[off]
    if b < 0xfd:
        return b, 1
    if b == 0xfd and off + 3 <= len(data):
        return struct.unpack_from('<H', data, off + 1)[0], 3
    if b == 0xfe and off + 5 <= len(data):
        return struct.unpack_from('<I', data, off + 1)[0], 5
    if off + 9 <= len(data):
        return struct.unpack_from('<Q', data, off + 1)[0], 9
    return 0, 1


class MKeyRecord:
    __slots__ = ('encrypted_key', 'salt', 'method', 'iterations', 'file_offset')

    def __init__(self):
        self.encrypted_key: bytes = b''
        self.salt: bytes = b''
        self.method: int = 0
        self.iterations: int = 0
        self.file_offset: int = -1


class CKeyRecord:
    __slots__ = ('pubkey', 'encrypted_privkey', 'file_offset')

    def __init__(self):
        self.pubkey: bytes = b''
        self.encrypted_privkey: bytes = b''
        self.file_offset: int = -1


def parse_wallet_bdb(path: str) -> Dict[str, Any]:
    """Parse wallet.dat BDB file.  Extract mkey and all ckey records."""
    data = open(path, 'rb').read()

    # Detect page size
    page_size = 4096
    if len(data) >= 24:
        ps_le = struct.unpack_from('<I', data, 20)[0]
        if 512 <= ps_le <= 65536 and (ps_le & (ps_le - 1)) == 0:
            page_size = ps_le
        else:
            ps_be = struct.unpack_from('>I', data, 20)[0]
            if 512 <= ps_be <= 65536 and (ps_be & (ps_be - 1)) == 0:
                page_size = ps_be

    result: Dict[str, Any] = {
        'raw': data,
        'mkey': None,
        'ckeys': [],
        'page_size': page_size,
    }

    # Scan BDB leaf pages
    for pg in range(len(data) // page_size):
        base = pg * page_size
        if base + 26 > len(data):
            continue
        if data[base + 25] not in (5, 6):
            continue

        num_entries = struct.unpack_from('<H', data, base + 20)[0]
        idx_off = base + 26
        indices: List[int] = []
        for _ in range(min(num_entries, (page_size - 26) // 2)):
            if idx_off + 2 > base + page_size:
                break
            indices.append(struct.unpack_from('<H', data, idx_off)[0])
            idx_off += 2

        for i in range(0, len(indices) - 1, 2):
            koff = base + indices[i]
            voff = base + (indices[i + 1] if i + 1 < len(indices) else 0)
            if voff == base or koff + 3 >= len(data) or voff + 3 >= len(data):
                continue
            try:
                klen = struct.unpack_from('<H', data, koff)[0]
                vlen = struct.unpack_from('<H', data, voff)[0]
                if klen > page_size or vlen > page_size:
                    continue
                kd = data[koff + 3:koff + 3 + klen]
                vd = data[voff + 3:voff + 3 + vlen]
            except Exception:
                continue
            if not kd:
                continue

            tl, tn = _read_compact_size(kd, 0)
            if tn == 0 or tl + tn > len(kd):
                continue
            try:
                ts = kd[tn:tn + tl].decode('ascii').lower()
            except Exception:
                continue

            if ts == 'mkey' and vd:
                m = MKeyRecord()
                m.file_offset = voff
                o2 = 0
                el, en = _read_compact_size(vd, o2); o2 += en
                if o2 + el <= len(vd):
                    m.encrypted_key = vd[o2:o2 + el]; o2 += el
                sl, sn = _read_compact_size(vd, o2); o2 += sn
                if o2 + sl <= len(vd):
                    m.salt = vd[o2:o2 + sl]; o2 += sl
                if o2 + 4 <= len(vd):
                    m.method = struct.unpack_from('<I', vd, o2)[0]; o2 += 4
                if o2 + 4 <= len(vd):
                    m.iterations = struct.unpack_from('<I', vd, o2)[0]
                if len(m.encrypted_key) in (32, 48, 64):
                    result['mkey'] = m

            elif ts == 'ckey' and vd:
                ck = CKeyRecord()
                ck.file_offset = voff
                rest = kd[tn + tl:]
                pl, pn = _read_compact_size(rest, 0)
                if pn > 0 and pl + pn <= len(rest):
                    ck.pubkey = rest[pn:pn + pl]
                el2, en2 = _read_compact_size(vd, 0)
                if en2 > 0 and el2 + en2 <= len(vd):
                    ck.encrypted_privkey = vd[en2:en2 + el2]
                if ck.encrypted_privkey:
                    result['ckeys'].append(ck)

    # Fallback: raw marker scan
    if result['mkey'] is None:
        pos = 0
        while True:
            idx = data.find(MKEY_MARKER, pos)
            if idx == -1:
                break
            m = MKeyRecord()
            m.file_offset = idx
            scan = data[idx + len(MKEY_MARKER):idx + len(MKEY_MARKER) + 256]
            for so in range(len(scan) - 48):
                el, en = _read_compact_size(scan, so)
                if en > 0 and el in (32, 48, 64) and so + en + el + 16 <= len(scan):
                    m.encrypted_key = scan[so + en:so + en + el]
                    salt_off = so + en + el
                    sl, sn = _read_compact_size(scan, salt_off)
                    if sn > 0 and sl > 0 and salt_off + sn + sl <= len(scan):
                        m.salt = scan[salt_off + sn:salt_off + sn + sl]
                    result['mkey'] = m
                    break
            pos = idx + 1
            if result['mkey']:
                break

    if not result['ckeys']:
        pos = 0
        while True:
            idx = data.find(CKEY_MARKER, pos)
            if idx == -1:
                break
            ck = CKeyRecord()
            ck.file_offset = idx
            scan = data[idx + len(CKEY_MARKER):idx + len(CKEY_MARKER) + 256]
            pl, pn = _read_compact_size(scan, 0)
            if pn > 0 and 33 <= pl <= 65 and pn + pl < len(scan):
                ck.pubkey = scan[pn:pn + pl]
                rest = scan[pn + pl:]
                el, en = _read_compact_size(rest, 0)
                if en > 0 and el in (32, 48, 64) and en + el <= len(rest):
                    ck.encrypted_privkey = rest[en:en + el]
                    result['ckeys'].append(ck)
            pos = idx + 1

    return result


# ═══════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════
def _shannon_entropy(block: bytes) -> float:
    """Compute Shannon entropy (bits) of a byte sequence."""
    if not block:
        return 0.0
    freq: Dict[int, int] = {}
    for b in block:
        freq[b] = freq.get(b, 0) + 1
    n = len(block)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _is_bdb_structural(block: bytes) -> bool:
    """Heuristic: is this block likely BDB page metadata?"""
    if block == b'\x00' * BLOCK_SIZE:
        return True
    for prefix in BDB_STRUCTURAL_PREFIXES:
        if block.startswith(prefix):
            return True
    if len(set(block)) <= 2:
        return True
    return False


def _ckey_iv(pubkey: bytes) -> bytes:
    """Derive the AES-CBC IV for a ckey record: SHA256(SHA256(pubkey))[:16]."""
    return hashlib.sha256(hashlib.sha256(pubkey).digest()).digest()[:BLOCK_SIZE]


def _build_aligned_block_counter(data: bytes) -> collections.Counter:
    """Build a frequency counter of all non-zero 16-byte aligned blocks."""
    counter: collections.Counter = collections.Counter()
    zero_block = b'\x00' * BLOCK_SIZE
    for i in range(len(data) // BLOCK_SIZE):
        blk = data[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        if blk != zero_block:
            counter[blk] += 1
    return counter


# ═══════════════════════════════════════════════════════════════════
# Section 1 — Sliding-window collision detector
# ═══════════════════════════════════════════════════════════════════
def scan_sliding_window(wallet_path: str) -> List[Tuple[str, List[int]]]:
    """Slide a 16-byte window one byte at a time across the entire file.

    For every offset i from 0 to len(data)-16, extract data[i:i+16].
    Build a frequency map of all such slices, ignoring all-zero blocks.
    Return a list of (block_hex, offsets_list) for every repeated block
    (count >= 2), sorted by number of occurrences descending.
    """
    data = open(wallet_path, 'rb').read()
    zero_block = b'\x00' * BLOCK_SIZE

    # Map: 16-byte block -> list of byte offsets
    block_offsets: Dict[bytes, List[int]] = {}

    total_slices = len(data) - BLOCK_SIZE + 1
    if total_slices <= 0:
        return []

    for i in range(total_slices):
        slc = data[i:i + BLOCK_SIZE]
        if slc == zero_block:
            continue
        if slc in block_offsets:
            block_offsets[slc].append(i)
        else:
            block_offsets[slc] = [i]

    # Filter to repeated blocks only (count >= 2)
    repeated = [
        (blk.hex(), offsets)
        for blk, offsets in block_offsets.items()
        if len(offsets) >= 2
    ]

    # Sort by number of occurrences descending
    repeated.sort(key=lambda x: len(x[1]), reverse=True)

    return repeated


def _scan_sliding_window_full(data: bytes) -> Tuple[Dict[bytes, List[int]], int]:
    """Internal: full sliding-window scan returning raw bytes keys.

    Returns (block_offsets_dict, total_non_zero_slices).
    """
    zero_block = b'\x00' * BLOCK_SIZE
    block_offsets: Dict[bytes, List[int]] = {}
    total_slices = len(data) - BLOCK_SIZE + 1
    non_zero_count = 0

    if total_slices <= 0:
        return {}, 0

    for i in range(total_slices):
        slc = data[i:i + BLOCK_SIZE]
        if slc == zero_block:
            continue
        non_zero_count += 1
        if slc in block_offsets:
            block_offsets[slc].append(i)
        else:
            block_offsets[slc] = [i]

    return block_offsets, non_zero_count


# ═══════════════════════════════════════════════════════════════════
# Section 2 — Key material boundary helpers
# ═══════════════════════════════════════════════════════════════════
def _key_material_ranges(wallet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a list of byte-offset ranges for all key material blobs.

    Each entry: {'type': 'mkey'|'ckey', 'start': int, 'end': int,
                 'index': int (for ckey), 'record': MKeyRecord|CKeyRecord}
    """
    ranges = []
    data = wallet['raw']
    mkey = wallet.get('mkey')
    ckeys = wallet.get('ckeys', [])

    if mkey and mkey.encrypted_key and mkey.file_offset >= 0:
        # The encrypted_key blob starts somewhere after file_offset.
        # We need to find the exact byte position of the encrypted_key
        # within the file.  The file_offset points to the BDB value slot.
        # Scan forward from file_offset for the encrypted_key bytes.
        enc = mkey.encrypted_key
        search_start = max(0, mkey.file_offset)
        search_end = min(len(data), search_start + 512)
        pos = data.find(enc, search_start, search_end)
        if pos != -1:
            ranges.append({
                'type': 'mkey',
                'start': pos,
                'end': pos + len(enc),
                'index': 0,
                'record': mkey,
            })
        else:
            # Fallback: search entire file
            pos = data.find(enc)
            if pos != -1:
                ranges.append({
                    'type': 'mkey',
                    'start': pos,
                    'end': pos + len(enc),
                    'index': 0,
                    'record': mkey,
                })

    for ci, ck in enumerate(ckeys):
        if not ck.encrypted_privkey or ck.file_offset < 0:
            continue
        enc = ck.encrypted_privkey
        search_start = max(0, ck.file_offset)
        search_end = min(len(data), search_start + 512)
        pos = data.find(enc, search_start, search_end)
        if pos != -1:
            ranges.append({
                'type': 'ckey',
                'start': pos,
                'end': pos + len(enc),
                'index': ci,
                'record': ck,
            })
        else:
            pos = data.find(enc)
            if pos != -1:
                ranges.append({
                    'type': 'ckey',
                    'start': pos,
                    'end': pos + len(enc),
                    'index': ci,
                    'record': ck,
                })

    return ranges


def _offset_touches_range(offset: int, rng: Dict[str, Any]) -> bool:
    """Check if a 16-byte slice at `offset` overlaps with a key material range."""
    slice_start = offset
    slice_end = offset + BLOCK_SIZE
    return slice_start < rng['end'] and slice_end > rng['start']


def _classify_offset(offset: int,
                     key_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return all key-material ranges that a 16-byte slice at `offset` touches."""
    return [r for r in key_ranges if _offset_touches_range(offset, r)]


# ═══════════════════════════════════════════════════════════════════
# Section 3 — Cross-reference sliding-window collisions with key material
# ═══════════════════════════════════════════════════════════════════
def cross_reference_collisions(
    sliding_collisions: Dict[bytes, List[int]],
    wallet: Dict[str, Any],
) -> Dict[str, Any]:
    """For each repeated sliding-window block, determine whether any of its
    offsets fall within the boundaries of the mkey or any ckey encrypted blob.

    Flags:
      - Collisions where ≥2 occurrences touch key material (mkey or ckey)
      - Collisions where one occurrence is in a key record and another in
        BDB metadata (attacker may know the metadata plaintext)
      - Prioritises collisions across different encrypted records
        (mkey-ckey, ckey-ckey with different indices)

    Returns a dict with categorised collision lists and summary statistics.
    """
    key_ranges = _key_material_ranges(wallet)

    result: Dict[str, Any] = {
        'mkey_ckey_collisions': [],     # block in both mkey and a ckey
        'ckey_ckey_collisions': [],     # block in two different ckeys
        'mkey_metadata_collisions': [], # block in mkey and in non-key area
        'ckey_metadata_collisions': [], # block in ckey and in non-key area
        'key_internal_collisions': [],  # block repeated within same key blob
        'total_key_touching': 0,
        'total_cross_record': 0,
    }

    for blk, offsets in sliding_collisions.items():
        if len(offsets) < 2:
            continue

        # Classify each offset
        offset_classifications = []
        for off in offsets:
            touches = _classify_offset(off, key_ranges)
            offset_classifications.append({
                'offset': off,
                'touches': touches,
            })

        # Determine which key records are involved
        mkey_offsets = []
        ckey_offsets: Dict[int, List[int]] = {}  # ckey_index -> [offsets]
        metadata_offsets = []

        for oc in offset_classifications:
            if not oc['touches']:
                metadata_offsets.append(oc['offset'])
            else:
                for t in oc['touches']:
                    if t['type'] == 'mkey':
                        mkey_offsets.append(oc['offset'])
                    elif t['type'] == 'ckey':
                        ci = t['index']
                        if ci not in ckey_offsets:
                            ckey_offsets[ci] = []
                        ckey_offsets[ci].append(oc['offset'])

        has_mkey = len(mkey_offsets) > 0
        ckey_indices = list(ckey_offsets.keys())
        has_ckey = len(ckey_indices) > 0
        has_metadata = len(metadata_offsets) > 0

        collision_entry = {
            'block_hex': blk.hex(),
            'total_occurrences': len(offsets),
            'all_offsets': offsets,
            'mkey_offsets': mkey_offsets,
            'ckey_offsets': dict(ckey_offsets),
            'metadata_offsets': metadata_offsets,
            'entropy': round(_shannon_entropy(blk), 4),
        }

        touches_key = has_mkey or has_ckey
        if touches_key:
            result['total_key_touching'] += 1

        # Categorise
        if has_mkey and has_ckey:
            collision_entry['priority'] = 'CRITICAL'
            collision_entry['category'] = 'mkey-ckey'
            result['mkey_ckey_collisions'].append(collision_entry)
            result['total_cross_record'] += 1

        if len(ckey_indices) >= 2:
            collision_entry_ck = dict(collision_entry)
            collision_entry_ck['priority'] = 'HIGH'
            collision_entry_ck['category'] = 'ckey-ckey'
            collision_entry_ck['ckey_indices'] = ckey_indices
            result['ckey_ckey_collisions'].append(collision_entry_ck)
            result['total_cross_record'] += 1

        if has_mkey and has_metadata:
            collision_entry_mm = dict(collision_entry)
            collision_entry_mm['priority'] = 'HIGH'
            collision_entry_mm['category'] = 'mkey-metadata'
            result['mkey_metadata_collisions'].append(collision_entry_mm)

        if has_ckey and has_metadata:
            collision_entry_cm = dict(collision_entry)
            collision_entry_cm['priority'] = 'MEDIUM'
            collision_entry_cm['category'] = 'ckey-metadata'
            result['ckey_metadata_collisions'].append(collision_entry_cm)

        # Internal: same key blob has the block at multiple offsets
        if has_mkey and len(mkey_offsets) >= 2:
            collision_entry_int = dict(collision_entry)
            collision_entry_int['priority'] = 'INFO'
            collision_entry_int['category'] = 'mkey-internal'
            result['key_internal_collisions'].append(collision_entry_int)

        for ci, ci_offsets in ckey_offsets.items():
            if len(ci_offsets) >= 2:
                collision_entry_int2 = dict(collision_entry)
                collision_entry_int2['priority'] = 'INFO'
                collision_entry_int2['category'] = f'ckey[{ci}]-internal'
                result['key_internal_collisions'].append(collision_entry_int2)

    return result


# ═══════════════════════════════════════════════════════════════════
# Section 4 — Amplified feasibility tests (A–D) with sliding-window data
# ═══════════════════════════════════════════════════════════════════
def _test_a_sliding(wallet: Dict[str, Any],
                    xref: Dict[str, Any]) -> Dict[str, Any]:
    """Test A (amplified): Known-plaintext brute-force on reused mkey blocks.

    Uses sliding-window mkey-ckey and mkey-metadata collisions.
    """
    result: Dict[str, Any] = {
        'test': 'A',
        'name': 'Known-plaintext brute-force on reused mkey blocks (sliding)',
        'score': 0,
        'feasible': False,
        'details': [],
    }

    mkey_ckey = xref.get('mkey_ckey_collisions', [])
    mkey_meta = xref.get('mkey_metadata_collisions', [])

    if mkey_ckey:
        result['score'] += 50
        result['feasible'] = True
        result['details'].append(
            f'{len(mkey_ckey)} mkey-ckey collision(s) found via sliding window. '
            f'CBC chaining recovery is directly applicable.')
        for col in mkey_ckey[:3]:
            result['details'].append(
                f'  Block {col["block_hex"][:32]}... '
                f'mkey@{col["mkey_offsets"]} ckey@{col["ckey_offsets"]}')

    if mkey_meta:
        result['score'] += 25
        result['details'].append(
            f'{len(mkey_meta)} mkey-metadata collision(s). '
            f'Known BDB metadata plaintext can reveal mkey block content.')

    if not mkey_ckey and not mkey_meta:
        result['details'].append(
            'No mkey collisions found in sliding-window scan.')

    return result


def _test_b_sliding(wallet: Dict[str, Any],
                    xref: Dict[str, Any]) -> Dict[str, Any]:
    """Test B (amplified): Structure-aided plaintext inference."""
    result: Dict[str, Any] = {
        'test': 'B',
        'name': 'Structure-aided plaintext inference (sliding)',
        'score': 0,
        'feasible': False,
        'details': [],
    }

    all_key_collisions = (
        xref.get('mkey_ckey_collisions', []) +
        xref.get('ckey_ckey_collisions', []) +
        xref.get('mkey_metadata_collisions', []) +
        xref.get('ckey_metadata_collisions', [])
    )

    if not all_key_collisions:
        result['details'].append('No key-material collisions for inference.')
        return result

    # Count collisions with metadata (known plaintext source)
    meta_collisions = (
        xref.get('mkey_metadata_collisions', []) +
        xref.get('ckey_metadata_collisions', [])
    )

    if meta_collisions:
        result['score'] += 30
        result['details'].append(
            f'{len(meta_collisions)} collision(s) between key material and '
            f'BDB metadata (potential known-plaintext source).')

        # Check entropy of colliding blocks
        low_entropy = [c for c in meta_collisions if c.get('entropy', 8) < 3.0]
        if low_entropy:
            result['score'] += 20
            result['feasible'] = True
            result['details'].append(
                f'{len(low_entropy)} low-entropy block(s) in key-metadata '
                f'collisions — high confidence known-plaintext recovery.')

    cross_record = (
        xref.get('mkey_ckey_collisions', []) +
        xref.get('ckey_ckey_collisions', [])
    )
    if cross_record:
        result['score'] += 20
        result['details'].append(
            f'{len(cross_record)} cross-record collision(s) enable '
            f'CBC chaining analysis between different encrypted blobs.')

    if result['score'] >= 40:
        result['feasible'] = True

    return result


def _test_c_sliding(wallet: Dict[str, Any],
                    sliding_collisions: Dict[bytes, List[int]],
                    xref: Dict[str, Any]) -> Dict[str, Any]:
    """Test C (amplified): Statistical entropy check on sliding-window blocks."""
    result: Dict[str, Any] = {
        'test': 'C',
        'name': 'Statistical entropy check (sliding)',
        'score': 0,
        'feasible': False,
        'details': [],
        'block_classifications': [],
    }

    key_ranges = _key_material_ranges(wallet)

    low_entropy_key = 0
    high_entropy_key = 0
    total_repeated = 0

    for blk, offsets in sliding_collisions.items():
        if len(offsets) < 2:
            continue
        total_repeated += 1

        entropy = _shannon_entropy(blk)
        touches_key = any(
            _classify_offset(off, key_ranges) for off in offsets
        )

        if touches_key:
            if entropy < 3.0:
                low_entropy_key += 1
            else:
                high_entropy_key += 1

            if len(result['block_classifications']) < 50:
                result['block_classifications'].append({
                    'block_hex': blk.hex(),
                    'count': len(offsets),
                    'entropy_bits': round(entropy, 4),
                    'touches_key_material': True,
                })

    result['details'].append(
        f'{total_repeated} total repeated blocks in sliding-window scan.')

    if low_entropy_key > 0:
        result['score'] += min(70, 30 + low_entropy_key * 15)
        result['feasible'] = True
        result['details'].append(
            f'{low_entropy_key} low-entropy block(s) touching key material — '
            f'exploitable via known-plaintext.')

    if high_entropy_key > 0:
        result['score'] = max(result['score'], 20)
        result['details'].append(
            f'{high_entropy_key} high-entropy block(s) touching key material — '
            f'potential ciphertext reuse.')

    return result


def _test_d_sliding(wallet: Dict[str, Any],
                    xref: Dict[str, Any]) -> Dict[str, Any]:
    """Test D (amplified): ckey-ckey brute-force with sliding-window collisions."""
    result: Dict[str, Any] = {
        'test': 'D',
        'name': 'CKey collision brute-force (sliding)',
        'score': 0,
        'feasible': False,
        'details': [],
        'ckey_collisions': [],
    }

    ckey_ckey = xref.get('ckey_ckey_collisions', [])
    ckeys = wallet.get('ckeys', [])

    if not ckey_ckey:
        result['details'].append('No ckey-ckey collisions in sliding-window scan.')
        return result

    result['score'] += 25
    result['details'].append(
        f'{len(ckey_ckey)} ckey-ckey collision(s) found via sliding window.')

    for col in ckey_ckey[:5]:
        indices = col.get('ckey_indices', [])
        if len(indices) >= 2:
            ci_a, ci_b = indices[0], indices[1]
            if ci_a < len(ckeys) and ci_b < len(ckeys):
                iv_a = _ckey_iv(ckeys[ci_a].pubkey)
                iv_b = _ckey_iv(ckeys[ci_b].pubkey)
                iv_xor = bytes(a ^ b for a, b in zip(iv_a, iv_b))

                col_info = {
                    'block_hex': col['block_hex'][:32] + '...',
                    'ckey_a': ci_a,
                    'ckey_b': ci_b,
                    'iv_xor_hex': iv_xor.hex(),
                    'relationship': (
                        f'P_ckey[{ci_b}] = P_ckey[{ci_a}] ⊕ IV_a ⊕ IV_b'
                    ),
                }
                result['ckey_collisions'].append(col_info)
                result['score'] += 10

    if result['score'] >= 35:
        result['feasible'] = True
        result['details'].append(
            'FEASIBLE: ckey-ckey sliding-window collisions with known IVs '
            'enable algebraic private key recovery.')

    return result


# ═══════════════════════════════════════════════════════════════════
# Section 5 — Diff: new collisions vs aligned scan
# ═══════════════════════════════════════════════════════════════════
def compute_collision_diff(
    aligned_counter: collections.Counter,
    sliding_collisions: Dict[bytes, List[int]],
) -> Dict[str, Any]:
    """Compute which collisions are NEW (found only by sliding window).

    A collision is "new" if the block does NOT appear as a repeated block
    in the aligned scan (i.e., aligned_counter[block] < 2), OR if the
    block appears at non-aligned offsets that the aligned scan cannot see.
    """
    aligned_repeated = {
        blk for blk, cnt in aligned_counter.items() if cnt >= 2
    }

    new_blocks = []
    shared_blocks = []
    new_offset_only = []  # block is aligned-repeated but has extra non-aligned offsets

    for blk, offsets in sliding_collisions.items():
        if len(offsets) < 2:
            continue

        if blk in aligned_repeated:
            # Check if any offsets are non-aligned
            non_aligned = [o for o in offsets if o % BLOCK_SIZE != 0]
            if non_aligned:
                new_offset_only.append({
                    'block_hex': blk.hex(),
                    'total_offsets': len(offsets),
                    'non_aligned_offsets': non_aligned,
                    'aligned_count': aligned_counter[blk],
                })
            shared_blocks.append(blk.hex())
        else:
            new_blocks.append({
                'block_hex': blk.hex(),
                'offsets': offsets,
                'count': len(offsets),
            })

    return {
        'new_collision_count': len(new_blocks),
        'shared_collision_count': len(shared_blocks),
        'new_offset_only_count': len(new_offset_only),
        'new_collisions': new_blocks[:50],  # cap for report size
        'new_offset_only': new_offset_only[:50],
        'total_sliding_repeated': sum(
            1 for offsets in sliding_collisions.values() if len(offsets) >= 2
        ),
        'total_aligned_repeated': len(aligned_repeated),
    }


# ═══════════════════════════════════════════════════════════════════
# Section 6 — Main analysis pipeline
# ═══════════════════════════════════════════════════════════════════
def run_sliding_window_analysis(
    wallet_path: str,
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full aggressive overlapping-window collision analysis."""

    t_start = time.time()

    # ── Parse wallet ──
    wallet = parse_wallet_bdb(wallet_path)
    data = wallet['raw']
    mkey = wallet.get('mkey')
    ckeys = wallet.get('ckeys', [])

    print(f'  [*] Wallet: {wallet_path}')
    print(f'  [*] Size:   {len(data):,} bytes')
    print(f'  [*] mkey:   {"found" if mkey else "NOT found"}')
    print(f'  [*] ckeys:  {len(ckeys)}')

    # ── Aligned-scan baseline ──
    print(f'\n  {"═" * 60}')
    print(f'  ALIGNED-SCAN BASELINE (16-byte boundaries)')
    print(f'  {"═" * 60}')

    aligned_counter = _build_aligned_block_counter(data)
    aligned_total = sum(aligned_counter.values())
    aligned_unique = len(aligned_counter)
    aligned_repeated = sum(1 for c in aligned_counter.values() if c > 1)
    aligned_rep_rate = (aligned_total - aligned_unique) / max(aligned_total, 1)

    print(f'  Total aligned blocks:   {aligned_total}')
    print(f'  Unique aligned blocks:  {aligned_unique}')
    print(f'  Repeated aligned blocks:{aligned_repeated}')
    print(f'  Aligned repetition rate:{aligned_rep_rate:.4%}')

    # ── Sliding-window scan ──
    print(f'\n  {"═" * 60}')
    print(f'  SLIDING-WINDOW SCAN (1-byte stride)')
    print(f'  {"═" * 60}')

    sliding_map, sliding_non_zero = _scan_sliding_window_full(data)
    sliding_unique = len(sliding_map)
    sliding_repeated_count = sum(
        1 for offsets in sliding_map.values() if len(offsets) >= 2
    )
    sliding_total_excess = sum(
        len(offsets) - 1 for offsets in sliding_map.values() if len(offsets) >= 2
    )
    sliding_rep_rate = sliding_total_excess / max(sliding_non_zero, 1)

    total_possible = len(data) - BLOCK_SIZE + 1
    print(f'  Total sliding slices:   {total_possible:,}')
    print(f'  Non-zero slices:        {sliding_non_zero:,}')
    print(f'  Unique slices:          {sliding_unique:,}')
    print(f'  Repeated slices:        {sliding_repeated_count:,}')
    print(f'  Sliding repetition rate:{sliding_rep_rate:.4%}')
    print(f'  Aligned baseline rate:  {aligned_rep_rate:.4%}')

    if sliding_rep_rate > aligned_rep_rate:
        delta = sliding_rep_rate - aligned_rep_rate
        print(f'  DELTA (sliding - aligned): +{delta:.4%}')
    else:
        print(f'  DELTA (sliding - aligned): {sliding_rep_rate - aligned_rep_rate:.4%}')

    # ── Cross-reference with key material ──
    print(f'\n  {"═" * 60}')
    print(f'  CROSS-REFERENCE WITH KEY MATERIAL')
    print(f'  {"═" * 60}')

    xref = cross_reference_collisions(sliding_map, wallet)

    print(f'  Total key-touching collisions:  {xref["total_key_touching"]}')
    print(f'  Cross-record collisions:        {xref["total_cross_record"]}')
    print(f'  mkey-ckey collisions:           {len(xref["mkey_ckey_collisions"])}')
    print(f'  ckey-ckey collisions:           {len(xref["ckey_ckey_collisions"])}')
    print(f'  mkey-metadata collisions:       {len(xref["mkey_metadata_collisions"])}')
    print(f'  ckey-metadata collisions:       {len(xref["ckey_metadata_collisions"])}')
    print(f'  Key-internal collisions:        {len(xref["key_internal_collisions"])}')

    if xref['mkey_ckey_collisions']:
        print(f'\n  [!!] CRITICAL: mkey-ckey collisions detected!')
        for col in xref['mkey_ckey_collisions'][:3]:
            print(f'       Block: {col["block_hex"][:32]}...')
            print(f'       mkey offsets: {col["mkey_offsets"][:5]}')
            print(f'       ckey offsets: {col["ckey_offsets"]}')

    if xref['ckey_ckey_collisions']:
        print(f'\n  [!!] HIGH: ckey-ckey collisions detected!')
        for col in xref['ckey_ckey_collisions'][:3]:
            print(f'       Block: {col["block_hex"][:32]}...')
            print(f'       ckey indices: {col.get("ckey_indices", [])}')

    # ── Collision diff ──
    print(f'\n  {"═" * 60}')
    print(f'  COLLISION DIFF (sliding vs aligned)')
    print(f'  {"═" * 60}')

    diff = compute_collision_diff(aligned_counter, sliding_map)

    print(f'  New collisions (sliding only):  {diff["new_collision_count"]}')
    print(f'  Shared collisions:              {diff["shared_collision_count"]}')
    print(f'  New non-aligned offsets:         {diff["new_offset_only_count"]}')

    if diff['new_collision_count'] > 0:
        print(f'\n  [+] {diff["new_collision_count"]} NEW collision(s) found '
              f'that the aligned scan missed!')
        for nc in diff['new_collisions'][:5]:
            print(f'      Block: {nc["block_hex"][:32]}... '
                  f'count={nc["count"]} offsets={nc["offsets"][:5]}')

    # ── Amplified feasibility tests ──
    print(f'\n  {"═" * 60}')
    print(f'  AMPLIFIED FEASIBILITY TESTS (A–D)')
    print(f'  {"═" * 60}')

    test_a = _test_a_sliding(wallet, xref)
    test_b = _test_b_sliding(wallet, xref)
    test_c = _test_c_sliding(wallet, sliding_map, xref)
    test_d = _test_d_sliding(wallet, xref)

    test_results = [test_a, test_b, test_c, test_d]

    for t in test_results:
        status = "YES ✓" if t['feasible'] else "NO ✗"
        print(f'\n  Test {t["test"]}: {t["name"]}')
        print(f'    Score: {t["score"]}/100  Feasible: {status}')
        for detail in t['details']:
            print(f'    {detail}')

    # ── Overall assessment ──
    any_feasible = any(t['feasible'] for t in test_results)
    max_score = max(t['score'] for t in test_results) if test_results else 0
    avg_score = sum(t['score'] for t in test_results) / max(len(test_results), 1)
    confidence = min(100, int(max_score * 0.7 + avg_score * 0.3))

    elapsed = time.time() - t_start

    print(f'\n  {"═" * 60}')
    print(f'  SLIDING-WINDOW ANALYSIS COMPLETE')
    print(f'  {"═" * 60}')
    print(f'  Elapsed:                {elapsed:.2f}s')
    print(f'  Aligned repetition rate:{aligned_rep_rate:.4%}')
    print(f'  Sliding repetition rate:{sliding_rep_rate:.4%}')
    print(f'  New collisions found:   {diff["new_collision_count"]}')
    print(f'  mkey-ckey collisions:   {len(xref["mkey_ckey_collisions"])}')
    print(f'  ckey-ckey collisions:   {len(xref["ckey_ckey_collisions"])}')
    print(f'  Overall feasible:       {"YES" if any_feasible else "NO"}')
    print(f'  Confidence:             {confidence}%')
    print(f'  {"═" * 60}')

    # ── Build full result ──
    full_result = {
        'finding': FINDING_ID,
        'wallet_file': wallet_path,
        'file_size': len(data),
        'mkey_found': mkey is not None,
        'ckey_count': len(ckeys),
        'aligned_scan': {
            'total_blocks': aligned_total,
            'unique_blocks': aligned_unique,
            'repeated_blocks': aligned_repeated,
            'repetition_rate': round(aligned_rep_rate, 6),
        },
        'sliding_window_scan': {
            'total_slices': total_possible,
            'non_zero_slices': sliding_non_zero,
            'unique_slices': sliding_unique,
            'repeated_slices': sliding_repeated_count,
            'repetition_rate': round(sliding_rep_rate, 6),
        },
        'cross_reference': {
            'total_key_touching': xref['total_key_touching'],
            'total_cross_record': xref['total_cross_record'],
            'mkey_ckey_count': len(xref['mkey_ckey_collisions']),
            'ckey_ckey_count': len(xref['ckey_ckey_collisions']),
            'mkey_metadata_count': len(xref['mkey_metadata_collisions']),
            'ckey_metadata_count': len(xref['ckey_metadata_collisions']),
            'mkey_ckey_collisions': xref['mkey_ckey_collisions'][:20],
            'ckey_ckey_collisions': xref['ckey_ckey_collisions'][:20],
            'mkey_metadata_collisions': xref['mkey_metadata_collisions'][:20],
            'ckey_metadata_collisions': xref['ckey_metadata_collisions'][:20],
        },
        'collision_diff': {
            'new_collision_count': diff['new_collision_count'],
            'shared_collision_count': diff['shared_collision_count'],
            'new_offset_only_count': diff['new_offset_only_count'],
            'new_collisions': diff['new_collisions'][:20],
        },
        'feasibility_tests': {
            'test_a': test_a,
            'test_b': test_b,
            'test_c': test_c,
            'test_d': test_d,
        },
        'overall': {
            'feasible': any_feasible,
            'confidence_percent': confidence,
            'max_test_score': max_score,
            'average_test_score': round(avg_score, 1),
            'feasible_tests': [t['name'] for t in test_results if t['feasible']],
        },
        'elapsed_seconds': round(elapsed, 2),
    }

    # ── Write JSON report ──
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(full_result, f, indent=2, default=str)
        print(f'\n  [*] Report written to: {output_json}')

    return full_result


# ═══════════════════════════════════════════════════════════════════
# Section 7 — CLI entry point
# ═══════════════════════════════════════════════════════════════════
def main():
    """CLI entry point."""
    print(f'''
╔══════════════════════════════════════════════════════════════╗
║  {FINDING_ID}                    ║
║  Aggressive Overlapping-Window Collision Scanner             ║
║                                                              ║
║  Slides a 16-byte window one byte at a time to find          ║
║  collisions that aligned scans miss.                         ║
╚══════════════════════════════════════════════════════════════╝
''')

    parser = argparse.ArgumentParser(
        description=(
            f'{FINDING_ID}: Aggressive overlapping-window collision scanner. '
            f'Finds IV-reuse collisions at non-aligned offsets that the '
            f'standard 16-byte-aligned scan misses.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  %(prog)s --wallet wallet.dat\n'
            '  %(prog)s --wallet wallet.dat --output-json sliding_report.json\n'
            '  %(prog)s --wallet wallet.dat -v\n'
        ),
    )

    parser.add_argument(
        '--wallet', type=str,
        default=os.path.expanduser('~/.bitcoin/wallet.dat'),
        help='Path to wallet.dat file (default: ~/.bitcoin/wallet.dat)',
    )
    parser.add_argument(
        '--output-json', nargs='?', const='sliding_window_report.json',
        default=None,
        help='Write JSON report (default name: sliding_window_report.json)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable detailed output',
    )

    args = parser.parse_args()

    if not os.path.isfile(args.wallet):
        print(f'  [!] Wallet file not found: {args.wallet}')
        sys.exit(1)

    run_sliding_window_analysis(
        wallet_path=args.wallet,
        output_json=args.output_json,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
