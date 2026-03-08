#!/usr/bin/env python3
"""Preprocess cs6727 DS1/DS2 and add user-payee temporal features."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def slope3(vals: List[float]) -> float:
    # slope for points (1,y1),(2,y2),(3,y3)
    x = [1.0, 2.0, 3.0]
    xbar = 2.0
    ybar = sum(vals) / 3.0
    num = sum((x[i] - xbar) * (vals[i] - ybar) for i in range(3))
    den = sum((x[i] - xbar) ** 2 for i in range(3))
    return 0.0 if den == 0 else num / den


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["person_id"], r["timestamp"], r["transaction_id"]))
    return rows


def is_penatly(row: dict) -> int:
    txt = " ".join([
        row.get("spend_category", ""),
        row.get("payee_category", ""),
        row.get("payee_id", ""),
    ]).lower()
    return 1 if any(k in txt for k in ["tax", "fee", "penalty", "penalties"]) else 0


def preprocess(rows: List[dict], add_labels: bool) -> List[dict]:
    prev_amount: Dict[Tuple[str, str], float] = {}
    txn_index: Dict[Tuple[str, str], int] = defaultdict(int)
    increase_count: Dict[Tuple[str, str], int] = defaultdict(int)
    hist: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=3))

    seen_payees: Dict[str, set] = defaultdict(set)
    last_seen_ts: Dict[Tuple[str, str], dt.datetime] = {}

    scam_user = defaultdict(int)
    if add_labels:
        for r in rows:
            if int(r.get("is_fraud_signal", "0")) == 1:
                scam_user[r["person_id"]] = 1

    out = []
    for r in rows:
        user = r["person_id"]
        payee = r["payee_id"]
        key = (user, payee)

        amt = float(r["amount_usd"])
        ts = parse_ts(r["timestamp"])

        # First definition in prompt; retained with explicit name.
        new_payee_first_txn = 1 if payee not in seen_payees[user] else 0

        # Second (duplicated) definition in prompt; emitted as separate column.
        if key not in last_seen_ts:
            new_payee_30d = 1
        else:
            gap_days = (ts - last_seen_ts[key]).total_seconds() / 86400.0
            new_payee_30d = 1 if gap_days > 30 else 0

        pen = is_penatly(r)

        if key in prev_amount:
            prev = prev_amount[key]
            delta = amt - prev
            ratio = amt / prev if prev != 0 else 1.0
            if amt > prev:
                increase_count[key] += 1
        else:
            delta = 0.0
            ratio = 1.0

        txn_index[key] += 1
        idx = txn_index[key]

        h = hist[key]
        h.append(amt)
        agg3 = slope3(list(h)) if len(h) == 3 else 0.0

        prev_amount[key] = amt
        seen_payees[user].add(payee)
        last_seen_ts[key] = ts

        row = dict(r)
        # Use the latest duplicated prompt definition for `new_payee`.
        row["new_payee_first_txn"] = int(new_payee_first_txn)
        row["new_payee"] = int(new_payee_30d)
        row["is_penatly"] = int(pen)
        row["amount_delta"] = round(delta, 2)
        row["amount_ratio"] = round(ratio, 6)
        row["increase_count"] = int(increase_count[key])
        row["aggregate3"] = round(agg3, 6)
        row["txn_index"] = int(idx)
        row["new_payee_30d"] = int(new_payee_30d)

        if add_labels:
            y_target = 1 if (
                int(r.get("is_fraud_signal", "0")) == 1
                and row["new_payee"] == 1
                and row["is_penatly"] == 1
                and (row["increase_count"] >= 1 or row["amount_ratio"] > 1)
            ) else 0
            row["y_target_escalating_fraud"] = y_target
            row["y_case_scam_user"] = int(scam_user[user])

        out.append(row)

    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[dict]) -> dict:
    users = sorted({r["person_id"] for r in rows})
    fraud_users = sorted({r["person_id"] for r in rows if int(r.get("is_fraud_signal", "0")) == 1})
    return {
        "rows": len(rows),
        "users": len(users),
        "fraud_users": len(fraud_users),
        "date_min": min(r["timestamp"] for r in rows),
        "date_max": max(r["timestamp"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=67270308)
    ap.add_argument("--ds1", type=Path, default=Path("data/cs6727_DS1.csv"))
    ap.add_argument("--ds2", type=Path, default=Path("data/cs6727_DS2.csv"))
    ap.add_argument("--out1", type=Path, default=Path("data/cs6727_DS1_PP.csv"))
    ap.add_argument("--out2", type=Path, default=Path("data/cs6727_DS2_PP.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("data/data_preprocessing_manifest.json"))
    args = ap.parse_args()

    random.seed(args.seed)

    ds1_pp = preprocess(load_rows(args.ds1), add_labels=True)
    ds2_pp = preprocess(load_rows(args.ds2), add_labels=False)

    write_csv(args.out1, ds1_pp)
    write_csv(args.out2, ds2_pp)

    manifest = {
        "seed": args.seed,
        "inputs": {
            "ds1": str(args.ds1).replace("\\", "/"),
            "ds2": str(args.ds2).replace("\\", "/"),
        },
        "outputs": {
            str(args.out1).replace("\\", "/"): {**summarize(ds1_pp), "sha256": sha256(args.out1)},
            str(args.out2).replace("\\", "/"): {**summarize(ds2_pp), "sha256": sha256(args.out2)},
        },
        "notes": {
            "duplicate_new_payee_resolution": "Used new_payee as 30-day definition (latest prompt line) and preserved first-ever signal as new_payee_first_txn",
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
