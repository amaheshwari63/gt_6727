#!/usr/bin/env python3
"""Generate synthetic 65+ spending datasets for research."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_bls_xlsx(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = ["".join((t.text or "") for t in si.findall(".//a:t", NS)).strip() for si in shared_root.findall("a:si", NS)]
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: Dict[int, Dict[str, str]] = {}
    for row in sheet.findall(".//a:sheetData/a:row", NS):
        rnum = int(row.attrib["r"])
        vals: Dict[str, str] = {}
        for c in row.findall("a:c", NS):
            m = re.match(r"([A-Z]+)(\d+)", c.attrib.get("r", ""))
            if not m:
                continue
            col = m.group(1)
            t = c.attrib.get("t")
            v = c.find("a:v", NS)
            if v is None:
                continue
            val = v.text or ""
            if t == "s":
                val = shared[int(val)]
            vals[col] = val.strip()
        rows[rnum] = vals

    annual_65_74 = float(rows[51]["I"])
    annual_75_plus = float(rows[51]["J"])

    shares_65_74 = {}
    shares_75_plus = {}
    for r in sorted(rows):
        label = rows[r].get("A", "").strip()
        if not label or label in {"Mean", "SE", "RSE", "Share", "Average annual expenditures"}:
            continue
        share_row = rows.get(r + 2, {})
        if share_row.get("A", "").strip() != "Share":
            continue
        try:
            s1 = float(share_row.get("I", ""))
            s2 = float(share_row.get("J", ""))
        except ValueError:
            continue
        if s1 <= 0 and s2 <= 0:
            continue
        cleaned = re.sub(r"\s+", " ", label).strip().replace("/", "-")
        shares_65_74[cleaned] = s1
        shares_75_plus[cleaned] = s2

    def norm(d: Dict[str, float]) -> Dict[str, float]:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()}

    return {
        "annual": {"65-74": annual_65_74, "75+": annual_75_plus},
        "shares": {"65-74": norm(shares_65_74), "75+": norm(shares_75_plus)},
    }


def monthly_series(start: dt.date, months: int) -> List[dt.date]:
    out = []
    y, m = start.year, start.month
    for _ in range(months):
        out.append(dt.date(y, m, 1))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def random_ts_in_month(rng: random.Random, month_start: dt.date) -> dt.datetime:
    next_month = dt.date(month_start.year + (month_start.month == 12), 1 if month_start.month == 12 else month_start.month + 1, 1)
    days = (next_month - month_start).days
    return dt.datetime(month_start.year, month_start.month, rng.randint(1, days), rng.randint(8, 20), rng.randint(0, 59), rng.randint(0, 59))


def weighted_pick(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    x = rng.random()
    c = 0.0
    for k, p in items:
        c += p
        if x <= c:
            return k
    return items[-1][0]


def amounts_with_total(rng: random.Random, n: int, total: float) -> List[float]:
    x = [rng.gammavariate(1.6, 1.0) for _ in range(n)]
    s = sum(x)
    vals = [max(1.0, total * v / s) for v in x]
    vals = [round(v, 2) for v in vals]
    diff = round(total - sum(vals), 2)
    vals[-1] = round(vals[-1] + diff, 2)
    return vals


def build_users(seed: int, bls: dict, n_users: int = 100) -> List[dict]:
    rng = random.Random(seed)
    users = []
    for i in range(1, n_users + 1):
        if rng.random() < 0.6:
            grp = "65-74"
            age = rng.randint(65, 74)
        else:
            grp = "75+"
            age = rng.randint(75, 90)
        annual = max(24000.0, rng.gauss(bls["annual"][grp], bls["annual"][grp] * 0.12))
        users.append({
            "person_id": f"P{i:03d}",
            "person_age": age,
            "age_group": grp,
            "annual_expenditure": annual,
            "shares": bls["shares"][grp],
        })
    return users


def generate_dataset(dataset_name: str, users: List[dict], seed: int, scam_users: set, scam_start_month_idx: int, months: int) -> List[dict]:
    records = []
    start = dt.date(2023, 1, 1)
    month_list = monthly_series(start, months)

    categories = sorted({c for u in users for c in u["shares"].keys()})
    normal_sources = {c: [f"{re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_')}_payee_{i:02d}" for i in range(1, 13)] for c in categories}
    scam_sources = {
        "investment taxes": ["inv_tax_service", "inv_tax_processing"],
        "investment fees": ["inv_fee_desk", "inv_fee_settlement"],
        "investment penalties": ["inv_penalty_release", "inv_penalty_clearance"],
    }

    for u in users:
        rng = random.Random(seed * 1000 + int(u["person_id"][1:]))
        share_items = list(u["shares"].items())
        seen_payees = set()

        for mi, mstart in enumerate(month_list):
            scam_month = u["person_id"] in scam_users and mi >= scam_start_month_idx
            month_total = u["annual_expenditure"] / 12.0
            if scam_month:
                month_total *= (1.0 + 0.05 * (mi - scam_start_month_idx + 1))
            tx_count = 50
            amts = amounts_with_total(rng, tx_count, month_total)

            scam_idx = set()
            if scam_month:
                n_scam = min(18, 6 + (mi - scam_start_month_idx) * 2)
                scam_idx = set(rng.sample(range(tx_count), n_scam))

            for ti in range(tx_count):
                ts = random_ts_in_month(rng, mstart)
                amount = amts[ti]
                is_fraud = 0
                red_new = 0
                red_escalated = 0
                red_tax_fee_penalty = 0

                if ti in scam_idx:
                    is_fraud = 1
                    phase = (mi - scam_start_month_idx + 1)
                    cat = weighted_pick(rng, [("investment taxes", 0.35), ("investment fees", 0.35), ("investment penalties", 0.30)])
                    payee = weighted_pick(rng, [(p, 1.0 / len(scam_sources[cat])) for p in scam_sources[cat]])
                    red_new = 1 if payee not in seen_payees else 0
                    red_tax_fee_penalty = 1
                    amount = round(max(20.0, amount * (2.5 + 0.35 * phase)), 2)
                    red_escalated = 1 if phase >= 2 else 0
                else:
                    cat = weighted_pick(rng, share_items)
                    payee = rng.choice(normal_sources[cat])

                seen_payees.add(payee)
                records.append({
                    "dataset_name": dataset_name,
                    "person_id": u["person_id"],
                    "person_age": u["person_age"],
                    "age_group": u["age_group"],
                    "timestamp": ts.isoformat(),
                    "amount_usd": round(amount, 2),
                    "spend_category": cat,
                    "payee_id": payee,
                    "payee_category": re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_"),
                    "is_fraud_signal": is_fraud,
                    "red_flag_new_payee": red_new,
                    "red_flag_escalated_payments": red_escalated,
                    "red_flag_tax_fee_penalty": red_tax_fee_penalty,
                    "currency": "USD",
                    "status": "posted",
                })

    records.sort(key=lambda r: (r["timestamp"], r["person_id"], r["payee_id"], r["amount_usd"]))
    for i, r in enumerate(records, start=1):
        r["transaction_id"] = f"TX{i:08d}"
    return records


def write_csv(path: Path, rows: List[dict]) -> None:
    cols = [
        "transaction_id",
        "dataset_name",
        "person_id",
        "person_age",
        "age_group",
        "timestamp",
        "amount_usd",
        "spend_category",
        "payee_id",
        "payee_category",
        "is_fraud_signal",
        "red_flag_new_payee",
        "red_flag_escalated_payments",
        "red_flag_tax_fee_penalty",
        "currency",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def summarize(rows: List[dict]) -> dict:
    users = {r["person_id"] for r in rows}
    fraud_users = {r["person_id"] for r in rows if r["is_fraud_signal"] == 1}
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
    ap.add_argument("--input", type=Path, default=Path("data/reference-person-age-ranges-2024.xlsx"))
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    bls = parse_bls_xlsx(args.input)
    users = build_users(args.seed, bls, 100)
    rng = random.Random(args.seed)
    scam_users = set(rng.sample([u["person_id"] for u in users], 20))

    ds1 = generate_dataset("cs6727_DS1", users, args.seed + 11, scam_users, scam_start_month_idx=12, months=18)
    ds2 = generate_dataset("cs6727_DS2", users, args.seed + 29, scam_users, scam_start_month_idx=0, months=18)

    p1 = args.outdir / "cs6727_DS1.csv"
    p2 = args.outdir / "cs6727_DS2.csv"
    write_csv(p1, ds1)
    write_csv(p2, ds2)

    manifest = {
        "seed": args.seed,
        "input": str(args.input).replace("\\", "/"),
        "annual_means": bls["annual"],
        "scam_users": sorted(scam_users),
        "outputs": {
            str(p1).replace("\\", "/"): {**summarize(ds1), "sha256": sha256(p1)},
            str(p2).replace("\\", "/"): {**summarize(ds2), "sha256": sha256(p2)},
        },
    }
    mp = args.outdir / "data_generation_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
