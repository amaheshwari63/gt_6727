#!/usr/bin/env python3
"""Evaluate DS2 preprocessed dataset with trained NumPy RNN+attention model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CONT_COLS = [
    "amount_usd",
    "amount_delta",
    "amount_ratio",
    "increase_count",
    "aggregate3",
    "txn_index",
    "month_index",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


@dataclass
class EvalPrepared:
    x: np.ndarray
    m: np.ndarray
    lengths: np.ndarray
    meta: pd.DataFrame


class NumpyRNNWithAttention:
    def __init__(self, params: Dict[str, np.ndarray]):
        self.params = params

    def forward(self, x: np.ndarray, mask: np.ndarray, lengths: np.ndarray):
        p = self.params
        b, tmax, _ = x.shape
        hdim = p["bh"].shape[0]

        h = np.zeros((b, tmax, hdim), dtype=np.float32)
        h_prev = np.zeros((b, hdim), dtype=np.float32)

        for t in range(tmax):
            z = x[:, t, :] @ p["Wx"] + h_prev @ p["Wh"] + p["bh"]
            h_t = np.tanh(z)
            m = mask[:, t : t + 1]
            h_t = m * h_t + (1.0 - m) * h_prev
            h[:, t, :] = h_t
            h_prev = h_t

        idx = np.clip(lengths - 1, 0, tmax - 1)
        s = h[np.arange(b), idx, :]

        u_pre = h @ p["Wa"] + (s @ p["Ua"])[:, None, :]
        u = np.tanh(u_pre)
        e = u @ p["va"]
        e = e - (1.0 - mask) * 1e9
        e = e - np.max(e, axis=1, keepdims=True)
        ex = np.exp(e) * mask
        a = ex / np.clip(np.sum(ex, axis=1, keepdims=True), 1e-8, None)
        c = np.sum(a[:, :, None] * h, axis=1)

        logits = c @ p["Wt"] + p["bt"]
        return logits, a


def load_ds2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"person_id": "user_id", "transaction_id": "txn_id", "is_fraud_signal": "is_scam"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    base_month = pd.Timestamp(df["timestamp"].min().year, df["timestamp"].min().month, 1)
    df["month_index"] = (df["timestamp"].dt.year - base_month.year) * 12 + (df["timestamp"].dt.month - base_month.month)
    return df


def prepare_windows(df_raw: pd.DataFrame, feature_cols: List[str], norm_stats: Dict[str, Dict[str, float]], seq_len: int) -> EvalPrepared:
    df = df_raw.copy()
    for c in CONT_COLS:
        st = norm_stats[c]
        df[c] = (df[c] - st["mean"]) / st["std"]

    df = df.sort_values(["user_id", "payee_id", "timestamp", "txn_id"]).reset_index(drop=True)

    x, m, l, meta = [], [], [], []

    for (uid, payee), g in df.groupby(["user_id", "payee_id"], sort=False):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        for i in range(len(g)):
            s = max(0, i - seq_len + 1)
            w = arr[s : i + 1]
            ln = w.shape[0]
            pad = np.zeros((seq_len - ln, arr.shape[1]), dtype=np.float32)
            xx = np.vstack([pad, w])
            mm = np.concatenate([np.zeros((seq_len - ln,), dtype=np.float32), np.ones((ln,), dtype=np.float32)])

            r = g.iloc[i]
            x.append(xx); m.append(mm); l.append(ln)
            meta.append(
                {
                    "user_id": uid,
                    "payee_id": payee,
                    "txn_id": r["txn_id"],
                    "timestamp": r["timestamp"],
                    "is_scam": int(r["is_scam"]),
                    "amount_usd": float(r["amount_usd"]),
                    "spend_category": str(r.get("spend_category", "")),
                    "payee_category": str(r.get("payee_category", "")),
                }
            )

    return EvalPrepared(
        x=np.array(x, dtype=np.float32),
        m=np.array(m, dtype=np.float32),
        lengths=np.array(l, dtype=np.int32),
        meta=pd.DataFrame(meta),
    )


def predict_all(model: NumpyRNNWithAttention, prepared: EvalPrepared, batch: int = 1024) -> np.ndarray:
    n = prepared.x.shape[0]
    out = np.zeros((n,), dtype=np.float32)
    for i in range(0, n, batch):
        j = min(n, i + batch)
        logits, _ = model.forward(prepared.x[i:j], prepared.m[i:j], prepared.lengths[i:j])
        out[i:j] = sigmoid(logits).reshape(-1)
    return out


def user_case_metrics(meta: pd.DataFrame, pred: np.ndarray):
    d = meta.copy().reset_index(drop=True)
    d["pred"] = pred.astype(int)
    d = d.sort_values(["user_id", "timestamp", "txn_id"]).reset_index(drop=True)

    rows = []
    lats = []
    for uid, g in d.groupby("user_id", sort=True):
        has_scam = bool((g["is_scam"] == 1).any())
        if has_scam:
            first_scam = g[g["is_scam"] == 1].iloc[0]["timestamp"]
            alarms = g[(g["pred"] == 1) & (g["timestamp"] >= first_scam)]
            detected = not alarms.empty
            if detected:
                lats.append((alarms.iloc[0]["timestamp"] - first_scam).total_seconds() / 3600.0)
        else:
            detected = bool((g["pred"] == 1).any())
        rows.append({"user_id": uid, "actual": int(has_scam), "pred": int(detected)})

    ur = pd.DataFrame(rows)
    cm = confusion_binary(ur["actual"].to_numpy(), ur["pred"].to_numpy())
    lat = float(np.mean(lats)) if lats else math.inf
    return cm, lat, ur


def explain_tp_fp(meta: pd.DataFrame, user_rows: pd.DataFrame, pred: np.ndarray) -> List[str]:
    d = meta.copy().reset_index(drop=True)
    d["pred"] = pred.astype(int)
    d = d.sort_values(["user_id", "timestamp", "txn_id"]).reset_index(drop=True)

    case = {r.user_id: (int(r.actual), int(r.pred)) for r in user_rows.itertuples(index=False)}

    lines = []
    for uid, g in d.groupby("user_id", sort=True):
        actual, pu = case[uid]
        if pu != 1:
            continue
        tag = "TP" if actual == 1 else "FP"

        f = g[
            g[["spend_category", "payee_category"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.lower()
            .str.contains("fee|penalt|tax", regex=True)
        ]

        best = None
        prev = None
        for r in f.itertuples(index=False):
            amt = float(r.amount_usd)
            if prev is not None and amt > prev["amt"]:
                inc = amt - prev["amt"]
                if best is None or inc > best["inc"]:
                    best = {
                        "inc": inc,
                        "a0": prev["amt"],
                        "a1": amt,
                        "ts0": prev["ts"],
                        "tx0": prev["tx"],
                        "ts1": r.timestamp,
                        "tx1": r.txn_id,
                    }
            prev = {"amt": amt, "ts": r.timestamp, "tx": r.txn_id}

        if best is None:
            lines.append(f"User {uid} [{tag}] - Predicted as scam due to learned sequential risk signal; explicit Fees/Penalties increase pair not found.")
        else:
            lines.append(
                f"User {uid} [{tag}] - Unusual spending detected: Your transactions in the Fees/Penalties category increased "
                f"from ${best['a0']:.2f} ({best['ts0']}, {best['tx0']}) to ${best['a1']:.2f} ({best['ts1']}, {best['tx1']}). "
                f"This change may indicate a potential fraudulent trend and is worth reviewing."
            )
    return lines


def write_audit(path: Path, meta: pd.DataFrame, pred: np.ndarray) -> None:
    out = meta.copy().reset_index(drop=True)
    out["transaction_type"] = np.where(out["is_scam"] == 1, "scam", "nonscam")
    out["predicted_type"] = np.where(pred == 1, "scam", "nonscam")
    out = out[["user_id", "txn_id", "timestamp", "transaction_type", "predicted_type"]]
    out = out.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    out["timestamp"] = out["timestamp"].astype(str)
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data/cs6727_DS2_PP.csv"))
    ap.add_argument("--model", type=Path, default=Path("../training/nn_rnn.pkl"))
    ap.add_argument("--audit-out", type=Path, default=Path("evaluation_audit_DS2.csv"))
    ap.add_argument("--summary-out", type=Path, default=Path("evaluation_summary_DS2.json"))
    args = ap.parse_args()

    model_blob = pickle.load(args.model.open("rb"))
    params = model_blob["params"]
    threshold = float(model_blob["threshold"])
    feature_cols = model_blob["config"]["input_features"]
    seq_len = int(model_blob["config"]["seq_len"])
    norm_stats = model_blob["norm_stats"]
    source_iteration = int(model_blob["iteration"])

    df = load_ds2(args.data)
    prep = prepare_windows(df, feature_cols, norm_stats, seq_len=seq_len)

    model = NumpyRNNWithAttention(params)
    proba = predict_all(model, prep)
    pred = (proba >= threshold).astype(int)

    tx_cm = confusion_binary(prep.meta["is_scam"].to_numpy(), pred)
    user_cm, lat, user_rows = user_case_metrics(prep.meta, pred)

    print(f"Model iteration: {source_iteration}")
    print(f"Threshold: {threshold}")
    print("Transaction-level confusion matrix:", tx_cm)
    print("User-level confusion matrix:", user_cm)
    print("User latency avg hours:", lat if not math.isinf(lat) else "inf")

    for r in user_rows.itertuples(index=False):
        print(f"User {r.user_id}: actual_scam_user={r.actual} predicted_scam_user={r.pred} correct={int(r.actual==r.pred)}")

    lines = explain_tp_fp(prep.meta, user_rows, pred)
    if lines:
        print("TP/FP user explanations:")
        for ln in lines:
            print(ln)
    else:
        print("TP/FP user explanations: none")

    write_audit(args.audit_out, prep.meta, pred)

    summary = {
        "model_iteration": source_iteration,
        "threshold": threshold,
        "transaction_confusion": tx_cm,
        "user_confusion": user_cm,
        "user_latency_hours_avg": lat,
        "rows": int(len(prep.meta)),
        "users": int(user_rows.shape[0]),
        "data_path": str(args.data).replace("\\", "/"),
        "model_path": str(args.model).replace("\\", "/"),
        "audit_path": str(args.audit_out).replace("\\", "/"),
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
