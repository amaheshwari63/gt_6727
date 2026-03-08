#!/usr/bin/env python3
"""NumPy/Pandas RNN+Additive-Attention training with threshold tuning.

- Per-user+payee sliding windows
- 32 hidden units, tanh recurrent state
- 5 iterations with timing and confusion matrices
- Best model: least user-level FN, then least user-level FP
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_FEATURE_COLS = [
    "amount_usd",
    "new_payee",
    "is_penatly",
    "amount_delta",
    "amount_ratio",
    "increase_count",
    "aggregate3",
    "txn_index",
    "new_payee_30d",
    "month_index",
]

CONT_COLS = [
    "amount_usd",
    "amount_delta",
    "amount_ratio",
    "increase_count",
    "aggregate3",
    "txn_index",
    "month_index",
]


@dataclass
class Prepared:
    x_train: np.ndarray
    y_train: np.ndarray
    m_train: np.ndarray
    len_train: np.ndarray
    meta_train: pd.DataFrame
    x_eval: np.ndarray
    y_eval: np.ndarray
    m_eval: np.ndarray
    len_eval: np.ndarray
    meta_eval: pd.DataFrame
    feature_cols: List[str]
    norm_stats: Dict[str, Dict[str, float]]


@dataclass
class CaseMetrics:
    tp: int
    fn: int
    fp: int
    tn: int
    avg_latency_hours: float
    user_rows: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def load_base_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "person_id": "user_id",
            "transaction_id": "txn_id",
            "is_fraud_signal": "is_scam",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    base_month = pd.Timestamp(df["timestamp"].min().year, df["timestamp"].min().month, 1)
    df["month_index"] = (df["timestamp"].dt.year - base_month.year) * 12 + (df["timestamp"].dt.month - base_month.month)
    return df


def split_users_random(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    by_user = df.groupby("user_id")["is_scam"].max().reset_index()
    scam = np.array(sorted(by_user[by_user["is_scam"] == 1]["user_id"].tolist()))
    non = np.array(sorted(by_user[by_user["is_scam"] == 0]["user_id"].tolist()))

    if len(scam) < 10 or len(non) < 40:
        raise ValueError("Need >=10 scam users and >=40 non-scam users")

    rng.shuffle(scam)
    rng.shuffle(non)

    train = sorted(list(scam[:8]) + list(non[:32]))
    eval_ = sorted(list(scam[8:10]) + list(non[32:40]))
    return train, eval_


def apply_norm(df: pd.DataFrame, train_users: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out = df.copy()
    tr = out[out["user_id"].isin(train_users)]
    stats = {}
    for c in CONT_COLS:
        mu = float(tr[c].mean())
        sd = float(tr[c].std())
        if sd < 1e-9:
            sd = 1.0
        out[c] = (out[c] - mu) / sd
        stats[c] = {"mean": mu, "std": sd}
    return out, stats


def prepare_sequences(df_raw: pd.DataFrame, train_users: List[str], eval_users: List[str], seq_len: int) -> Prepared:
    need = [
        "new_payee",
        "is_penatly",
        "amount_delta",
        "amount_ratio",
        "increase_count",
        "aggregate3",
        "txn_index",
        "new_payee_30d",
    ]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in preprocessed file: {missing}")

    df_sorted = df_raw.sort_values(["user_id", "payee_id", "timestamp", "txn_id"]).reset_index(drop=True)
    df, stats = apply_norm(df_sorted, train_users)

    x_tr, y_tr, m_tr, l_tr, meta_tr = [], [], [], [], []
    x_ev, y_ev, m_ev, l_ev, meta_ev = [], [], [], [], []

    for (uid, payee), g in df.groupby(["user_id", "payee_id"], sort=False):
        arr = g[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
        labels = g["is_scam"].astype(int).to_numpy(dtype=np.float32)

        for i in range(len(g)):
            s = max(0, i - seq_len + 1)
            w = arr[s : i + 1]
            l = w.shape[0]
            pad = np.zeros((seq_len - l, arr.shape[1]), dtype=np.float32)
            x = np.vstack([pad, w])
            m = np.concatenate([np.zeros((seq_len - l,), dtype=np.float32), np.ones((l,), dtype=np.float32)])
            row = g.iloc[i]
            row_raw = df_sorted.loc[row.name]
            meta = {
                "txn_id": row_raw["txn_id"],
                "user_id": uid,
                "payee_id": payee,
                "timestamp": row_raw["timestamp"],
                "is_scam": int(row_raw["is_scam"]),
                "amount_usd": float(row_raw["amount_usd"]),
                "spend_category": str(row_raw.get("spend_category", "")),
                "payee_category": str(row_raw.get("payee_category", "")),
            }

            if uid in train_users:
                x_tr.append(x); y_tr.append(labels[i]); m_tr.append(m); l_tr.append(l); meta_tr.append(meta)
            elif uid in eval_users:
                x_ev.append(x); y_ev.append(labels[i]); m_ev.append(m); l_ev.append(l); meta_ev.append(meta)

    return Prepared(
        x_train=np.array(x_tr, dtype=np.float32),
        y_train=np.array(y_tr, dtype=np.float32),
        m_train=np.array(m_tr, dtype=np.float32),
        len_train=np.array(l_tr, dtype=np.int32),
        meta_train=pd.DataFrame(meta_tr),
        x_eval=np.array(x_ev, dtype=np.float32),
        y_eval=np.array(y_ev, dtype=np.float32),
        m_eval=np.array(m_ev, dtype=np.float32),
        len_eval=np.array(l_ev, dtype=np.int32),
        meta_eval=pd.DataFrame(meta_ev),
        feature_cols=BASE_FEATURE_COLS,
        norm_stats=stats,
    )


class NumpyRNNWithAttention:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        s = 0.08
        self.params = {
            "Wx": rng.normal(0, s, (input_dim, hidden_dim)).astype(np.float32),
            "Wh": rng.normal(0, s, (hidden_dim, hidden_dim)).astype(np.float32),
            "bh": np.zeros((hidden_dim,), dtype=np.float32),
            "Wa": rng.normal(0, s, (hidden_dim, hidden_dim)).astype(np.float32),
            "Ua": rng.normal(0, s, (hidden_dim, hidden_dim)).astype(np.float32),
            "va": rng.normal(0, s, (hidden_dim,)).astype(np.float32),
            "Wt": rng.normal(0, s, (hidden_dim,)).astype(np.float32),
            "bt": np.zeros((1,), dtype=np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, x: np.ndarray, mask: np.ndarray, lengths: np.ndarray):
        p = self.params
        b, tmax, _ = x.shape
        hdim = p["bh"].shape[0]

        h = np.zeros((b, tmax, hdim), dtype=np.float32)
        h_prev_cache = np.zeros((b, tmax, hdim), dtype=np.float32)

        h_prev = np.zeros((b, hdim), dtype=np.float32)
        for t in range(tmax):
            h_prev_cache[:, t, :] = h_prev
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
        cache = {
            "x": x,
            "mask": mask,
            "lengths": lengths,
            "h": h,
            "h_prev_cache": h_prev_cache,
            "s": s,
            "u": u,
            "a": a,
            "c": c,
            "logits": logits,
        }
        return logits, a, cache

    def backward(self, cache: dict, y: np.ndarray, pos_weight: float):
        p = self.params
        x = cache["x"]
        mask = cache["mask"]
        lengths = cache["lengths"]
        h = cache["h"]
        h_prev_cache = cache["h_prev_cache"]
        u = cache["u"]
        a = cache["a"]
        c = cache["c"]
        logits = cache["logits"]

        b, tmax, hdim = h.shape
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        prob = sigmoid(logits)
        w = np.where(y == 1.0, pos_weight, 1.0).astype(np.float32)
        d_logit = (prob - y) * w / float(b)

        grads["Wt"] += c.T @ d_logit
        grads["bt"] += np.array([np.sum(d_logit)], dtype=np.float32)
        d_c = d_logit[:, None] * p["Wt"][None, :]

        d_a = np.sum(d_c[:, None, :] * h, axis=2)
        d_h = a[:, :, None] * d_c[:, None, :]

        d_e = a * (d_a - np.sum(d_a * a, axis=1, keepdims=True))
        d_e *= mask

        grads["va"] += np.sum(u * d_e[:, :, None], axis=(0, 1))
        d_u = d_e[:, :, None] * p["va"][None, None, :]
        d_u_pre = d_u * (1.0 - u * u)

        grads["Wa"] += np.tensordot(h, d_u_pre, axes=([0, 1], [0, 1]))
        d_h += d_u_pre @ p["Wa"].T

        tmp = np.sum(d_u_pre, axis=1)
        grads["Ua"] += cache["s"].T @ tmp
        d_s = tmp @ p["Ua"].T

        idx = np.clip(lengths - 1, 0, tmax - 1)
        for i in range(b):
            d_h[i, idx[i], :] += d_s[i]

        d_h_next = np.zeros((b, hdim), dtype=np.float32)
        for t in range(tmax - 1, -1, -1):
            m = mask[:, t : t + 1]
            h_t = h[:, t, :]
            h_prev = h_prev_cache[:, t, :]
            d_cur = d_h[:, t, :] + d_h_next
            d_z = d_cur * (1.0 - h_t * h_t) * m
            grads["Wx"] += x[:, t, :].T @ d_z
            grads["Wh"] += h_prev.T @ d_z
            grads["bh"] += np.sum(d_z, axis=0)
            d_h_next = d_z @ p["Wh"].T + d_cur * (1.0 - m)

        loss = -np.mean(
            w * (y * np.log(np.clip(prob, 1e-8, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - prob, 1e-8, 1.0)))
        )
        return grads, float(loss)

    def adam_step(self, grads: dict, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.step += 1
        t = self.step
        for k in self.params:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g * g)
            m_hat = self.m[k] / (1.0 - beta1**t)
            v_hat = self.v[k] / (1.0 - beta2**t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def predict_proba(model: NumpyRNNWithAttention, x: np.ndarray, m: np.ndarray, lengths: np.ndarray, batch: int = 512) -> np.ndarray:
    out = np.zeros((x.shape[0],), dtype=np.float32)
    for i in range(0, x.shape[0], batch):
        j = min(x.shape[0], i + batch)
        logits, _a, _c = model.forward(x[i:j], m[i:j], lengths[i:j])
        out[i:j] = sigmoid(logits).reshape(-1)
    return out


def user_case_metrics(meta: pd.DataFrame, pred: np.ndarray) -> CaseMetrics:
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

        rows.append({
            "user_id": uid,
            "actual": int(has_scam),
            "pred": int(detected),
        })

    ur = pd.DataFrame(rows)
    cm = confusion_binary(ur["actual"].to_numpy(), ur["pred"].to_numpy())
    avg_lat = float(np.mean(lats)) if lats else math.inf
    return CaseMetrics(tp=cm["TP"], fn=cm["FN"], fp=cm["FP"], tn=cm["TN"], avg_latency_hours=avg_lat, user_rows=ur)


def tune_threshold(y_true: np.ndarray, meta_eval: pd.DataFrame, proba: np.ndarray):
    cands = np.concatenate([np.array([0.01, 0.02, 0.05]), np.linspace(0.1, 0.9, 17)])
    best = None
    best_obj = (10**9, 10**9, 10**9, 10**9)

    for th in cands:
        pred = (proba >= th).astype(int)
        tx = confusion_binary(y_true.astype(int), pred.astype(int))
        case = user_case_metrics(meta_eval, pred)
        lat_num = 1e9 if math.isinf(case.avg_latency_hours) else case.avg_latency_hours
        obj = (case.fn, case.fp, lat_num, tx["FP"])
        if obj < best_obj:
            best_obj = obj
            best = {
                "threshold": float(th),
                "pred": pred,
                "tx_cm": tx,
                "case": case,
            }
    return best


def explain_tp_fp(meta_eval: pd.DataFrame, pred: np.ndarray) -> List[str]:
    d = meta_eval.copy().reset_index(drop=True)
    d["pred"] = pred.astype(int)
    d = d.sort_values(["user_id", "timestamp", "txn_id"]).reset_index(drop=True)

    case = user_case_metrics(meta_eval, pred)
    case_map = {r.user_id: (int(r.actual), int(r.pred)) for r in case.user_rows.itertuples(index=False)}

    lines = []
    for uid, g in d.groupby("user_id", sort=True):
        actual, p = case_map[uid]
        if p != 1:
            continue
        tag = "TP" if actual == 1 else "FP"

        s = g[
            g[["spend_category", "payee_category"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.lower()
            .str.contains("fee|penalt|tax", regex=True)
        ]
        best = None
        prev = None
        for r in s.itertuples(index=False):
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


def prepare_all_windows(df_raw: pd.DataFrame, norm_stats: Dict[str, Dict[str, float]], seq_len: int):
    df = df_raw.sort_values(["user_id", "payee_id", "timestamp", "txn_id"]).reset_index(drop=True).copy()
    for c in CONT_COLS:
        st = norm_stats[c]
        df[c] = (df[c] - st["mean"]) / st["std"]

    x, m, l, meta = [], [], [], []
    for (uid, payee), g in df.groupby(["user_id", "payee_id"], sort=False):
        arr = g[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
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
                    "txn_id": r["txn_id"],
                    "timestamp": r["timestamp"],
                    "transaction_type": "scam" if int(r["is_scam"]) == 1 else "nonscam",
                }
            )

    return np.array(x, dtype=np.float32), np.array(m, dtype=np.float32), np.array(l, dtype=np.int32), pd.DataFrame(meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("../data/cs6727_DS1_PP.csv"))
    ap.add_argument("--seed", type=int, default=67270308)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--model-out", type=Path, default=Path("nn_rnn.pkl"))
    ap.add_argument("--audit-out", type=Path, default=Path("evaluation_audit_DS1.csv"))
    args = ap.parse_args()

    set_seed(args.seed)
    df = load_base_df(args.input)

    split_rng = np.random.default_rng(args.seed)

    best_key = (10**9, 10**9, 10**9)
    best_model = None

    print("Model: RNN(hidden=32,tanh)+additive_attention+dense(sigmoid), numpy+pandas")

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()

        train_users, eval_users = split_users_random(df, split_rng)
        prep = prepare_sequences(df, train_users, eval_users, seq_len=args.seq_len)

        model_seed = args.seed + it * 101
        model = NumpyRNNWithAttention(input_dim=prep.x_train.shape[2], hidden_dim=args.hidden, seed=model_seed)

        best_epoch_loss = float("inf")
        best_epoch_params = None
        no_imp = 0
        patience = 2

        idx_all = np.arange(prep.x_train.shape[0])
        for ep in range(1, args.epochs + 1):
            np.random.default_rng(model_seed + ep).shuffle(idx_all)
            losses = []
            for i in range(0, len(idx_all), args.batch_size):
                bidx = idx_all[i : i + args.batch_size]
                x = prep.x_train[bidx]
                y = prep.y_train[bidx]
                m = prep.m_train[bidx]
                ln = prep.len_train[bidx]

                _logits, _a, cache = model.forward(x, m, ln)
                grads, loss = model.backward(cache, y, pos_weight=args.pos_weight)
                model.adam_step(grads, lr=args.lr)
                losses.append(loss)

            p_eval = predict_proba(model, prep.x_eval, prep.m_eval, prep.len_eval, batch=512)
            w = np.where(prep.y_eval == 1.0, args.pos_weight, 1.0)
            val_loss = float(-np.mean(w * (prep.y_eval * np.log(np.clip(p_eval, 1e-8, 1.0)) + (1.0 - prep.y_eval) * np.log(np.clip(1.0 - p_eval, 1e-8, 1.0)))))

            if val_loss < best_epoch_loss - 1e-6:
                best_epoch_loss = val_loss
                best_epoch_params = {k: v.copy() for k, v in model.params.items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        model.params = {k: v.copy() for k, v in best_epoch_params.items()}
        proba_eval = predict_proba(model, prep.x_eval, prep.m_eval, prep.len_eval, batch=512)
        # Calibrate threshold on train windows to reduce optimistic bias on eval.
        proba_train = predict_proba(model, prep.x_train, prep.m_train, prep.len_train, batch=512)
        tuned = tune_threshold(prep.y_train, prep.meta_train, proba_train)

        pred_eval = (proba_eval >= tuned["threshold"]).astype(int)
        tx_cm = confusion_binary(prep.y_eval.astype(int), pred_eval.astype(int))
        case = user_case_metrics(prep.meta_eval, pred_eval)

        elapsed = time.perf_counter() - t0

        print(f"\nIteration {it}/{args.iterations}")
        print(f"Time: {elapsed:.4f} sec")
        print(f"Threshold: {tuned['threshold']:.3f}")
        print("Transaction-level confusion matrix:", tx_cm)
        print("User-level confusion matrix:", {"TP": case.tp, "FP": case.fp, "TN": case.tn, "FN": case.fn})
        print(f"User latency avg hours: {case.avg_latency_hours if not math.isinf(case.avg_latency_hours) else 'inf'}")

        case_map = {r.user_id: (int(r.actual), int(r.pred)) for r in case.user_rows.itertuples(index=False)}
        for uid in sorted(case_map):
            a, p = case_map[uid]
            print(f"User {uid}: actual_scam_user={a} predicted_scam_user={p} correct={int(a==p)}")

        lines = explain_tp_fp(prep.meta_eval, pred_eval)
        if lines:
            print("TP/FP user explanations:")
            for ln in lines:
                print(ln)
        else:
            print("TP/FP user explanations: none")

        rank_key = (case.fn, case.fp, case.avg_latency_hours if not math.isinf(case.avg_latency_hours) else 1e9)
        if rank_key < best_key:
            best_key = rank_key
            best_model = {
                "iteration": it,
                "seed": model_seed,
                "config": {
                    "hidden": args.hidden,
                    "seq_len": args.seq_len,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "pos_weight": args.pos_weight,
                    "attention": True,
                    "activation": "tanh",
                    "input_features": prep.feature_cols,
                },
                "threshold": tuned["threshold"],
                "metrics_tx": tx_cm,
                "metrics_user": {"TP": case.tp, "FP": case.fp, "TN": case.tn, "FN": case.fn},
                "avg_latency_hours": case.avg_latency_hours,
                "train_users": train_users,
                "eval_users": eval_users,
                "norm_stats": prep.norm_stats,
                "params": {k: v.copy() for k, v in model.params.items()},
            }

    if best_model is None:
        raise RuntimeError("No model selected")

    with args.model_out.open("wb") as f:
        pickle.dump(best_model, f)

    model = NumpyRNNWithAttention(
        input_dim=len(best_model["config"]["input_features"]),
        hidden_dim=best_model["config"]["hidden"],
        seed=best_model["seed"],
    )
    model.params = {k: v.copy() for k, v in best_model["params"].items()}

    x_all, m_all, l_all, meta_all = prepare_all_windows(df, best_model["norm_stats"], seq_len=best_model["config"]["seq_len"])
    p_all = predict_proba(model, x_all, m_all, l_all, batch=1024)
    pred_all = (p_all >= best_model["threshold"]).astype(int)

    out = meta_all.copy()
    out["predicted_type"] = np.where(pred_all == 1, "scam", "nonscam")
    out = out.rename(columns={"transaction_type": "transaction_type", "txn_id": "txn_id", "user_id": "user_id"})
    out = out[["user_id", "txn_id", "timestamp", "transaction_type", "predicted_type"]]
    out = out.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    out["timestamp"] = out["timestamp"].astype(str)
    out.to_csv(args.audit_out, index=False)

    print("\nBest model iteration:", best_model["iteration"], "(least user FN, then user FP)")
    print("Best user-level confusion:", best_model["metrics_user"])
    print("Best transaction-level confusion:", best_model["metrics_tx"])
    print("Saved model:", str(args.model_out).replace("\\", "/"))
    print("Saved audit:", str(args.audit_out).replace("\\", "/"))


if __name__ == "__main__":
    main()
