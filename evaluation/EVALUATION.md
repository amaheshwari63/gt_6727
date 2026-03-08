# EVALUATION.md

## Outputs
All files are in `evaluation/`:
- `evaluation/evaluate_with_rnn.py`
- `evaluation/evaluation_audit_DS2.csv`
- `evaluation/evaluation_summary_DS2.json`
- `evaluation/EVALUATION.md`

## Input / Model
- Data: `data/cs6727_DS2_PP.csv`
- Model: `training/nn_rnn.pkl`

## Reproducible Command
```bash
cd evaluation
python3 evaluate_with_rnn.py --data ../data/cs6727_DS2_PP.csv --model ../training/nn_rnn.pkl --audit-out evaluation_audit_DS2.csv --summary-out evaluation_summary_DS2.json
```

## Environment
- Python: `3.11.5`
- Libraries: `numpy`, `pandas`
- Seeded model from training artifact (`training/nn_rnn.pkl`)

## Evaluation Logic
- Uses same sliding-window framing as training (`user_id + payee_id`, seq_len from model config)
- Applies saved normalization stats from model artifact
- Applies saved threshold from model artifact
- Produces transaction-level and user-level confusion matrices
- Produces per-user scenario details with TP/FP explainability text

## Iteration Context
`nn_rnn.pkl` stores the selected model from training iteration `1`.
Therefore evaluation confusion matrices are reported for that selected iteration context.

## Confusion Matrices (Selected Iteration = 1)
- Transaction-level: `TP=4577 FP=36 TN=84324 FN=1063`
- User-level: `TP=20 FP=19 TN=61 FN=0`

## User-Level Detection Summary
For each user, the script prints:
- `actual_scam_user`
- `predicted_scam_user`
- `correct`

(See runtime output and `evaluation_summary_DS2.json`.)

## TP/FP User Scenario Details (examples)
- User P002 [TP] - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $5.68 (2024-06-10 16:15:14, TX00086595) to $50.67 (2024-06-10 20:22:33, TX00086656). This change may indicate a potential fraudulent trend and is worth reviewing.
- User P003 [FP] - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $0.08 (2024-03-14 10:56:14, TX00072105) to $2.44 (2024-04-23 15:34:05, TX00078756). This change may indicate a potential fraudulent trend and is worth reviewing.
- User P100 [TP] - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $1.13 (2024-06-12 08:40:09, TX00086846) to $50.04 (2024-06-12 09:00:10, TX00086852). This change may indicate a potential fraudulent trend and is worth reviewing.

## Audit File
`evaluation/evaluation_audit_DS2.csv` contains one row per transaction with:
- `user_id`
- `txn_id`
- `timestamp`
- `transaction_type` (`scam|nonscam`)
- `predicted_type` (`scam|nonscam`)

## SHA256
- `evaluation/evaluate_with_rnn.py`: `021fb7ceeb7df70c2c89ea03ccf0cc91efa4385979c5dfbdd93c19e1a66ca382`
- `evaluation/evaluation_audit_DS2.csv`: `8a2d7d814b89e655bb1b58f1a6306e83033247ca7c91ec2376f940eaa3e41203`
- `evaluation/evaluation_summary_DS2.json`: `90c9add5b19518f0f9bc4c0515dbe61a08d134f7f8db33b4b23503676a33ca54`
- `training/nn_rnn.pkl`: `da9730a1ebdfe0e49bab62c88c88e8c9487d4aaec1cefdca13a033eee7481560`

## Model/Agent Metadata
- Agent: Codex
- Model family: `GPT-5`
- Exact internal model version string: not exposed in this runtime

## Full Prompt Given to Codex
```text
Evaluation
All the files for this will go in evaluation folder

Write python code (evaluate_with_rnn.py) that evaluates ../data/cs6727_DS2_PP.csv with ../training/nn_rnn.pkl
This code will be used in research paper so needs to be repeatable by anybody.
Generate a EVALUATION.md file that provides all details about which GPT model was used , what version - every possible detail
In EVALUATION.md  add this whole prompt/command given to codex

add transaction level and user level confusion matrix for each iteration

For each of TP and FP at user scenario - add details that can be explained
like this

Generate a file called evaluation_audit_DS2.csv - for each txn add user id, txn_id, timestamp, what was the transaction type [scam|nonscam] and what was predicted type

User 81 - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $<value> (timestamp , txnid) to $<value> (timestamp , txnid) This change may indicate a potential fraudulent trend and is worth reviewing
```
