# TRAINING.md

## Outputs
All files are in `training/`:
- `training/train_rnn.py`
- `training/evaluation_audit_DS1.csv`
- `training/nn_rnn.pkl`
- `training/TRAINING.md`

## Input
- `data/cs6727_DS1_PP.csv`

## Reproducible Command
```bash
cd training
python3 train_rnn.py --input ../data/cs6727_DS1_PP.csv --seed 67270308 --iterations 5 --hidden 32 --model-out nn_rnn.pkl --audit-out evaluation_audit_DS1.csv
```

## Environment
- Python: `3.11.5`
- Libraries: `numpy`, `pandas`
- No `torch` in training code
- Seed: `67270308`

## Model
- RNN with tanh recurrent state (hidden size `32`)
- Additive attention (Bahdanau-style)
- Attention weights via softmax
- Dense sigmoid output
- Optimizer: Adam (numpy implementation)
- Reference: https://arxiv.org/abs/1409.0473

## Split
Per iteration, user-level split:
- Train: `32` non-scam + `8` scam users
- Validation: `8` non-scam + `2` scam users

## Anti-Leakage Update
To avoid overfitting/perfect leakage behavior, training input features exclude:
- `red_flag_new_payee`
- `red_flag_escalated_payments`
- `red_flag_tax_fee_penalty`

Used feature set:
- `amount_usd`, `new_payee`, `is_penatly`, `amount_delta`, `amount_ratio`, `increase_count`, `aggregate3`, `txn_index`, `new_payee_30d`, `month_index`

## Training Setup
- Iterations: `5`
- Sequence framing: per `user_id + payee_id` sliding windows
- Sequence length: `8`
- Epochs per iteration: `14` with early stopping (`patience=2`)
- Learning rate: `0.008`
- Batch size: `128`
- Positive class weight: `5.0`
- Threshold tuning on training windows each iteration (objective: least user FN, then user FP, then latency)

## Iteration Results
### Iteration 1
- Time: `19.0994 sec`
- Threshold: `0.900`
- Transaction-level confusion: `TP=121 FP=3 TN=8865 FN=11`
- User-level confusion: `TP=2 FP=1 TN=7 FN=0`

### Iteration 2
- Time: `19.7421 sec`
- Threshold: `0.900`
- Transaction-level confusion: `TP=120 FP=2 TN=8866 FN=12`
- User-level confusion: `TP=2 FP=1 TN=7 FN=0`

### Iteration 3
- Time: `22.2614 sec`
- Threshold: `0.900`
- Transaction-level confusion: `TP=125 FP=5 TN=8863 FN=7`
- User-level confusion: `TP=2 FP=3 TN=5 FN=0`

### Iteration 4
- Time: `21.8905 sec`
- Threshold: `0.900`
- Transaction-level confusion: `TP=124 FP=7 TN=8861 FN=8`
- User-level confusion: `TP=2 FP=4 TN=4 FN=0`

### Iteration 5
- Time: `23.9947 sec`
- Threshold: `0.900`
- Transaction-level confusion: `TP=123 FP=3 TN=8865 FN=9`
- User-level confusion: `TP=2 FP=2 TN=6 FN=0`

## Best Model Saved
Selection rule: least user-level FN, then least user-level FP.
- Best iteration: `1`
- Best user-level confusion: `TP=2 FP=1 TN=7 FN=0`
- Best transaction-level confusion: `TP=121 FP=3 TN=8865 FN=11`
- Saved model: `training/nn_rnn.pkl`

## Evaluation Audit
`training/evaluation_audit_DS1.csv` columns:
- `user_id`
- `txn_id`
- `timestamp`
- `transaction_type` (`scam|nonscam`)
- `predicted_type` (`scam|nonscam`)

Row count: `90000`
Counts:
- `nonscam -> nonscam`: `88647`
- `nonscam -> scam`: `33`
- `scam -> scam`: `1212`
- `scam -> nonscam`: `108`

## TP/FP User Scenario Explanations (Best Iteration)
- User P002 [TP] - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $110.17 (2024-06-09 09:27:22, TX00086349) to $1668.56 (2024-06-10 16:28:00, TX00086630). This change may indicate a potential fraudulent trend and is worth reviewing.
- User P039 [TP] - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $137.49 (2024-05-03 16:53:35, TX00080440) to $1528.05 (2024-05-04 20:13:15, TX00080645). This change may indicate a potential fraudulent trend and is worth reviewing.
- User P049 [FP] - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $148.23 (2023-03-08 19:28:17, TX00011320) to $361.46 (2023-03-22 10:26:25, TX00013481). This change may indicate a potential fraudulent trend and is worth reviewing.

## SHA256
- `training/train_rnn.py`: `d071e627217d94ca2823e9cacaef4e790db257a6831d454dcb0c16bf2098206e`
- `training/evaluation_audit_DS1.csv`: `63713e08702d2777d75a46666506df543d66eefef661d172aea5f891112a6e42`
- `training/nn_rnn.pkl`: `da9730a1ebdfe0e49bab62c88c88e8c9487d4aaec1cefdca13a033eee7481560`

## Model/Agent Metadata
- Agent: Codex
- Model family: `GPT-5`
- Exact internal model version string: not exposed in this runtime

## Full Prompt Given to Codex
```text
Training
All the files for this will go in training folder
Train a model with python code train_rnn.py that takes ../data/cs6727_DS1_PP.csv, as input trains RNN model as referenced here
https://arxiv.org/abs/1409.0473 with 32 hidden layer , additive attention layer, attention weights, output dense layer, gated unit tanh
Iterate 5 times , record time for each iteration
For training  take data for 32 users that do not have scam , 8 that had scam
For validation use 8 that do not have scam and 2 that had scam (80:20 random split)
For each iteration print out confusion matrix
For evaluation
For each user print if user had scam transaction and if model was able to detect it correctly
Generate a file called evaluation_audit_DS1.csv - for each txn add user id, txn_id, timestamp, what was the transaction type [scam|nonscam] and what was predicted type
Save the model with least FN, followed by least FP nn_rnn.pkl
This code will be used in research paper so needs to be repeatable by anybody.
Generate a TRAINING.md file that provides all details about which GPT model was used , what version - every possible detail
In TRAINING.md  add this whole prompt/command given to codex
So 3 output expected (a) train_rnn.py (b) evaluation_audit_DS1.csv (c) TRAINING.md
Put all files  in “training” folder
Remove references to local path just keep relative path from current folder

add transaction level and user level confusion matrix for each iteration

For each of TP and FP at user scenario - add details that can be explained
like this

User 81 - Unusual spending detected: Your transactions in the Fees/Penalties category increased from $<value> (timestamp , txnid) to $<value> (timestamp , txnid) This change may indicate a potential fraudulent trend and is worth reviewing
```
