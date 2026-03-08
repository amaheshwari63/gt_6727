# Data Preprocessing Reproducibility Notes

## Artifacts
- `data/preprocess_datasets.py`
- `data/cs6727_DS1_PP.csv`
- `data/cs6727_DS2_PP.csv`
- `data/data_preprocessing_manifest.json`

## Input Dependencies
- `data/cs6727_DS1.csv`
- `data/cs6727_DS2.csv`

Upstream generation/data source lineage:
- `data/reference-person-age-ranges-2024.xlsx` (copied from BLS source)
- BLS source URL: https://www.bls.gov/cex/tables/calendar-year/mean-item-share-average-standard-error/reference-person-age-ranges-2024.xlsx
- BLS table index URL: https://www.bls.gov/cex/tables.htm
- BLS publication cited for density motivation: https://www.bls.gov/opub/btn/volume-4/consumer-expenditures-vary-by-age.htm
- FinCEN report URL used for fraud red-flag context: https://www.fincen.gov/sites/default/files/shared/FinCEN_Alert_Pig_Butchering_FINAL_508c.pdf

## Reproducible Command
```bash
python3 data/preprocess_datasets.py --seed 67270308
```

## Runtime and Seed
- Python: `3.11.5`
- Seed: `67270308`
- Determinism: preprocessing is deterministic for same inputs + code + seed.

## Output Files
- `data/cs6727_DS1_PP.csv`
- `data/cs6727_DS2_PP.csv`

## Added Columns
Added per transaction in context of user and payee:
- `new_payee_first_txn`
  - 1 if first transaction for this user+payee else 0.
- `new_payee`
  - 1 if payee is new for user within last 30 days relative to transaction date, else 0.
- `is_penatly`
  - 1 if payee/category text has tax/fee/penalty signals, else 0.
- `amount_delta`
  - current amount - previous amount for same user+payee, else 0 for first occurrence.
- `amount_ratio`
  - current amount / previous amount for same user+payee, else 1 for first occurrence.
- `increase_count`
  - running count of increases for same user+payee where current amount > previous amount.
- `aggregate3`
  - slope of amount over last 3 transactions for same user+payee, else 0 when <3.
- `txn_index`
  - starts at 1 and increments for each subsequent transaction for same user+payee.
- `new_payee_30d`
  - explicit duplicate of 30-day signal for traceability.

## Labels Added (DS1 only)
In `data/cs6727_DS1_PP.csv` only:
- `y_target_escalating_fraud`
  - 1 if `is_fraud_signal=1` and `new_payee=1` and `is_penatly=1` and (`increase_count>=1` or `amount_ratio>1`), else 0.
- `y_case_scam_user`
  - 1 if user has any scam transaction in dataset, else 0.

## Ambiguity Handling
The prompt defines `new_payee` twice with different meanings. Resolution used:
- `new_payee` uses the second definition (new payee in prior 30-day context).
- `new_payee_first_txn` preserves the first definition (first-ever user+payee transaction).

## What Was Ignored
- No demographic augmentation beyond existing generated data.
- No race/gender/education enrichment.
- No external joins.
- No schema drops from original DS1/DS2 columns.

## Counts and SHA256
- `data/cs6727_DS1_PP.csv`
  - rows: `90000`
  - users: `100`
  - fraud users: `20`
  - sha256: `bb8465127af6ef008f93a9336cdeb95ebeee39a6ca8a6026d63437a85086625b`
- `data/cs6727_DS2_PP.csv`
  - rows: `90000`
  - users: `100`
  - fraud users: `20`
  - sha256: `4ab064d274388d578367acddc4d0cfd4f69aa5c783a27100f07c25f4684cd5d8`

Additional artifact hashes:
- `data/preprocess_datasets.py`: `c388d6767e8a1e96212b191da2c3df0860c11b65131f31bfecfd3f235f50d1b7`
- `data/data_preprocessing_manifest.json`: `80c3603e65337adbfd08858bc45e144e036920be503a66024e2ebe9524879efc`
- `data/cs6727_DS1.csv`: `4ecb60b0954b1a2e34127ee077ba0697180e941ffec226be88b937342875731d`
- `data/cs6727_DS2.csv`: `0bea057299f52755a8303f7c96779ddbfce2fd3b8e7f7dd87ab847fdbf11cf2c`
- `data/generate_datasets.py`: `b39553e25c3384cb929bb4d2c1cab94c8323121f6bd100bdbe6f6562ae2614ee`
- `data/reference-person-age-ranges-2024.xlsx`: `0abaa86afa5d2b226c21952ff958a2c5300287b2b1fefc115a452f361a55ba86`

## Model Metadata
- Agent: Codex coding agent
- Model family: `GPT-5`
- Exact model version string: not exposed in this environment

## Full Prompt/Command Given to Codex
```text
Data pre-processing

1. This data will be used for research paper - use some seed to make sure this data generation is reproducible when some one tries to reproduce this data independently
2. Add the code that is used to generate this data
3. create a DATAPREPROCESSING_README.md file that provides reproducible command and all other relevant details like seed , how many users, how many rows SHA ... what is ignored , from where the distribution data came , from where Finn report came , which GPT model was used , what version - every possible detail. Also include command to preprocess the data,
4. In DATAPREPROCESSING_README. add this whole prompt/command given to codex
5. Put dataset , DATAPREPROCESSING_README. file in “data” folder
6. Remove references to local path just keep relative path from current folder

preprocess cs6727_DS1 and create a new cs6727_DS1_PP that has original columns from DS1 and these additional columns
preprocess cs6727_DS2 and create a new cs6727_DS2_PP that has original columns from DS1 and these additional columns

For each transaction, add these columns all are in context of a given user
1. new_payee
   - 1 if first txn from this payee to this user else 0.
2. is_penatly
   - 1 if payee in tax/fee/penalty else 0.
3. amount_delta
   - difference of current amount - previous amount for same user+payee, 0 in case first time user+payee transaction.
4. amount_ratio
   - ratio of current amount /previous amount for same user+payee, 1 in case first time user+payee transaction
5. increase_count
   - number of txn for same payee in increasing order
6. aggregate3
   - slope of amount over last 3 transactions from same payee 0 when <3 txn
7. txn_index
   - start with 1 and increment for each subsequent txn for that payee
8. new_payee
   - 1 if new payee for user last 30 days relative to txn date, else 0

In cs6727_DS1_PP  add two labels:
9. y_target_escalating_fraud
   If scam transaction and new_payee and is_penatly and (increase_count >=1 or amount_ratio >1) mark it 1 else 0
10. y_case_scam_user
   1 if user has any scam transaction in dataset else 0.
```
