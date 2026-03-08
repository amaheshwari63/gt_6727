# Data Generation Reproducibility Notes

## Artifacts
- `data/reference-person-age-ranges-2024.xlsx`
- `data/generate_datasets.py`
- `data/cs6727_DS1.csv`
- `data/cs6727_DS2.csv`
- `data/data_generation_manifest.json`

## Reproducible Command
```bash
python3 data/generate_datasets.py --seed 67270308
```

## Runtime and Seed
- Python: `3.11.5`
- Seed: `67270308`
- Determinism: dataset generation is deterministic for same seed + same input spreadsheet + same code.

## Data Sources
- BLS spending table source used for age-group expenditure behavior:
  - https://www.bls.gov/cex/tables/calendar-year/mean-item-share-average-standard-error/reference-person-age-ranges-2024.xlsx
- BLS tables index:
  - https://www.bls.gov/cex/tables.htm
- Motivation for dense monthly transaction simulation (50 txns/month):
  - https://www.bls.gov/opub/btn/volume-4/consumer-expenditures-vary-by-age.htm
- FinCEN report used for scam red-flag simulation:
  - https://www.fincen.gov/sites/default/files/shared/FinCEN_Alert_Pig_Butchering_FINAL_508c.pdf

## What Was Modeled
- Population scope: age `65+` only.
- Age subgroups used from BLS columns:
  - `65-74 years`
  - `75 years and older`
- BLS annual means used:
  - `65-74`: `65354`
  - `75+`: `55834`
- Assumption: annual expenditure is uniformly distributed by month (`annual/12` baseline).
- Transaction density: `50` transactions per person per month.
- Transaction ID rule: IDs assigned only after full timestamp sort, so IDs increase with timestamp.

## What Was Ignored
- Race
- Gender/sex
- Education
- Housing tenure
- Any other non-age demographic splits from BLS table

## Dataset Definitions

### Data Set 1 (`data/cs6727_DS1.csv`)
- Dataset name in rows: `cs6727_DS1`
- Users: `100`
- Months: `18` (`2023-01` to `2024-06`)
- First 12 months: all 100 users normal pattern
- Final 6 months:
  - 80 users continue normal pattern
  - 20 users receive scam-indicating transactions

### Data Set 2 (`data/cs6727_DS2.csv`)
- Dataset name in rows: `cs6727_DS2`
- Users: `100`
- Months: `18` (`2023-01` to `2024-06`)
- 80 users normal for all 18 months
- 20 users scam pattern for all 18 months

## Fraud Red Flags Simulated
For scam transactions:
- New payees
- Escalated/increasing payments to new payees over scam progression
- Payee categories tied to investment-linked `taxes`, `fees`, or `penalties`

## Output Schema
- `transaction_id`
- `dataset_name`
- `person_id`
- `person_age`
- `age_group`
- `timestamp`
- `amount_usd`
- `spend_category`
- `payee_id`
- `payee_category`
- `is_fraud_signal`
- `red_flag_new_payee`
- `red_flag_escalated_payments`
- `red_flag_tax_fee_penalty`
- `currency`
- `status`

## Output Counts and SHA256
- `data/cs6727_DS1.csv`
  - rows: `90000`
  - sha256: `4ecb60b0954b1a2e34127ee077ba0697180e941ffec226be88b937342875731d`
- `data/cs6727_DS2.csv`
  - rows: `90000`
  - sha256: `0bea057299f52755a8303f7c96779ddbfce2fd3b8e7f7dd87ab847fdbf11cf2c`

Additional artifact hashes:
- `data/reference-person-age-ranges-2024.xlsx`: `0abaa86afa5d2b226c21952ff958a2c5300287b2b1fefc115a452f361a55ba86`
- `data/generate_datasets.py`: `b39553e25c3384cb929bb4d2c1cab94c8323121f6bd100bdbe6f6562ae2614ee`
- `data/data_generation_manifest.json`: `70fbe7a712f845c0b8f89c74ee9e39db78ca2cfa486aed1ccc4e8f31c2f23c6c`

## Model / Tool Metadata
- Agent: Codex coding agent
- Model family: `GPT-5`
- Exact model build/version string: not exposed by this environment

## Full Prompt Given to Codex
```text
Data generation

generate few data sets that will be used for research

1. Use this spreadsheet  (https://www.bls.gov/cex/tables/calendar-year/mean-item-share-average-standard-error/reference-person-age-ranges-2024.xlsx) as source of spending habits by age group for a year, this spreadsheet came from https://www.bls.gov/cex/tables.htm
2. generate simulated data for age group 65 and older (65 and all above 65 age categories)
Assume the annual expenditure is uniformly  distributed across months. Make the data reasonably dense about 50 transactions per month
3. To generate the data set ignore race/gender/education etc - just focus on age group
4. Why 50 transactions - https://www.bls.gov/opub/btn/volume-4/consumer-expenditures-vary-by-age.htm
5. Transaction id should increase with timestamp
6.Here is list of  red flags that can be used to simulate data that points to a scam using this pdf https://www.fincen.gov/sites/default/files/shared/FinCEN_Alert_Pig_Butchering_FINAL_508c.pdf
6. This data will be used for research paper - use some seed to make sure this data generation is reproducible when some one tries to reproduce this data independently
7. Add the code that is used to generate this data
8. create a DATAGENERATION_README.md file that provides reproducible command and all other relevant details like seed , how many users, how many rows SHA ... what is ignored , from where the distribution data came , from where Finn report came , which GPT model was used , what version - every possible detail. Also include command to generate the data
9. In DATAGENERATION_README. add this whole prompt/command given to codex
10. Put dataset , DATAGENERATION_README. file in “data” folder
11. Remove references to local path just keep relative path from current folder

Details of data sets to be generated
Data Set 1
    1. name it cs6727_DS1
    2. Generate data for 100 people - where first 12 month data is normal data as per earlier assumptions (the spreadsheet given and 50 transactions per month)
    3. For 80 people add another 6 months of data that is continuing same pattern i.e. no scam
    4.  For remaining 20 -  insert 6 months of data that includes fraud indicating transactions
    5. For fraud indicating transactions use these red flags (a) New payees (b) increased - escalated payments to new payees (c) category of payees in - “taxes,” “fees,” or “penalties” tied to the investment.

Data Set 2
1. name it cs6727_DS2
2. same as data set 1  with slight variation 80 users with 18 months of data without any scam ; 20 users where   scam data is continuing for 18 months - use the same red flags used for Data Set1
```
