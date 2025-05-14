## Run
1. Create a GCP project
2. Enable sheets API
3. Go to `IAM & Admin` -> `Service Accounts` -> `Create service account` and then generate its application token in a GCP project
4. Save the token on your device and reference it by `GOOGLE_APPLICATION_CREDENTIALS` environment variable (i.e. set this variable in `.env` to the path to your .json file with service account credentials)
5. Create a new worksheet where you wish to save the results of the evaluation.
6. `export SHEET="your_worksheet_url"`
7. python -m translations.main use_sheets=True
