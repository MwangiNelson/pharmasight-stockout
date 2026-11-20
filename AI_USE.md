# AI use disclosure

## Tool

Claude (model: Claude Opus 5.5), Suggested code snippets for pipeline setup to deployment configuration pipelines. 
## What the AI did

- Wrote the synthetic data simulator (`pipeline/simulate.py`), the data pipeline (`pipeline/build_dataset.py`,
  `pipeline/features.py`), the training and evaluation script (`pipeline/train.py`) and the tests.
- Cut my earlier PharmaSight service, a weekly demand-forecasting API, down to this stockout service: it removed
- Suggested code snippets for the pipeline, the tests and the Docker build, and suggested fixes for what failed. One example: the first simulator
  produced stockouts on about half of all item-days. It traced this to the simulated pharmacist estimating demand
  from dispensing that had been cut short by stockouts, and changed the estimate to use in-stock days only.
- Drafted `README.md`, `REPORT.md` and this file.

## What I did

- Chose the problem and the approach, set out in my proposal: predicting 7-day stockouts with a days-of-stock
  baseline, logistic regression and gradient-boosted trees, chronological splits, and a Dockerised FastAPI
  service.
- Decided to use synthetic data so the repository could be public without any hospital records, and to keep the
  Docker and FastAPI setup from my earlier PharmaSight project.
- Reviewed the code, results and report before submission.

## How the output was checked

- Every number in `README.md` and `REPORT.md` comes from files the pipeline writes (`reports/metrics.json`,
  `reports/data_quality.json`, `reports/model_selection.csv`), not from the AI's memory. Claims in the report
  that go beyond those files were checked with separate queries on the processed data, and two were corrected
  as a result.
- The tests check that no feature changes when future records change, that labels match a hand-built case,
  that the splits share no label window, and that the API authenticates, validates input and returns sorted
  probabilities.
- The whole pipeline was run inside Docker as well as locally; every metric matched exactly.

No hospital or patient records were used to build, train or test anything in this repository, and none were
given to the AI tool for this work.
