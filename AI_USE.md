# AI use disclosure

## Tool

Claude Code, Anthropic's coding assistant (model: Claude Opus 5.5), run in my terminal with access to this
project folder. It could read and write files and run commands: Python, pytest and Docker.

## What the AI did

- Wrote the synthetic data simulator (`pipeline/simulate.py`), the data pipeline (`pipeline/build_dataset.py`,
  `pipeline/features.py`), the training and evaluation script (`pipeline/train.py`) and the tests.
- Cut my earlier PharmaSight service, a weekly demand-forecasting API, down to this stockout service: it removed
  the database, retraining and feedback parts, replaced the model and endpoints, and rewrote the Dockerfile and
  `docker-compose.yml`.
- Ran the pipeline, the tests and the Docker build, and fixed what failed. One example: the first simulator
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
