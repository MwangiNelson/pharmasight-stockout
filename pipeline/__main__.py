"""Run the whole pipeline from scratch: python -m pipeline"""

from pipeline import build_dataset, simulate, train

for step in (simulate, build_dataset, train):
    print(f"\n== {step.__name__} ==")
    step.main()
