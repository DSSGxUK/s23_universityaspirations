import logging

from uni_asp.analysis.risk_analysis import risk_analysis
from uni_asp.modelling.inference import run_lgbm_inference
from uni_asp.modelling.training import run_lgbm_training
from uni_asp.preprocessing.preprocess_data import preprocess_data

# A useful log format
LOG_FORMAT = "%(asctime)s: %(levelname)-8s - %(name)s - line %(lineno)3d - %(message)s"


def run_pipeline(retrain=True):
    preprocess_data()

    if retrain:
        run_lgbm_training("dest_ks4")
        run_lgbm_training("mat_y11")
        run_lgbm_training("eng_y11")

    run_lgbm_inference("dest_ks4")
    run_lgbm_inference("mat_gcse")
    run_lgbm_inference("eng_gcse")

    risk_analysis()


def main():
    # This function exists so that we have a function to point to in pyproject.toml
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    run_pipeline()


if __name__ == "__main__":
    main()
