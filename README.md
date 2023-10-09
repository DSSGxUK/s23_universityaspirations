# University Aspirations DSSGx-UK Project 2023

University enrolment in the UK is higher today than ever before. Despite this, many students across the country do not pursue their aspirations towards higher education (HE), even if they have the academic ability. Schools would like to make sure that pupils make use of their academic potential to achieve their desired pathway, and could help overcome potential hurdles these students face in pursuing higher education – if they knew which students may need support.

In this project, we identify year 9 pupils who might be lacking the confidence to fulfil their academic potential, as those with a high chance of getting good GCSE grades but a low chance of attending sixth form. United Learning can then look into these students on a case-by-case basis, providing career support where appropriate.

## Installation
Prior to installation, you can clone the repository. We use python 3.10.11, and the requirements file specifies versions of all other packages. A virtual environment ul-env (replace ul-env with any other name) can be created in the command line:
```bash
conda create -n ul-env python=3.10.11
```
Then, to activate the virtual environment, run:
```bash
conda activate ul-env
```

Following this, change your current directory to the folder with the `requirements.txt` file. Ensure that you activated your python 3.10 virtual environment, then run the following to install the requirements and the `uni_asp` package:
```bash
pip install -r requirements.txt
pip install -e .
```

Then, our full pipeline can be run with the newly installed `run-uni-asp` command.
```bash
run-uni-asp
```

To run tests, install the tests optional dependencies, the use pytest:
```bash
pip install -e .[test]
pytest tests
```

### Files Structure

The `01_raw` files are preprocessed, and the final dataframe is in the `02_processed` folder. This dataset is what the models are run on, and results are output into the `03_modelling` folder. The risk analysis will then be available in the `04_analysis` folder.

```
/ul_exp
.
├── 01_raw
│   ├── conduct
│   ├── destinations
│   ├── GCSE_Alevels
│   ├── KS3_EOY
│   ├── other_data
│   ├── pupil_char
│   ├── school_level
│   └── Y10_EOY
├── 02_processed
│   ├── full_df.csv
│   └── full_df.parquet
├── 03_modelling
│   ├── encoders
│   |   ├── dest_ks3.pkl
│   |   ├── eng_y11.pkl
│   |   └── mat_y11.pkl
│   ├── final_csvs
│   |   └── coh_8_results
│   |       ├── model_results_dest_ks4.csv
│   |       ├── model_results_dest_ks4.parquet
│   |       ├── model_results_eng_gcse.csv
│   |       ├── model_results_eng_gcse.parquet
│   |       ├── model_results_mat_gcse.csv
│   |       └── model_results_mat_gcse.parquet
│   ├── saved_models
│   |   ├── lgb_dest_ks4.txt
│   |   ├── lgb_gcse_eng.txt
│   |   └── lgb_gcse_mat.txt
└── 04_analysis
    ├── binary_analysis
    ├── multi_category_analysis
    ├── risk_analysis.csv
    └── risk_analysis.parquet
```

Notes: 
- `04_analysis` has a more elaborate file structure. We recommend referring to the handoff document for a thorough breakdown of binary and multi_category analysis
- In `01_raw`, we only displayed the first level of folders, thus we recommend looking into this directory to see all the files.

### NBStripOut
To prevent us from committing sensitive outputs to git, we use `nbstripout` to automatically strip the output. The output is only stripped from your staged & committed versions. To set this up, run:
```bash
nbstripout --install --attributes .gitattributes
```

### Linting etc. (black & pre-commit)
In development we use a tool called pre-commit to orchestrate code linting as well as some other checks (such as to prevent committing directly to main). The main linter we use is `black`.
The hooks can be installed to run before each git commit with
```bash
pre-commit install --install-hooks
```
Alternatively, they can be installed as a pre-push hook using
```bash
pre-commit install --install-hooks --hook-type pre-push
```

If your code is not formatted correctly then the git commit / git push will fail. However, black will modify your working directory to format the code correctly.
