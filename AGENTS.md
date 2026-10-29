# AGENTS.md — Engineering Rulebook & Operational Contract

> **Target Audience:** All AI coding agents, autonomous subagents, and human contributors operating on this codebase.  
> **Repository:** `Streamlit App for Network Intrusion Detection`  
> **Status:** Active & Enforced

---

## 1. Project Mission & Scope

This project is a machine learning–powered **Network Intrusion Detection System (NIDS)** and security analytics web application.

- **Primary Framework:** Streamlit (Python)
- **Machine Learning Library:** scikit-learn (Pipelines, ColumnTransformer, Tree Ensembles)
- **Dataset:** UNSW-NB15 (Apache Parquet & CSV)
- **Deployment Target:** Streamlit Community Cloud (via GitHub)
- **Primary Operational Goal:** Provide high-precision, low-false-positive defensive flow classification, visual exploratory telemetry analytics, model evaluation transparency, and batch flow auditing.
- **Strict Limitation:** This is an **offline defensive analytics tool**. Under no circumstances should any agent implement live packet sniffing, promiscuous socket listening, automated firewall reconfiguration, network scanning, or offensive exploit execution.

---

## 2. Non-Negotiable Engineering Principles

Every agent touching this repository must adhere to the following core tenets:

1. **Reconnaissance First:** Always inspect the actual file tree, imports, and existing implementations before modifying code. Never guess function names, module exports, or dataset column names.
2. **Zero Fabrication:** Never fabricate test coverage, model benchmarks, evaluation metrics, or feature completion. Every metric reported to the user or committed to documentation must be derived from actual executed code or verified artifacts (`models/model_metadata.json`).
3. **Preserve Working Functionality:** Never perform unsolicited whole-file rewrites. Implement minimal, contiguous, targeted edits that solve the user's specific request while preserving unrelated features, CSS styles, comments, and docstrings.
4. **Honest Reporting:** If a command fails or an external tool errors (e.g., browser drivers or package downloads), report the exact error truthfully. Never claim tests passed unless you executed them in the shell.
5. **Protect the Dataset:** The original dataset files (located in `~/UNSW-NB15/` or other host paths) are **strictly read-only**. Never overwrite, modify, move, or delete original dataset files.
6. **No Secrets or Oversized Binaries:** Never commit credentials, API keys, or large data files (>25 MB) to git. Keep `.gitignore` strictly enforced.
7. **Consistent Branding:** The canonical project name is **Streamlit App for Network Intrusion Detection**. The shorter title **Network Intrusion Detection System** may be used in headings. Do **not** reintroduce the retired "AEGIS" brand name.

---

## 3. Architecture & File Ownership

The codebase maintains strict separation between presentation, preprocessing, model orchestration, and testing:

```
.
├── app.py                        # PRESENTATION ONLY: Streamlit multi-page UI, layout & routing
├── src/
│   ├── preprocessing.py          # ML PREPROCESSING: Feature lists, ColumnTransformer, validation
│   ├── data_loader.py            # DATA ACCESS: File resolution, caching, sampling & fallbacks
│   ├── train_model.py            # ORCHESTRATION: Reproducible model training & evaluation CLI
│   ├── evaluation.py             # VISUALS & INFERENCE: Plotly figures & batch inference engine
│   └── generate_demo_samples.py  # FIXTURES: Generates small test fixtures for deployment
├── models/
│   ├── intrusion_pipeline.joblib # ARTIFACT: Serialized binary classification pipeline
│   ├── multiclass_pipeline.joblib# ARTIFACT: Serialized multiclass classification pipeline
│   └── model_metadata.json       # ARTIFACT: Verified test metrics, hyperparameters & top features
├── data/
│   ├── sample_eval.parquet       # FIXTURE: 5,000-flow stratified test subset for cloud deployment
│   └── sample_batch_demo.csv     # FIXTURE: 100-flow demo file for batch analysis testing
├── tests/
│   ├── test_data_loader.py       # TESTS: Dataset loading and statistical summary tests
│   ├── test_preprocessing.py     # TESTS: ColumnTransformer, clean_dataframe, unknown categories
│   └── test_prediction.py        # TESTS: Single-flow and batch prediction pipeline tests
├── .streamlit/config.toml        # CONFIG: Cyberpunk dark theme and server settings
├── requirements.txt              # CONFIG: Pinned production dependencies
└── pytest.ini                    # CONFIG: Test runner path configuration (pythonpath = .)
```

### Module Boundary Rules:
- **`app.py` must NOT contain raw ML training loops or raw model building logic.** Models must be trained offline via `src/train_model.py` and loaded into `app.py` via `@st.cache_resource`.
- **`src/preprocessing.py` is the single source of truth for feature definitions.** The 34 flow feature names (`CATEGORICAL_FEATURES` and `NUMERICAL_FEATURES`) and `validate_input_features` must reside here.
- **`src/data_loader.py` must maintain path portability.** Never rely solely on a hardcoded local machine path. Candidate directory lists and fallbacks to `data/sample_eval.parquet` are mandatory.

---

## 4. Machine Learning Integrity & Leakage Prevention

### Rule 1: No Target Leakage
- The feature matrix $X$ must strictly exclude:
  - `label` (binary ground truth)
  - `attack_cat` (multiclass ground truth)
  - Surrogate identifiers, sequence IDs, row numbers, or timestamps.

### Rule 2: Strict Split Isolation
- The official UNSW-NB15 dataset contains pre-partitioned files:
  - Training: `UNSW_NB15_training-set.parquet` (175,341 rows)
  - Testing: `UNSW_NB15_testing-set.parquet` (82,332 rows)
- **Transformers must be fitted strictly on training data:**
  ```python
  # CORRECT:
  preprocessor = create_preprocessor()
  preprocessor.fit(X_train)
  X_train_trans = preprocessor.transform(X_train)
  X_test_trans = preprocessor.transform(X_test)
  
  # FORBIDDEN (Data Leakage):
  preprocessor.fit_transform(full_dataset)
  ```

### Rule 3: Robust Out-of-Vocabulary Category Handling
- In UNSW-NB15, the test partition contains connection states (`CLO`, `ACC`) absent from the training partition.
- `OneHotEncoder` **must always** be instantiated with `handle_unknown="ignore", sparse_output=False`.

### Rule 4: Model Size & Hyperparameter Discipline
- To deploy seamlessly on Streamlit Community Cloud and avoid GitHub file limits, model artifacts must remain compact (<20 MB).
- Always constrain tree ensembles (e.g., `max_depth <= 16`, `min_samples_leaf >= 4`).
- Always compress Joblib artifacts (`joblib.dump(pipeline, path, compress=3)`).

### Rule 5: No Rerun Retraining
- Streamlit reruns the script on every user interaction.
- **Never trigger `fit()` inside `app.py` during normal runtime.** Use `joblib.load()` decorated with `@st.cache_resource`.

---

## 5. Dataset Handling Rules

1. **Immutable Ground Truth:** Treat raw dataset files as read-only. Write operations are restricted to `models/`, `data/sample_*`, and generated audit reports.
2. **Schema Verification:**
   - 3 Categorical features: `proto`, `service`, `state`.
   - 31 Numerical features: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `is_sm_ips_ports`.
   - 2 Target columns: `label` (int), `attack_cat` (str).
3. **Graceful Fallback Sequence:**
   `data_loader.load_dataset()` must probe in this order:
   - Primary: Path defined in `UNSW_DATA_DIR` environment variable
   - Secondary: `~/UNSW-NB15/`
   - Tertiary: `./data/`
   - Fallback: Bundled `data/sample_eval.parquet`

---

## 6. Streamlit & UX Standards

1. **Aesthetic Consistency:**
   - Maintain the refined dark cyberpunk theme configured in `.streamlit/config.toml` (`#0A0D14` background, `#131824` secondary background, `#E6EDF3` text).
   - Accents: Electric pink (`#FF1493`) for primary accents/threat alerts, cyber cyan (`#00E5FF`) for data/protocols, neon green (`#00FF66`) for benign status.
2. **Performance Caching:**
   - Use `@st.cache_resource` for deserializing models and pipelines.
   - Use `@st.cache_data` for data reading and sampling operations.
3. **State & Form Controls:**
   - Forms (`st.form`) must be used for multi-parameter inputs to prevent premature reruns while the operator enters values.
4. **Resilient Error States:**
   - If a model file is missing, display an informative `st.error()` explaining the remediation step (`python -m src.train_model`) instead of letting an unhandled traceback crash the page.

---

## 7. Python Coding Standards

- **Language Version:** Python 3.10 to 3.12 compatible.
- **Typing:** Provide explicit type hints (`from typing import Tuple, List, Dict, Optional, Any`) on all utility and core functions.
- **Imports:**
  - Absolute imports within `src/` (e.g., `from src.preprocessing import ALL_FEATURES`).
  - Standard library -> Third-party -> Local modules.
- **Exception Handling:**
  - Catch specific exceptions (`FileNotFoundError`, `ValueError`, `KeyError`).
  - If a generic `except Exception` is required for safe batch parsing, log or report the exception message cleanly.
- **Paths:**
  - Always use `pathlib.Path` for filesystem resolution. Avoid OS-specific hardcoded string separators (`\\` vs `/`).

---

## 8. Testing & Validation Policy

Every agent modifying code or configuration must execute the test suite:

```bash
# Execute test suite
pytest -v
```

### When to Test:
1. After modifying any file in `src/`.
2. After updating `app.py` logic or input schemas.
3. After adding new dependencies to `requirements.txt`.
4. Prior to concluding any user request.

### Rules for Tests:
- Never comment out, delete, or modify existing tests to hide a regression.
- New features or schema changes must be accompanied by corresponding unit tests in `tests/`.
- Ensure tests run with `pythonpath = .` (configured in `pytest.ini`).

---

## 9. Dependency & Deployment Rules

1. **Keep `requirements.txt` Lean:**
   - Only add packages that are strictly imported by `app.py`, `src/`, or `tests/`.
   - Never add heavy, unused frameworks (e.g., TensorFlow, PyTorch, LightGBM) unless explicitly requested and approved by the operator.
2. **Streamlit Community Cloud Constraints:**
   - Free tier memory is limited (~1 GB RAM). Do not load the full 257k-row dataset into memory at app startup.
   - Ensure `sample_eval.parquet` (5,000 rows, ~315 KB) is always available in `data/`.

---

## 10. Security & Input Sanitization

1. **Untrusted Uploads:**
   - All CSVs uploaded via the Batch Analysis view (`st.file_uploader`) must be treated as untrusted data.
   - Validate headers against `ALL_FEATURES`.
   - Coerce numeric columns via `pd.to_numeric(..., errors="coerce").fillna(0.0)`.
   - Strip whitespace from categorical strings.
2. **No Arbitrary Deserialization:**
   - Do **not** allow users to upload arbitrary `.pkl` or `.joblib` files to the web interface. Models must only be loaded from the local `models/` directory.

---

## 11. Change Management Workflow for Agents

Follow this rigorous cycle on every task:

```
1. RECONNAISSANCE ──> Inspect existing code, git status, and metadata
2. DESIGN         ──> Plan minimal, non-breaking modifications
3. IMPLEMENT      ──> Apply targeted edits (preserving existing architecture)
4. VERIFY         ──> Run pytest -v and verify server startup
5. DOCUMENT       ──> Update metadata or docs if behavior changed
6. REPORT         ──> Provide clear, honest summary with executed commands
```

---

## 12. Definition of Done (DoD)

A task is considered complete **only** when all of the following criteria are satisfied:

- [ ] Code strictly fulfills the user's explicit request.
- [ ] No regression introduced to existing pages or features.
- [ ] `pytest -v` passes with **100% green status** (all 10+ tests passing).
- [ ] Application starts cleanly (`streamlit run app.py` or health endpoint returns `ok`).
- [ ] No temporary debug scripts, scratch files, or credentials left behind.
- [ ] Project branding consistently matches: **Streamlit App for Network Intrusion Detection** (or **Network Intrusion Detection System** where shortened).
- [ ] Documentation (`README.md` and `AGENTS.md`) accurately reflects the real code state.
