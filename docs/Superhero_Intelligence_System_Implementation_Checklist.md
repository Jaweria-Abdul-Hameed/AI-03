# Superhero Intelligence System — Implementation & Deliverables Checklist

**Machine Learning Project · Assignment 03**  
*Can six ability scores tell a hero from a villain?* · **Groups of 2–3 students** · **Due 06 December 2026 at 11:59 PM** · **85 Marks (Sections 2–5)**

> **Core Non-Negotiables (Apply Across Every Task):**
> - **Inputs:** Every model uses **exactly 6 numeric ability scores** (`intelligence_score`, `strength_score`, `speed_score`, `durability_score`, `power_score`, `combat_score`) — nothing more, nothing less.
> - **Seeds:** Use `random_state=42` in every function that accepts it, and `tf.keras.utils.set_random_seed(42)` for neural networks.
> - **Test Set Lock:** Never merge `train.csv` and `test.csv`, never make your own train/test split, never call `.fit()` / `.fit_transform()` on `test.csv`, and never tune on `test.csv`. Open `test.csv` **only once** at the very end of Task 1 and Task 3 (`test.csv` is **not used** in Task 2).
> - **Deliverable Standards:** Number every graph (`G1–G16`) and table (`T1–T11`) **exactly** as in the brief; include title, labelled axes, legend (when > 1 colour), fixed hex colours / `tab10` cluster order, and **1–2 sentences underneath** stating what it shows and what you conclude. Answer all 12 report questions (`Q1–Q12`).

---

## Phase 1: Project Setup, Submission Logistics & Global Standards
*Section 1 & Section 2 Global Rules · Slides 1–2, 5, 11*

- [ ] **Form Group (2–3 People) & Track Submission Deadline / Late Penalties**
  - Work in a group of **2–3 people** and submit completed work by **06 December 2026 at 11:59 PM**.
  - Respect the strict late penalty schedule: up to 5 minutes late = **−10%**; > 5 minutes to 60 minutes late = **−20%**; > 1 hour to 12 hours late = **−50%**; > 12 hours late = **0 marks**.

- [ ] **Download Dataset & Establish Exact Project Folder Layout**
  - Download the Superheroes dataset from Hugging Face: `https://huggingface.co/datasets/jrtec/Superheroes` (`train.csv` and `test.csv`).
  - Keep characters from **all publishers** (Marvel, DC, and others) — **do NOT filter by publisher**.
  - Arrange files in the exact required structure and read with relative paths `pd.read_csv("data/train.csv")` and `pd.read_csv("data/test.csv")`:
    ```text
    my_project/
    ├── superhero_project.ipynb
    └── data/
        ├── train.csv
        └── test.csv
    ```
  - **Do NOT** merge `train.csv` and `test.csv`, and **do NOT** create your own train/test split. Use `train.csv` for all training, CV, validation splitting, and clustering; lock `test.csv` for the single final evaluation of Tasks 1 & 3 only.

- [ ] **Lock Global Random Seeds & Approved Python Tool Stack**
  - Pass `random_state=42` to **every** function/estimator/splitter/PCA/KMeans call that supports it.
  - Set `tf.keras.utils.set_random_seed(42)` before building/training neural networks.
  - Use the approved tool stack: `pandas`, `scikit-learn`, `SciPy`, `matplotlib` (`seaborn` allowed), and `TensorFlow/Keras`.

- [ ] **Enforce Mandatory Styling, Hex Colours, Captions & Confusion Matrix Format**
  - Use exact fixed hex colours across all plots:
    - Hero / Good: `#1F5FBF` · Villain / Bad: `#D62839`
    - Baseline: `#9E9E9E` · Decision Tree: `#2E7D32` · KNN: `#F9A825` · Network A: `#6A1B9A` · Network B: `#00838F`
    - Clusters: matplotlib `tab10` order (`Cluster 1` = 1st colour, `Cluster 2` = 2nd colour, …).
  - Every graph (`G1–G16`) and table (`T1–T11`) must have: (1) a clear title, labelled axes, and legend whenever > 1 colour appears; (2) exact numbering `G1–G16` and `T1–T11` matching the brief; and (3) **1–2 sentences underneath** explaining **what it shows** and **what you conclude**.
  - Every confusion matrix (`G2`, `G3`, `G15`) must strictly follow: **rows = actual class**, **columns = predicted class**, class order = **`Good (Hero)` first, then `Bad (Villain)`** (`[[TN, FP], [FN, TP]]`, where `Villain (1)` is the positive class), with the **count displayed in every cell**.

---

## Phase 2: Data Preparation, Cleaning Pipeline & Leakage Lock
*10 Marks · Section 2 · Slides 3–10*

- [ ] **Restrict Columns Strictly to the 6 Inputs + Target + Label**
  - **USE ONLY** these columns:
    - 6 Input features (numbers `0–100`): `intelligence_score`, `strength_score`, `speed_score`, `durability_score`, `power_score`, `combat_score`.
    - Target: `alignment` (`Good` / `Bad` / `Neutral`, some blank).
    - Label only (never an input): `name`.
  - **DO NOT USE** any excluded column: `overall_score` (derived from the 6 scores — leaks the answer), `history_text`, `powers_text`, `superpowers`, any `has_…` power flag column, `creator`, `gender`, `height`, `weight`, `colours`, or any other column.

- [ ] **Step 1 — Inspect Both Files & Produce Table `T1`**
  - Apply identical inspection to both `train.csv` and `test.csv`.
  - Report in **Table `T1`**: shape, column names, data types, and missing counts for each file, **PLUS** how many `0`s and how many blanks became `NaN` in each of the 6 score columns (Slide 9 note).

- [ ] **Step 2 — Remove Duplicates (Within-File & Cross-File)**
  - Apply identically to both files:
    - Same `name` appearing twice within a file → **keep the first row**.
    - A `name` appearing in **both** `train.csv` and `test.csv` → **remove it from `train.csv`** (keep in `test.csv`).
    - Versions with different names (e.g., `"Captain X"` and `"Captain X (Film)"`) are distinct characters → **keep both**.
  - Record exact row counts after Step 2 for both files in **Table `T2`**.

- [ ] **Steps 3 & 4 — Mark Missing (`0` & Blank → `NaN`) & Remove Weak Rows**
  - Apply identically to both `train.csv` and `test.csv`:
    - **Step 3 (Mark missing):** Convert all 6 score columns to numeric; turn both blanks and `0` into `NaN` (a `0` means *"unknown"*, not *"no ability"*).
    - **Step 4 (Remove weak rows):** Drop every character with **3 or more (`≥ 3`)** of its 6 scores missing (keep rows with 0, 1, or 2 missing scores; e.g., 0 missing → keep as is; 2 missing → keep for pipeline median imputation; 4 missing → remove).
  - Record exact row counts after Step 4 for both files in **Table `T2`**.

- [ ] **Step 5 — Median Imputation Inside Pipeline (Training Data Only) & Four Leakage Rules**
  - Fill remaining `NaN`s using **training-set medians** inside a scikit-learn `Pipeline` (`SimpleImputer(strategy="median")`) — median is robust to extreme scores and keeps `test.csv` unseen.
  - Verify all **Four Rules** (Slide 10):
    1. Never call `.fit()` or `.fit_transform()` on test data (only `.transform()` / `.predict()` / score).
    2. Never choose settings by looking at test scores.
    3. Never use `overall_score`.
    4. Clustering never sees `alignment`.

- [ ] **Create Clean Set A & Clean Set B, Report Dropped Alignments & Complete Table `T2`**
  - **Clean Set A (→ Task 2 Clustering):** Cleaned `train.csv` after Steps 1–4 containing **every character** (`Good`, `Bad`, `Neutral`, or blank `alignment`). Put `alignment` aside and inspect it only in Step 8 after clustering. `test.csv` is **not used**.
  - **Clean Set B (→ Tasks 1 & 3 Classification & Neural Networks):** Cleaned `train.csv` and `test.csv` after Steps 1–4 containing **only rows where `alignment` is `Good` or `Bad`**.
    - Encode target: `Good → 0 (Hero)`, `Bad → 1 (Villain)`.
    - **Explicitly report** how many `Neutral` and blank rows were removed.
  - Populate **Table `T2` (Rows after each step)** with exact rows: `Loaded`, `After step 2 (duplicates)`, `After step 4 (weak rows)`, `Clean Set A` (`train.csv` count; `test.csv` = `not used`), and `Clean Set B` (both files).

---

## Phase 3: Task 1 — Classification (Baseline, Decision Tree & KNN)
*25 Marks · Section 3 · Slides 12–19*

- [ ] **Quantify & Plot Class Distribution on Clean Set B (`T3`, `G1`)**
  - Load Clean Set B (`X` = 6 scores, `y` = `0`/`1`).
  - Produce **Table `T3`**: class counts and `%` of Good (Hero) and Bad (Villain) in `train.csv` and `test.csv`.
  - Produce **Graph `G1`**: Class counts — Good (`#1F5FBF`) and Bad (`#D62839`) bars for `train.csv` and `test.csv` side by side, with `%` labels on bars + 1–2 sentence caption.

- [ ] **Build Exact Pipelines & Hyperparameter Grids for All 3 Models**
  - **Baseline (`#9E9E9E`):** `DummyClassifier(strategy="most_frequent")` — no hyperparameter search.
  - **Decision Tree (`#2E7D32`, no scaling needed):**
    - Pipeline (exact order): `SimpleImputer(strategy="median") → DecisionTreeClassifier(criterion="gini", random_state=42)`.
    - Grid (**15 combinations**): `max_depth: [3, 5, 7, 10, None]`, `min_samples_leaf: [1, 5, 10]`.
  - **KNN (`#F9A825`, `StandardScaler` required):**
    - Pipeline (exact order): `SimpleImputer(strategy="median") → StandardScaler() → KNeighborsClassifier(metric="euclidean")`.
    - Grid (**12 combinations**): `n_neighbors: [3, 5, 7, 9, 11, 15]`, `weights: ["uniform", "distance"]`.
  - **Strictly NOT Allowed:** Class weights, oversampling, undersampling, extra features, or tuning on `test.csv`.

- [ ] **Run 5-Fold Stratified GridSearchCV & Refit on All `train.csv` (`T4`)**
  - Execute exact CV search for both Decision Tree and KNN on Clean Set B `train.csv` (`test.csv` untouched):
    ```python
    GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        refit=True,
    )
    ```
  - Record best hyperparameters and best 5-fold CV Macro-F1 in **Table `T4`**.

- [ ] **Test Once on `test.csv`, Score Training Fit & Produce `T5`, `G2`, `G3`, `G4`**
  - Predict `test.csv` **once** using the refitted pipelines; compute metrics using `classification_report(y_test, y_pred, digits=3)` (Villain = 1 is positive class for TP/FP/FN/TN; Macro-F1 = `(F1_Hero + F1_Villain) / 2` is the **★ MAIN** metric).
  - Score each final refitted model on all of `train.csv` to obtain `Train Macro-F1`.
  - Populate **Table `T5` (Task 1 results)** — exact 10 columns for `Baseline`, `Decision Tree`, and `KNN`: `Model | Train Macro-F1 | Accuracy | Precision Hero | Recall Hero | F1 Hero | Precision Villain | Recall Villain | F1 Villain | Macro-F1`.
  - Produce **Graph `G2`** (Confusion matrix — Decision Tree, required format, counts in every cell), **Graph `G3`** (Confusion matrix — KNN, same format), and **Graph `G4`** (Model comparison — grouped bars for Accuracy, Macro-Precision, Macro-Recall, Macro-F1 across the 3 models with `y` from `0` to `1` and fixed model colours).

- [ ] **Apply 4 Comparison Rules & Answer Report Questions `Q1–Q4`**
  - Enforce exact Task 1 decision rules (Slide 18):
    1. **Winner:** Highest test Macro-F1.
    2. **Tie-break:** If test Macro-F1 gap `< 0.01`, higher **Villain recall** wins.
    3. **Useful?:** Only if test Macro-F1 is strictly higher than the Baseline's.
    4. **Overfitting?:** `Train Macro-F1 − Test Macro-F1 > 0.10` → it overfits.
  - Answer in report:
    - **`Q1`:** Which model wins by the rules? Quote the numbers.
    - **`Q2`:** Did both models beat the Baseline? By how much?
    - **`Q3`:** Which class is harder to predict? Use the confusion matrices.
    - **`Q4`:** Does either model overfit? Use the train–test gap.

---

## Phase 4: Task 2 — Unsupervised Clustering (K-Means & Agglomerative Ward)
*25 Marks · Section 4 · Slides 20–28*

- [ ] **Load Clean Set A (No Labels) & Fit Imputer + Scaler**
  - Load **Clean Set A** (cleaned `train.csv`, all characters, **6 scores only**).
  - Never pass `alignment`, `name`, or any other column into clustering; **never use `test.csv`** in Task 2.
  - Fit `SimpleImputer(strategy="median") → StandardScaler()` on Clean Set A.

- [ ] **Run K-Means (`k = 2..8`), Choose Final `k` & Plot `G5`, `G6`**
  - Run `KMeans(n_clusters=k, n_init=10, random_state=42)` for every `k ∈ {2, 3, 4, 5, 6, 7, 8}` on the scaled data, recording inertia and `silhouette_score`.
  - Apply the **exact `k`-selection rule** (Slide 24):
    - `Final k` = the `k` in `2–8` with the **highest Silhouette Score**.
    - Tie (identical to **3 decimal places**) → choose the **smaller `k`**.
    - Describe what the Elbow curve shows, **without letting it change the choice of `k`**.
    - Do **not** copy the illustrative example numbers from Slide 24.
  - Plot **Graph `G5`** (Elbow curve: inertia vs `k` for 2–8) and **Graph `G6`** (Silhouette score vs `k`, with **final `k` clearly marked**).

- [ ] **Run Agglomerative Clustering (Ward) & Plot Dendrogram `G7`**
  - Run `AgglomerativeClustering(n_clusters=k, linkage="ward")` using the **same `final k`** chosen from K-Means.
  - Compute hierarchical linkage via SciPy: `linkage(X_scaled, method="ward")` and plot **Graph `G7`**: `dendrogram(..., truncate_mode="lastp", p=30)` (last 30 merges) with a **dashed horizontal cut line** where the tree splits into `k` groups.

- [ ] **Compute All 4 Unsupervised Metrics on Scaled Data & Produce Table `T6`**
  - Compute all metrics on the **scaled data** (never use accuracy, precision, or recall — there are no true labels):
    - `silhouette_score` (−1 to 1, **↑ higher is better**; bands: `0.71–1.00` strong · `0.51–0.70` reasonable · `0.26–0.50` weak · `≤ 0.25` no real structure).
    - `davies_bouldin_score` (0 and up, **↓ lower is better**).
    - `calinski_harabasz_score` (0 and up, **↑ higher is better**; only compare models on the same data).
    - `adjusted_rand_score(kmeans_labels, agglo_labels)` (about 0 to 1, agreement; bands: `≥ 0.70` high · `0.40–0.69` moderate · `< 0.40` low agreement).
  - Complete **Table `T6` (Clustering metrics + ARI)**: `Algorithm | k | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Wins (of 3)` for `K-Means` and `Agglomerative`, plus an `ARI between the two` row.

- [ ] **Renumber Clusters by Mean Score, Fit Shared 2D PCA & Produce `G8–G11`**
  - For each algorithm, compute the mean of the 6 scores on the **original 0–100 scale** per cluster and **renumber in ascending order**: **lowest average = `Cluster 1`**, next = `Cluster 2`, etc.
  - Apply `tab10` colours consistently (`Cluster 1` = 1st colour, `Cluster 2` = 2nd colour, …) in **every** Task 2 graph.
  - Fit `PCA(n_components=2, random_state=42)` **once** on the scaled data and reuse that exact projection and axes for both:
    - **Graph `G8`:** PCA scatter — K-Means (axes: `PC1`, `PC2` with `% variance explained`).
    - **Graph `G9`:** PCA scatter — Agglomerative (exact same projection and axes as `G8`).
  - Plot **Graph `G10`** (Cluster profiles — K-Means: `x` = 6 scores, one bar per cluster, `y` = average `0–100`) and **Graph `G11`** (Cluster profiles — Agglomerative, same format as `G10`).

- [ ] **Build Named Profiles (`T7`), Reveal Alignment Mix (`T8`) & Answer `Q5–Q8`**
  - Complete **Table `T7` (Cluster profiles — one table per algorithm: K-Means & Agglomerative)**: `Cluster | Size | INT | STR | SPD | DUR | POW | COM | Name you give` (averages on the original `0–100` scale + short descriptive name per cluster).
  - **Only in Step 8 (after clusters are finalized)** inspect the held-aside `alignment` column to produce **Table `T8`**: `% Good`, `% Bad`, `% Neutral`, and `% blank` in each cluster.
  - Apply Task 2 comparison criteria (better separated = algorithm winning **at least 2 of 3** internal metrics; similarity = ARI band backed by profiles; easier to interpret = algorithm whose profiles `G10`/`G11` differ more clearly) and answer:
    - **`Q5`:** Which `k` did you choose? What does the Elbow say?
    - **`Q6`:** Did the two algorithms find similar groups? Use ARI and profiles.
    - **`Q7`:** Which grouping is better separated, and which is easier to interpret?
    - **`Q8`:** Describe each cluster in plain words. Is any cluster mostly villains?

---

## Phase 5: Task 3 — Neural Networks (MLPs) & Final 5-Model Ranking
*25 Marks · Section 5 · Slides 29–36*

- [ ] **Create Stratified 80/20 Validation Split & Fit Preprocessors on 80% Only**
  - From Clean Set B `train.csv`, create the fit/validation split using the exact call:
    ```python
    train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ```
  - Fit `SimpleImputer(strategy="median")` and `StandardScaler()` on the **80% fit split ONLY**; apply (`.transform()`) to the 20% validation split and `test.csv`.
  - Keep note for final comparison: Decision Tree & KNN trained on **100%** of `train.csv`, whereas the neural networks train on **80%** of `train.csv` — **this is expected and must be mentioned when comparing all 5 models** (Slide 31).

- [ ] **Build Exact Network A & Network B Architectures & Draw `G12`**
  - Set `tf.keras.utils.set_random_seed(42)` in TensorFlow / Keras.
  - **Network A (`#6A1B9A`, Basic — 1 hidden layer):** `Input (6 scores) → Dense(8, activation="relu") → Dense(1, activation="sigmoid")` → confirm **65 trainable parameters**.
  - **Network B (`#00838F`, Deeper — 2 hidden layers):** `Input (6 scores) → Dense(16, activation="relu") → Dense(8, activation="relu") → Dense(1, activation="sigmoid")` → confirm **257 trainable parameters**.
  - Output = probability of Villain: `≥ 0.5 → Villain (1)`, `< 0.5 → Hero (0)`.
  - Create **Graph `G12`**: Architecture diagrams for Network A and Network B (any drawing tool — **the ONLY graph not made by code**).

- [ ] **Train Both Networks with Locked Settings & Produce `T9`, `G13`, `G14`**
  - Train **both** networks on the exact same split with identical locked settings (only hidden layers differ):
    - Framework: `TensorFlow / Keras` · Seed: `tf.keras.utils.set_random_seed(42)`
    - Hidden activation: `ReLU` · Output layer: `1 neuron, sigmoid`
    - Loss: `binary_crossentropy` · Optimizer: `Adam(learning_rate=0.001)`
    - Batch size: `32` · Max epochs: `200` · Decision threshold: `0.5`
    - Early stopping: `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)`
  - **Strictly NOT Allowed:** Dropout, extra layers/neurons, class weights, resampling, changing any setting above, or tuning on `test.csv`. *(Small run-to-run differences are normal and accepted.)*
  - Record all network settings and **best epoch** (lowest validation loss) for Network A & Network B in **Table `T9`**.
  - Plot **Graph `G13`** (Loss curves — Network A) and **Graph `G14`** (Loss curves — Network B): `x` = epoch, `y` = binary cross-entropy loss, two lines (training loss & validation loss), plus a **dashed vertical line at the best epoch**.

- [ ] **Evaluate Train, Validation & Test Separately & Produce `T10` & `G15`**
  - Open `test.csv` **once** at the end for final evaluation; keep train, validation, and test metrics strictly separate.
  - Compute metrics using `classification_report` and `log_loss` from `sklearn.metrics`:
    - `Accuracy` and `Macro-F1 (★ MAIN)`: report on **train · validation · test**.
    - `Log loss` (lower is better): report on **validation · test** (`—` for train).
    - `Precision, Recall, F1 per class` and `Confusion matrix`: report on **test**.
  - Populate **Table `T10` (Results by split)**: `Network | Split | Accuracy | Macro-F1 | Log loss` for Network A (`train`, `validation`, `test`) and Network B (`train`, `validation`, `test`), **PLUS** per-class Precision, Recall, and F1 on `test.csv` inside `T10` or right below it.
  - Plot **Graph `G15`**: Confusion matrices for Network A and Network B on `test.csv` (required format: rows = actual, cols = predicted, Good then Bad, counts shown in every cell).

- [ ] **Compare A vs. B, Produce Final 5-Model Ranking (`T11`, `G16`) & Answer `Q9–Q12`**
  - **Network A vs. Network B rules (Slide 35):**
    - Winner = higher test Macro-F1.
    - Say *"the deeper network improved results"* **ONLY IF** Network B's test Macro-F1 is **at least `0.01` higher** than Network A's.
    - Overfitting: `train Macro-F1 − test Macro-F1 > 0.10`, **OR** validation loss rises while training loss falls.
  - **All 5 models on `test.csv` (Baseline, Decision Tree, KNN, Network A, Network B):**
    - Rank 1–5 by **test Macro-F1**; if gap `< 0.01`, higher **Villain recall** ranks higher.
    - `"Beats baseline?"` = test Macro-F1 strictly higher than Baseline's (`yes`/`no`).
  - Populate **Table `T11` (Final ranking)**: `Rank (1–5) | Model | Test accuracy | Test Macro-F1 | Villain recall | Beats baseline? (yes/no)`.
  - Plot **Graph `G16` (Final comparison)**: Grouped bars showing **Accuracy and Macro-F1** for **all 5 models** on `test.csv` using fixed model hex colours.
  - Answer in report:
    - **`Q9`:** Did Network B beat A by at least `0.01` Macro-F1? Why or why not?
    - **`Q10`:** Do the loss curves show over- or underfitting? Which epoch was best?
    - **`Q11`:** Did either network beat the Decision Tree and KNN? Give numbers *(and note the 80% vs. 100% training-data difference)*.
    - **`Q12`:** Name two limitations of this dataset or experiment.

---

## Phase 6: Final Integrity Audit & Pre-Submission Verification
*Cross-Section Audit · All 37 Slides Verified*

- [ ] **Clean Top-to-Bottom Notebook Execution & Leakage Check**
  - Restart kernel and run `my_project/superhero_project.ipynb` top-to-bottom reading `data/train.csv` and `data/test.csv`.
  - Confirm zero `.fit()` / `.fit_transform()` calls on `test.csv` (or validation split), zero use of forbidden columns (`overall_score`, text, `has_*`, etc.), `random_state=42` everywhere, `tf.keras.utils.set_random_seed(42)`, and exact parameter counts (`65` for Network A, `257` for Network B).

- [ ] **100% Deliverables & Formatting Sign-Off (`G1–G16`, `T1–T11`, `Q1–Q12`)**
  - Verify all **16 graphs (`G1–G16`)**, all **11 tables (`T1–T11`)** (including both algorithm tables for `T7` and per-class test metrics for `T10`), and all **12 report answers (`Q1–Q12`)** exist with exact numbering.
  - Confirm every single graph and table has a title, labelled axes, legend (if > 1 colour), mandated hex/`tab10` colours, and **1–2 sentences underneath** stating what it shows and what you conclude.

- [ ] **On-Time Group Submission Check**
  - Confirm group has 2–3 members and submit before **06 December 2026 at 11:59 PM** to incur zero late penalty.

---

## Master Reference: Complete Deliverables Index (`G1–G16`, `T1–T11`, `Q1–Q12`)

### Graphs (`G1–G16`)

| ID | Task | Graph Title & Exact Requirements | Slide |
|---|---|---|---|
| **G1** | Task 1 | **Class counts** — Good (`#1F5FBF`) & Bad (`#D62839`) bars for `train.csv` and `test.csv` side by side, with `%` labels | 19 |
| **G2** | Task 1 | **Confusion matrix — Decision Tree** — rows = actual, cols = predicted, Good then Bad, counts in every cell | 11, 19 |
| **G3** | Task 1 | **Confusion matrix — KNN** — same format as `G2` | 11, 19 |
| **G4** | Task 1 | **Model comparison** — grouped bars: Accuracy, Macro-Precision, Macro-Recall, Macro-F1 for the 3 models (`y` from `0` to `1`) | 19 |
| **G5** | Task 2 | **Elbow curve** — inertia vs `k` (`2–8`) | 24, 28 |
| **G6** | Task 2 | **Silhouette score vs `k`** (`2–8`), final `k` marked | 24, 28 |
| **G7** | Task 2 | **Dendrogram (Ward)** — `truncate_mode="lastp", p=30` (last 30 merges), dashed cut line at `k` | 25, 28 |
| **G8** | Task 2 | **PCA scatter — K-Means** — `PCA(n_components=2, random_state=42)`, axes: `PC1`, `PC2` with `% variance explained`, `tab10` colours | 25, 28 |
| **G9** | Task 2 | **PCA scatter — Agglomerative** — exact same PCA projection and axes as `G8`, `tab10` colours | 25, 28 |
| **G10** | Task 2 | **Cluster profiles — K-Means** — `x` = 6 scores, one bar per cluster (`tab10`), `y` = average `0–100` | 28 |
| **G11** | Task 2 | **Cluster profiles — Agglomerative** — same format as `G10` | 28 |
| **G12** | Task 3 | **Architecture diagrams — Network A & Network B** — any drawing tool (**only graph not made by code**); show 65 & 257 params | 30, 36 |
| **G13** | Task 3 | **Loss curves — Network A** — `x` = epoch, `y` = binary cross-entropy loss, train & validation lines, dashed vertical line at best epoch | 33, 36 |
| **G14** | Task 3 | **Loss curves — Network B** — same format as `G13` | 33, 36 |
| **G15** | Task 3 | **Confusion matrices — Network A & Network B** — required format (Good then Bad), counts shown in every cell | 11, 36 |
| **G16** | Task 3 | **Final comparison** — grouped bars: Accuracy and Macro-F1 for all 5 models on `test.csv`, fixed model hex colours | 36 |

### Tables (`T1–T11`)

| ID | Task | Table Title & Exact Required Columns / Contents | Slide |
|---|---|---|---|
| **T1** | Data Prep | **Dataset inspection** — shape, column names, data types, missing counts for `train.csv` & `test.csv`, PLUS count of `0`s and blanks turned into `NaN` in each of the 6 score columns | 7, 9 |
| **T2** | Data Prep | **Rows after each step** — `Step \| train.csv \| test.csv` for: `Loaded`, `After step 2 (duplicates)`, `After step 4 (weak rows)`, `Clean Set A` (`not used` for test), `Clean Set B` (+ report count of removed Neutral/blank rows) | 7, 9 |
| **T3** | Task 1 | **Class counts and `%`** — Good (Hero) and Bad (Villain) counts & `%` in `train.csv` and `test.csv` | 19 |
| **T4** | Task 1 | **Best settings and best CV Macro-F1** — winning GridSearchCV hyperparameters & 5-fold CV Macro-F1 for Decision Tree and KNN | 19 |
| **T5** | Task 1 | **Task 1 results** — `Model \| Train Macro-F1 \| Accuracy \| Precision Hero \| Recall Hero \| F1 Hero \| Precision Villain \| Recall Villain \| F1 Villain \| Macro-F1` (all on `test.csv` except Train column) for Baseline, DT, KNN | 18, 19 |
| **T6** | Task 2 | **Clustering metrics + ARI** — `Algorithm \| k \| Silhouette ↑ \| Davies–Bouldin ↓ \| Calinski–Harabasz ↑ \| Wins (of 3)` for K-Means & Agglomerative + `ARI between the two` row | 27, 28 |
| **T7** | Task 2 | **Cluster profiles with names (one per algorithm)** — `Cluster \| Size \| INT \| STR \| SPD \| DUR \| POW \| COM \| Name you give` (averages on original `0–100` scale, clusters renumbered lowest avg = Cluster 1) | 25, 27, 28 |
| **T8** | Task 2 | **Alignment mix in each cluster (`%`)** — `% Good`, `% Bad`, `% Neutral`, and `% blank` per cluster (computed only in Step 8) | 27, 28 |
| **T9** | Task 3 | **Network settings + best epoch** — locked architecture/training settings and best epoch (lowest `val_loss`) for Network A & Network B | 32, 33, 36 |
| **T10** | Task 3 | **Train / validation / test results** — `Network \| Split \| Accuracy \| Macro-F1 \| Log loss` (`—` for train log loss) for A & B across all 3 splits, PLUS per-class Precision, Recall & F1 on `test.csv` in `T10` or right below it | 34, 36 |
| **T11** | Task 3 | **Final ranking of all 5 models** — `Rank (1–5) \| Model \| Test accuracy \| Test Macro-F1 \| Villain recall \| Beats baseline? (yes/no)` | 35, 36 |

### Report Questions (`Q1–Q12`)

| ID | Task | Exact Question to Answer in the Report | Slide |
|---|---|---|---|
| **Q1** | Task 1 | Which model wins by the rules? Quote the numbers. | 19 |
| **Q2** | Task 1 | Did both models beat the Baseline? By how much? | 19 |
| **Q3** | Task 1 | Which class is harder to predict? Use the confusion matrices. | 19 |
| **Q4** | Task 1 | Does either model overfit? Use the train–test gap. | 19 |
| **Q5** | Task 2 | Which `k` did you choose? What does the Elbow say? | 28 |
| **Q6** | Task 2 | Did the two algorithms find similar groups? Use ARI and profiles. | 28 |
| **Q7** | Task 2 | Which grouping is better separated, and which is easier to interpret? | 28 |
| **Q8** | Task 2 | Describe each cluster in plain words. Is any cluster mostly villains? | 28 |
| **Q9** | Task 3 | Did Network B beat A by at least `0.01` Macro-F1? Why or why not? | 36 |
| **Q10** | Task 3 | Do the loss curves show over- or underfitting? Which epoch was best? | 36 |
| **Q11** | Task 3 | Did either network beat the Decision Tree and KNN? Give numbers. | 36 |
| **Q12** | Task 3 | Name two limitations of this dataset or experiment. | 36 |

---

## Fixed Colour Palette, Decision Thresholds & Tool Reference

| Category | Exact Specification |
|---|---|
| **Fixed Hex Colours** | Hero / Good `#1F5FBF` · Villain / Bad `#D62839` · Baseline `#9E9E9E` · Decision Tree `#2E7D32` · KNN `#F9A825` · Network A `#6A1B9A` · Network B `#00838F` · Clusters: matplotlib `tab10` order (`Cluster 1` = 1st colour) |
| **Seeds & Tools** | `random_state=42` everywhere; `tf.keras.utils.set_random_seed(42)`; `pandas`, `scikit-learn`, `SciPy`, `matplotlib` (`seaborn` allowed), `TensorFlow/Keras` |
| **Task 1 Decision Rules** | Winner = highest test Macro-F1; gap `< 0.01` → higher Villain recall wins; Useful only if test Macro-F1 > Baseline; Overfits if `Train Macro-F1 − Test Macro-F1 > 0.10` |
| **Task 2 Bands & Rules** | `k` = highest Silhouette in `2–8` (tie to 3 decimals → smaller `k`); Silhouette bands: `0.71–1.00` strong, `0.51–0.70` reasonable, `0.26–0.50` weak, `≤ 0.25` no real structure; ARI bands: `≥ 0.70` high, `0.40–0.69` moderate, `< 0.40` low; Better separated = wins `≥ 2 of 3` (Silhouette ↑, DB ↓, CH ↑) |
| **Task 3 Decision Rules** | Threshold `0.5` (`≥ 0.5 → Villain`); Claim *"the deeper network improved results"* ONLY if Network B test Macro-F1 ≥ Network A + `0.01`; Overfits if `train Macro-F1 − test Macro-F1 > 0.10` OR val loss rises while train loss falls; All 5 models ranked by test Macro-F1 (gap `< 0.01` → higher Villain recall) |

---

## Marks Breakdown & Section Mapping Summary

| Section | Phase | Marks | Core Evidence Assessed |
|---|---|---:|---|
| **Section 1: Instructions** | Phase 1, 6 | — | Group of 2–3, deadline `06 Dec 2026 11:59 PM`, late penalty compliance |
| **Section 2: Data Preparation** | Phase 1, 2 | **10** | Exact folder & 6 inputs, Steps 1–5 in order, Clean Set A & B, `T1`, `T2`, zero leakage, global rules |
| **Section 3 · Task 1: Classification** | Phase 3 | **25** | Baseline + DT (15 combos) + KNN (12 combos) via 5-fold stratified CV, single test eval, `G1–G4`, `T3–T5`, `Q1–Q4` |
| **Section 4 · Task 2: Clustering** | Phase 4 | **25** | Clean Set A (no labels, no `test.csv`), K-Means `k=2..8` + Ward Agglomerative, scaled metrics + ARI, renumbered clusters, shared PCA, `G5–G11`, `T6–T8`, `Q5–Q8` |
| **Section 5 · Task 3: Neural Network** | Phase 5 | **25** | 80/20 stratified split, Network A (65 params) & Network B (257 params), locked training settings, loss curves, train/val/test split results, all 5-model ranking, `G12–G16`, `T9–T11`, `Q9–Q12` |
| **Total (Listed in Brief)** | **Phases 1–6** | **85** | **16 Graphs (`G1–G16`), 11 Tables (`T1–T11`), 12 Report Answers (`Q1–Q12`)** |

---

## Critical Rules, Prohibitions & Penalties Checklist — Do Not Submit Before Checking

- **Late Submission Deductions (Up to 0 Marks):** Due **06 December 2026 at 11:59 PM** (groups of 2–3). Up to 5 min late = **−10%**; > 5 to 60 min late = **−20%**; > 1 to 12 hr late = **−50%**; > 12 hr late = **0 marks**.
- **Dataset & Split Prohibitions:** **Never** merge `train.csv` and `test.csv`; **never** create your own train/test split; **never** filter by publisher (keep Marvel, DC, and all others); **never** use `test.csv` in Task 2.
- **Forbidden Columns & Data Leakage Traps:** Use **only** the 6 numeric ability scores as model inputs. **Never** use `overall_score`, `history_text`, `powers_text`, `superpowers`, any `has_…` column, `creator`, `gender`, `height`, `weight`, `colours`, or `name` as an input. **Never** call `.fit()` / `.fit_transform()` on `test.csv` (or validation data) and **never** tune hyperparameters on `test.csv`.
- **Banned Modelling Tricks:**
  - *Task 1:* **No** class weights, oversampling, undersampling, extra features, or tuning on `test.csv`.
  - *Task 2:* **Never** pass `alignment` into clustering (look at it only in Step 8); **never** use accuracy, precision, or recall to evaluate clustering; **never** let the Elbow override Silhouette; **never** copy slide example numbers.
  - *Task 3:* **No** dropout, extra layers/neurons, class weights, resampling, changing any locked hyperparameter, or tuning on `test.csv`. Do **not** claim Network B improved results unless its test Macro-F1 beats Network A by `≥ 0.01`.
- **Mandatory Visual & Reporting Compliance:** Every graph/table must match exact numbering (`G1–G16`, `T1–T11`), exact fixed hex colours / `tab10` cluster order, required confusion matrix layout (rows = actual, cols = predicted, Good then Bad, counts shown), and include **1–2 sentences underneath** stating what it shows and what you conclude. Remember to note that DT & KNN trained on 100% of `train.csv` while Networks A & B trained on 80%.
