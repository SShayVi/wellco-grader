# CLAUDE.md — wellco-grader

## Project Purpose

Automated evaluation system for the WellCo Data Science home assignment.
Reads candidates from a Google Sheet (name + CSV link + recommended N), scores their
predictions against true churn labels, and serves a live leaderboard dashboard.

---

## High-Level Flow

```
Google Sheet (candidate_name, csv_url, recommended_n)
        ↓
  Download CSV  ←── direct URL (Google Drive sharing links normalized automatically)
        ↓
  Heuristic column mapping → (member_id, score)
        ↓
  SQLite Cache (keyed by candidate_name + MD5 of CSV content)
        ↓
  Scorer  ←── true labels (test_churn_labels.csv)
        ↓
  Streamlit Dashboard (leaderboard + precision@N chart)
```

**Pipeline trigger**: Manual CLI (`python -m grader`). Results are written to SQLite.
The dashboard auto-refreshes from the cache, decoupling processing from display.

---

## Folder Structure

```
wellco-grader/
├── grader/
│   ├── sources/
│   │   └── google_sheets.py    # Read sheet via public CSV export URL
│   ├── scoring/
│   │   ├── metrics.py          # precision_at_n() + TODO: uplift-aware metrics
│   │   └── scorer.py           # Joins predictions + true labels → precision curve
│   ├── storage/
│   │   ├── cache.py            # SQLite read/write keyed by (candidate, content_hash)
│   │   └── models.py           # Pydantic: CandidateResult, PredictionStatus
│   └── pipeline.py             # Orchestrator: sheet → download → map → score → cache
├── dashboard/
│   └── app.py                  # Streamlit app
├── config/
│   └── settings.py             # Pydantic Settings from env vars
├── tests/
│   ├── unit/
│   │   ├── test_scoring.py
│   │   └── test_cache.py
│   └── fixtures/
│       └── csvs/               # Edge-case prediction CSVs for unit tests
├── data/
│   └── test_churn_labels.csv   # True labels — never committed to public repo
├── future/                     # Archived agent-based system (see future/README.md)
├── .env.example
├── requirements.txt
└── CLAUDE.md
```

---

## Validation (`grader/validation.py`)

Standalone module that validates and standardises any prediction CSV.
Called internally by the pipeline; also available as a CLI command.

```bash
# Validate a remote CSV (with overlap check against true labels)
python -m grader validate https://github.com/.../predictions.csv

# Validate a local file
python -m grader validate /path/to/predictions.csv
```

**Function signature:**
```python
from grader.validation import validate_and_standardize

result = validate_and_standardize(
    raw,                      # bytes — raw CSV content
    true_member_ids=None,     # set[int] — test set member IDs; skips overlap checks if None
    min_overlap=0.5,          # float — minimum required ID overlap fraction
)
result.ok           # bool — False if any ERROR-level issue found
result.standardized # pd.DataFrame with [member_id, score] sorted descending; None on error
result.issues       # list[Issue] — full audit trail with severity + message
result.summary()    # formatted multi-line string for printing
```

**Issue codes:**

| Code | Severity | Meaning |
|---|---|---|
| `PARSE_ERROR` | ERROR | Cannot read CSV file |
| `EMPTY_CSV` | ERROR | File has no rows |
| `COLUMNS_NOT_FOUND` | ERROR | Cannot identify member_id or score column |
| `NO_VALID_ROWS` | ERROR | No numeric rows after normalization |
| `WRONG_DATASET` | ERROR | IDs don't match test set; specific message if training IDs detected (range 0–20,000) |
| `LOW_OVERLAP` | ERROR | < min_overlap of IDs found in test set |
| `COLUMN_REMAPPED` | INFO | Non-standard but recognized column names |
| `RANK_INVERTED` | INFO | Rank column detected and negated |
| `PARTIAL_SUBMISSION` | INFO | Fewer than 10,000 members submitted |
| `ROWS_DROPPED` | WARNING | Non-numeric rows removed |
| `DUPLICATE_IDS` | WARNING | Duplicate member_ids deduplicated |
| `DEGENERATE_SCORES` | WARNING | All scores identical |
| `LOW_SCORE_VARIETY` | WARNING | Fewer than 10 unique score values |
| `LOW_ROW_COUNT` | WARNING | Fewer than 10 rows submitted |

---

## Pipeline (`grader/pipeline.py`)

For each candidate:

1. **Download CSV** from `csv_url`. Google Drive sharing links are auto-converted to
   direct download URLs.
2. **Hash content** (MD5) — used as the cache key alongside `candidate_name`.
3. **Cache check** — skip if `(candidate_name, content_hash)` already in SQLite.
4. **Map columns** — three-stage heuristic (all case-insensitive):
   1. **Exact match** against known hint lists (see below)
   2. **Substring match** — hint contained in column name or vice-versa
   3. **Dtype fallback** — if exactly 2 numeric columns and neither matched, infer
      by value range (0-1 floats → score, large integers → member_id)

   **member_id hints** (in priority order):
   `member_id`, `memberid`, `member`, `id`, `user_id`, `userid`, `user`,
   `customer_id`, `client_id`, `patient_id`, `account_id`

   **score hints** (in priority order — higher = preferred when multiple match):
   `weighted_uplift`, `uplift`, `cate`, `cate_estimate`, `cate_score`,
   `benefit_score`,
   `propensity_score`, `propensity`, `priority_score`, `prioritization_score`,
   `churn_score`, `churn_prob`, `churn_probability`,
   `churn_prob_no_outreach`, `p_churn_no_outreach`, `baseline_churn_proba`,
   `score`, `probability`, `prob`, `risk`, `risk_score`, `pred`, `prediction`,
   `rank` *(inverted — rank 1 treated as highest score)*

   **Rank inversion**: columns named `rank`, `ranking`, or `position` are negated
   before sorting so that rank 1 ends up first (highest priority).

   **Partial submissions**: CSVs containing only the candidate's top-N recommended
   members are fully supported. The precision curve is computed for N=1..submitted_count;
   `precision_at_n(N)` returns `None` for N > submitted_count (shown as "—" in the dashboard).

5. **Validate** — member_id overlap with true labels must be ≥ 50%.
6. **Score** — compute full precision@N curve (N=1..len(predictions)).
7. **Cache** result.

**Status flags** (`PredictionStatus`):
- `OK` — passes all checks
- `CSV_DOWNLOAD_ERROR` — URL unreachable or request failed
- `SCHEMA_ERROR` — cannot parse CSV or map columns
- `INVALID_PREDICTIONS` — member_id overlap < 50%
- `DEGENERATE_PREDICTIONS` — all scores identical (no discrimination)

---

## Scoring

All four metrics are computed as full curves (one value per N from 1 to len(predictions))
and stored in SQLite. The dashboard slices any curve at the slider's N — no recomputation.

### precision@N (primary metric)
```
precision@N = |top_N ∩ churners| / N
```

### gain@N (cumulative recall)
```
gain@N = |top_N ∩ churners| / total_churners
```
Fraction of all churners captured. Random baseline: diagonal line `N / total_population`.

### lift@N (relative to random)
```
lift@N = precision@N / churn_rate
```
How many times better than a random ranker. Random baseline: 1.0 (constant).

### qini@N (uplift-aware)
```
qini@N = (treated_churners_in_topN / N_T) - (control_churners_in_topN / N_C)
```
- `treated` = `outreach == 1`; `control` = `outreach == 0`; `N_T`, `N_C` = total treatment/control sizes.
- Positive ⟹ model surfaces treated churners disproportionately vs. control churners.
- Random baseline: 0.0 (constant).
- Requires `outreach` column in `test_churn_labels.csv`.

### Implementation

- `grader/scoring/metrics.py` — `precision_curve`, `gain_curve`, `lift_curve`, `qini_curve` (all O(N))
- `grader/scoring/scorer.py` — `Scorer.score_all(df)` returns `{precision, gain, lift, qini}` dicts;
  `Scorer.fill_curves(result)` backfills missing curves locally; `_qini_data` pre-computed at init
- `grader/storage/models.py` — `CandidateResult` stores all four curves;
  `precision_at_n`, `gain_at_n`, `lift_at_n`, `qini_at_n` helper methods
- `grader/storage/cache.py` — `ResultCache.clear_all()` deletes all rows (used by Re-grade All)
- `grader/pipeline.py` — `run_pipeline(settings, scorer=None)` accepts a pre-built scorer so the
  dashboard on Streamlit Cloud can pass its already-loaded scorer instead of re-reading the labels file

**Dashboard backfill** (for cached results without the new curves): done inline in `app.py` using
`getattr(scorer, '_churn_rate', ...)` etc. — avoids calling new scorer methods that may not exist
on older Python bytecache versions on Streamlit Cloud.

---

## Dashboard (`dashboard/app.py`)

**Sidebar**: N slider (1–10,000), metric selector (Precision / Gain / Lift / Qini), baseline
toggle, valid-only filter, summary metrics.

**Section 1 — Leaderboard**
- Columns: status icon | Candidate | Precision@N | Gain@N | Lift@N | Qini@N | Rec. N | Status
- All four metrics shown simultaneously at the current slider N.
- Sorted by the metric selected in the sidebar. Metric formats: Precision (3 dp) · Gain (%) · Lift (×) · Qini (4 dp).
- All field access uses `getattr(r, 'field', None)` to tolerate old cached model versions.

**Section 2 — Metric Chart**
- Title and Y-axis update to match the selected metric.
- One line per candidate. X = N, Y = selected metric value.
- Dotted vertical line at each candidate's recommended N.
- Random baseline: precision → horizontal at churn_rate; gain → diagonal N/total_pop;
  lift → horizontal at 1.0; qini → horizontal at 0.
- Current slider N shown as a solid black vertical line.
- Old cached results (missing gain/lift/qini) are filled in automatically via inline backfill in `app.py`.

**Section 3 — Candidate Overlap**
- Multiselect to choose which candidates to compare.
- **Pairwise heatmap**: overlap % at current slider N for every pair.
- **Exclusivity bar**: fraction of each candidate's top-N that is unique (not in any other selected candidate's top-N).
- **Overlap-over-N line chart**: pairwise overlap % (Y) vs N (X), one colored line per pair. O(N) incremental algorithm; results cached in session state.
- **Score distribution histogram**: one semi-transparent histogram per candidate, using `probability density` normalization and 60 bins. Controlled by the same candidate multiselect. Optional "Standardise scores (0–1 scale)" checkbox applies min-max normalization before plotting so candidates with different score ranges can be compared visually. `ranked_scores` is stored in SQLite alongside `ranked_member_ids`.

**Section 4 — Validate a Submission**
- URL text input + Validate button.
- Runs `validate_and_standardize` and shows per-issue callouts (ERROR = red, WARNING = orange, INFO = blue).
- Shows a preview of the standardised output on success.

**Auto-refresh**: polls SQLite every 60 seconds.

**Sidebar grading buttons** (visible only when scorer is loaded):
- **Run Grader** — fetches Google Sheet, skips candidates already in cache (fast, incremental)
- **Re-grade All** — clears the entire cache first, then re-scores everyone from scratch (use after
  schema changes or to force new metrics onto existing candidates)

**Scorer loading on Streamlit Cloud**: `get_scorer()` tries (1) local `data/test_churn_labels.csv`,
then (2) secret `TRUE_LABELS_CSV_B64` (base64-encoded CSV), then (3) secret `TRUE_LABELS_CSV` (raw text).
Generate the secret with: `base64 -i data/test_churn_labels.csv | tr -d '\n' | pbcopy`
A "Retry loading scorer" button appears when the scorer fails, clearing the `@st.cache_resource`
so the secret is re-read without a full redeploy.

**Deployment**:
- Dev: `streamlit run dashboard/app.py`
- Public: Streamlit Community Cloud

---

## Data Sources

### Google Sheet

Public sheet, accessed via CSV export URL (no service account needed):
```
https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv
```
Sheet must be set to "Anyone with the link → Viewer".

Required columns: `candidate_name`, `csv_url`, `recommended_n`.
Deduplicates by `candidate_name` (last row wins).

### True Labels

`data/test_churn_labels.csv` — columns: `member_id`, `signup_date`, `churn`, `outreach`.
10,000 members, ~20% churn rate. Never committed to a public repo.

---

## Caching

SQLite at `CACHE_DB_PATH` (default `.cache/grader.db`).

**Cache key**: `(candidate_name, content_hash)` where `content_hash` = MD5 of the
downloaded CSV bytes. If a candidate re-submits the same CSV at the same URL, the
cached result is returned instantly. A new CSV (changed content) triggers reprocessing.

**Download errors** use `content_hash = "url:<md5_of_url>"` so they are cached and
appear in the leaderboard. They will be retried (and overwritten) on the next pipeline
run, allowing the candidate to fix their URL and resubmit.

**Schema migration**: on startup, if the DB has the old `commit_sha` column (from the
prior agent-based design), the table is automatically dropped and recreated.

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| CSV URL unreachable | `status=CSV_DOWNLOAD_ERROR`, cached (appears in leaderboard); retried on next run |
| `recommended_n` missing in sheet | Defaults to 1,000; leaderboard Status column shows "Rec. N defaulted to 1,000" |
| Cannot parse CSV | `status=SCHEMA_ERROR`, cached |
| Column names unrecognized | Three-stage heuristic attempted; if all fail → `status=SCHEMA_ERROR`. To add a name: extend `_MEMBER_ID_HINTS` or `_SCORE_HINTS` in `pipeline.py` |
| CSV with only top-N rows | Fully supported; precision curve covers N=1..submitted_count only |
| Rank column (1=best) | Auto-detected (`rank`/`ranking`/`position`) and negated so rank 1 sorts first |
| Wrong member ID dataset | `status=INVALID_PREDICTIONS` — candidate used IDs not in the test set (e.g., training set IDs) |
| member_id overlap < 50% | `status=INVALID_PREDICTIONS`, cached |
| All scores identical | `status=DEGENERATE_PREDICTIONS`, precision curve still computed |
| Google Drive sharing link | Auto-converted to `drive.google.com/uc?export=download&id=...` |
| Duplicate candidate rows in sheet | Deduped by `candidate_name`; last row wins |

---

## CLI

```bash
# Process all candidates from the Google Sheet
python -m grader

# Process a single candidate (no sheet needed)
python -m grader --candidate "Alice" https://example.com/predictions.csv 800

# Validate a CSV without scoring (candidates can use this to check before submitting)
python -m grader validate https://github.com/.../predictions.csv
python -m grader validate /local/path/predictions.csv
```

---

## Environment Variables

```bash
GOOGLE_SHEET_ID=                            # Required (unless using --candidate)
TRUE_LABELS_PATH=data/test_churn_labels.csv
CACHE_DB_PATH=.cache/grader.db
REFRESH_INTERVAL_SECONDS=60
LOG_LEVEL=INFO
MIN_MEMBER_ID_OVERLAP=0.5
```

---

## Testing

```bash
python -m pytest tests/unit/
```

- `test_scoring.py` — pure metric functions + Scorer integration (uses real true labels)
- `test_cache.py` — SQLite round-trips, migration, deduplication
- Fixture CSVs in `tests/fixtures/csvs/` for edge-case inputs

---

## Key Dependencies

```
streamlit          # Dashboard
streamlit-autorefresh
pandas             # Data manipulation
pydantic           # Data models
pydantic-settings  # Env-var config
requests           # CSV download
plotly             # Precision@N chart
pytest             # Tests
```

---

## Future: Agent-Based Grader

The `future/` folder contains the full prior system: GitHub repo traversal, LLM-based
column detection, N extraction, and YAML-driven code review scoring. See `future/README.md`
for reactivation instructions.
