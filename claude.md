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

## Pipeline (`grader/pipeline.py`)

For each candidate:

1. **Download CSV** from `csv_url`. Google Drive sharing links are auto-converted to
   direct download URLs.
2. **Hash content** (MD5) — used as the cache key alongside `candidate_name`.
3. **Cache check** — skip if `(candidate_name, content_hash)` already in SQLite.
4. **Map columns** — heuristic matching by column name:
   - `member_id`: `member_id`, `memberid`, `member`, `id`, `user_id`, `userid`, `user`
   - `score`: `score`, `churn_score`, `churn_prob`, `churn_probability`, `probability`,
     `prob`, `risk`, `risk_score`, `pred`, `prediction`
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

### precision_at_n (primary metric)

```
precision@N = |top_N_by_score ∩ true_churners| / N
```

- Computed for every N from 1 to len(predictions) and stored as a list in SQLite.
- Dashboard slices the stored curve at any N via slider — no recomputation.
- `true_churners`: member_ids where `churn == 1` in `test_churn_labels.csv`.

### TODO: Uplift-Aware Metric

The `outreach` column in `test_churn_labels.csv` records who was actually outreached.
Add `qini_at_n()` in `metrics.py` and register it in `Scorer` (accepts `metric: str`).

---

## Dashboard (`dashboard/app.py`)

**Sidebar**: N slider (1–10,000), baseline toggle, valid-only filter, summary metrics.

**Section 1 — Leaderboard**
- Columns: status icon | Candidate | Precision@N (slider) | Precision@Rec.N | Rec. N | Status
- Sorted by Precision@N descending. Updates live on slider move.

**Section 2 — Precision@N Chart**
- One line per candidate. X = N, Y = precision@N.
- Dotted vertical line at each candidate's recommended N.
- Horizontal dashed line for random baseline.
- Current slider N shown as a solid black vertical line.

**Auto-refresh**: polls SQLite every 60 seconds.

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

**Schema migration**: on startup, if the DB has the old `commit_sha` column (from the
prior agent-based design), the table is automatically dropped and recreated.

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| CSV URL unreachable | `status=CSV_DOWNLOAD_ERROR`, not cached (will retry next run) |
| Cannot parse CSV | `status=SCHEMA_ERROR`, cached |
| Column names unrecognized | `status=SCHEMA_ERROR`, cached; add hints to `_MEMBER_ID_HINTS` / `_SCORE_HINTS` in `pipeline.py` |
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
