# CLAUDE.md — wellco-grader

## Project Purpose

Automated evaluation system for the WellCo Data Science home assignment.
Reads candidate repos from a Google Sheet, runs AI agents to extract predictions
and review code, scores against true churn labels, and serves a live leaderboard dashboard.

---

## High-Level Flow

```
Google Sheet (candidate_name, repo_url)
        ↓
  GitHubClient  ←── traverses repo, downloads files
        ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  PredictionAgent   NExtractorAgent   CodeReviewerAgent      │
  │  (find + normalize  (find recommended  (YAML-driven          │
  │   predictions CSV)   outreach N)        code review)         │
  └─────────────────────────────────────────────────────────────┘
        ↓
  SQLite Cache (keyed by candidate + commit SHA)
        ↓
  Scorer  ←── true labels (test_churn_labels.csv)
        ↓
  Streamlit Dashboard (leaderboard + precision@N chart + review viewer)
```

**Pipeline trigger**: Manual CLI (`python -m grader.pipeline`). Results are written
to SQLite. The dashboard is always-on and auto-refreshes from the cache, decoupling
expensive LLM work from display.

---

## Folder Structure

```
wellco-grader/
├── grader/
│   ├── agents/
│   │   ├── base.py                  # BaseAgent: retry, rate-limit, logging
│   │   ├── prediction_agent.py      # Find CSV, normalize schema → (member_id, score, rank)
│   │   ├── n_extractor.py           # Find recommended N across entire repo
│   │   └── code_reviewer.py         # Answer YAML review questions with scores + justifications
│   ├── sources/
│   │   ├── google_sheets.py         # Read sheet via public CSV export URL
│   │   └── github.py                # Repo traversal + file download (PyGithub)
│   ├── scoring/
│   │   ├── metrics.py               # precision_at_n() + TODO: uplift-aware metrics
│   │   └── scorer.py                # Joins predictions + true labels → CandidateScore
│   ├── storage/
│   │   ├── cache.py                 # SQLite read/write keyed by (candidate, commit_sha)
│   │   └── models.py                # Pydantic: CandidateResult, ReviewResult, CandidateScore
│   └── pipeline.py                  # Orchestrator: sheet → agents → score → cache
├── dashboard/
│   └── app.py                       # Streamlit app
├── config/
│   ├── review_questions.yaml        # Modifiable review rubric (loaded at runtime)
│   └── settings.py                  # Pydantic Settings from env vars
├── tests/
│   ├── unit/
│   │   ├── test_schema_normalizer.py
│   │   ├── test_n_extractor.py
│   │   ├── test_scoring.py
│   │   └── test_cache.py
│   ├── integration/
│   │   └── test_full_pipeline.py    # Runs against Shay's real repo as fixture
│   └── fixtures/
│       ├── csvs/                    # Edge-case prediction CSVs for unit tests
│       │   ├── standard.csv
│       │   ├── fuzzy_columns.csv    # Non-standard column names
│       │   ├── all_same_score.csv   # Degenerate: no discrimination
│       │   ├── wrong_member_ids.csv # IDs not in true labels
│       │   └── missing_rank.csv     # Has score but no rank column
│       └── repos/                   # Mock repo file trees (dicts) for agent tests
├── data/
│   └── test_churn_labels.csv        # True labels — never committed to public repo
├── .env.example
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Agents

### Design: Three Separate Agents, Shared GitHubClient

Three specialized agents rather than one monolithic agent. Each is independently
testable, cacheable, and retryable. All share a `GitHubClient` that handles repo
traversal, file download, and GitHub rate-limit backoff — agents receive file
contents, not raw API calls.

All agents use **Claude claude-sonnet-4-6** via the Anthropic SDK with structured
output (JSON schema enforced via tool use) for reliable parsing.

---

### PredictionAgent

**Goal**: Return a standardized `DataFrame` with columns `(member_id, score, rank)`.

**Strategy**:
1. Ask `GitHubClient` for all `.csv` files in the repo.
2. For each CSV (largest first, up to 5), send column names + first 3 rows to the LLM.
3. LLM returns `{member_id_col, score_col, rank_col, confidence}` as JSON.
4. Accept the first CSV where confidence >= 0.7 AND member_id overlap with true labels >= 50%.
5. If LLM fails, fall back to heuristic column-name matching (see below).
6. Normalize: rename columns, sort by score descending (ties broken by member_id), assign
   canonical rank 1..N.

**Heuristic fallback column mapping**:
- `member_id`: `id`, `member`, `user_id`, `memberid`, `member_id`
- `score`: `score`, `probability`, `churn_prob`, `risk`, `churn_score`, `pred`
- `rank`: `rank`, `priority`, `position`, `order`

**Validation gates** (sets `PredictionStatus` flag):
- `MISSING_PREDICTIONS`: no CSV found after searching all files
- `SCHEMA_ERROR`: LLM + heuristics both fail to map columns
- `INVALID_PREDICTIONS`: member_id overlap with true labels < 50%
- `DEGENERATE_PREDICTIONS`: all scores identical (no discrimination)
- `OK`: passes all checks

---

### NExtractorAgent

**Goal**: Return the candidate's recommended outreach N as an integer.

**Search priority order** (stops at first confident result):
1. **Predictions CSV**: if it has <= 10,000 rows, N = row count (most common case); also
   look for an explicit column named `n_recommended`, `outreach_n`, etc.
2. **README / docs**: regex search + LLM extraction on matched paragraphs.
3. **Python / notebook files**: grep for assignment-like patterns near outreach/threshold
   logic. Send matches to LLM for extraction.
4. **PDF presentations**: extract text via PyMuPDF, search for N mention.
5. **Fallback**: N = len(predictions CSV), flagged as `N_SOURCE=inferred`.

**Output**: `{n: int, source: str, confidence: float}`
Valid range: [1, 10000]. N > 5000 is flagged with a warning (not an error).

---

### CodeReviewerAgent

**Goal**: For each question in `review_questions.yaml`, return a score and justification.

**Inputs**: all repo text files (`.py`, `.ipynb` notebooks flattened to code, `.md`, `.txt`,
`.yaml`) concatenated and truncated to fit context window (most recent / most central files
prioritized).

**Process**: Single LLM call with all questions + repo contents. LLM returns a JSON array:
```json
[
  {"id": "sabotage_outreach_leakage", "score": 0, "justification": "..."},
  ...
]
```
Score scale per question: `0` = not addressed, `1` = partially, `2` = fully.
Weighted total = `sum(score_i * weight_i) / sum(2 * weight_i)` → normalized 0–1.

**Review questions YAML** is loaded at runtime from `config/review_questions.yaml`.
Malformed YAML causes a loud startup error (not a silent per-candidate failure).

---

## Scoring

### precision_at_n (primary metric)

```
precision@N = |top_N_by_score ∩ true_churners| / N
```

- `top_N_by_score`: the N members with highest `score` (or equivalently lowest `rank`)
  from the candidate's standardized predictions.
- `true_churners`: `member_id`s where `churn == 1` in `test_churn_labels.csv`.
- Ties in score broken by `member_id` (ascending) for determinism.
- Computed for all N from 1 to 10,000 and stored; dashboard slices at any N.

### TODO: Uplift-Aware Metric

The `outreach` column in `test_churn_labels.csv` records who was actually outreached.
Infrastructure to add later:

```python
# scorer.py — add alongside precision_at_n
def qini_at_n(predictions_df, labels_df, n: int) -> float:
    """Qini coefficient: rewards targeting members who respond to outreach."""
    ...
```

The `Scorer` class accepts a `metric: str` parameter. Adding new metrics only
requires adding a new function in `metrics.py` and registering it in `Scorer`.

---

## Dashboard (Streamlit)

### Sections

**1. Leaderboard**
- Table: `candidate_name | precision@N | review_score | recommended_N | N_source | status`
- Interactive N slider (1–10,000). Leaderboard re-ranks live on slider move.
- Rows colored by status (INVALID/DEGENERATE shown in red with tooltip).

**2. Precision@N Chart**
- Line chart: x = N (1–10,000), y = precision@N.
- One colored line per candidate. Hover shows candidate name + value.
- Vertical dashed line at each candidate's recommended N.

**3. Code Review Viewer**
- Candidate selector dropdown.
- Per-question: question text | score (0/1/2) | justification | weight.
- Summary weighted score shown at top.

### Auto-refresh

Dashboard polls SQLite every 60 seconds (`st.rerun()` after `time.sleep(60)`).
When the pipeline adds a new candidate, they appear on next refresh without
any manual action.

### Deployment

- **Dev**: local Streamlit (`streamlit run dashboard/app.py`)
- **Public**: Streamlit Community Cloud (connects to this repo; reads SQLite
  from a mounted path, or switch to a hosted DB like Supabase for production)

---

## Data Sources

### Google Sheet

Public sheet, accessed via CSV export URL (no service account needed):
```
https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv
```
Sheet must be set to "Anyone with the link → Viewer".
Columns: `candidate_name`, `repo_url`.

Initial data:
| candidate_name | repo_url |
|---|---|
| Shay Shavit | https://github.com/SShayVi/wellco-churn-prediction-home-assignment |

Config: `GOOGLE_SHEET_ID` env var.

### True Labels

`data/test_churn_labels.csv` — columns: `member_id`, `signup_date`, `churn`, `outreach`.
10,000 members, 20% churn rate. Never committed to a public repo.

---

## Caching

SQLite at `CACHE_DB_PATH` (default `.cache/grader.db`).

Cache key: `(candidate_name, repo_commit_sha)`.

**Core invariant**: a repo is scanned at most once per commit SHA. On every pipeline
run, the first thing `pipeline.py` does for each candidate is fetch the latest commit
SHA and check the cache. If the SHA is already present, the entire candidate is skipped
— no GitHub file downloads, no LLM calls, no re-scoring. Work only happens for new
candidates or candidates who have pushed new commits since the last run.

Tables:
- `candidate_runs`: one row per (candidate, sha) with pipeline status + timestamps
- `predictions`: standardized member_id/score/rank per candidate run
- `n_extractions`: N value, source, confidence per candidate run
- `review_results`: per-question scores + justifications per candidate run
- `scores`: computed precision@N curve (stored as JSON array) per candidate run

---

## Edge Cases & Sabotage Handling

| Scenario | Behavior |
|---|---|
| No CSV found | `status=MISSING_PREDICTIONS`, shown in dashboard, skipped in scoring |
| CSV has wrong member_ids (< 50% overlap) | `status=INVALID_PREDICTIONS`, flagged in red |
| All scores identical | `status=DEGENERATE_PREDICTIONS`, precision@N shown as baseline |
| N > 5000 or N < 1 | Clamped to [1, 10000], `n_warning=True` shown in dashboard |
| Repo is private or 404 | `status=REPO_UNAVAILABLE`, shown in dashboard |
| LLM schema mapping fails | Heuristic fallback; if still fails → `status=SCHEMA_ERROR` |
| Review YAML malformed | Fatal startup error with clear message |
| GitHub rate limit hit | Exponential backoff up to 5 retries, then `status=GITHUB_ERROR` |
| Anthropic rate limit hit | Exponential backoff up to 3 retries, then propagate |
| Candidate submits wrong test set | Caught by member_id overlap check |
| Duplicate candidate rows in sheet | Deduped by repo_url; latest row wins |

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=                          # Required: Claude API key
GITHUB_TOKEN=                               # Recommended: avoids 60 req/hr anonymous limit
GOOGLE_SHEET_ID=                            # Required: sheet ID from URL
TRUE_LABELS_PATH=data/test_churn_labels.csv
CACHE_DB_PATH=.cache/grader.db
REFRESH_INTERVAL_SECONDS=60
LOG_LEVEL=INFO
```

---

## Testing Philosophy

- **Unit tests**: all scoring/metrics functions (pure, no LLM). Schema normalizer and
  N extractor tested with mocked LLM responses + fixture CSVs.
- **Edge-case fixtures**: CSVs with degenerate, fuzzy, and sabotaged inputs.
- **Integration test**: runs full pipeline against Shay's repo (real GitHub + real LLM).
  Gated behind `--integration` flag to avoid burning API credits in CI.
- **True labels never mocked**: tests use the real `test_churn_labels.csv`.

---

## Key Dependencies

```
anthropic          # Claude API
pygithub           # GitHub repo traversal
streamlit          # Dashboard
pandas             # Data manipulation
pydantic           # Data models
pydantic-settings  # Env-var config
pymupdf            # PDF text extraction
pyyaml             # Review questions config
pytest             # Tests
```