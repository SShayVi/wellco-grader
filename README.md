# wellco-grader

Automated evaluation system for the WellCo Data Science home assignment.

Reads a Google Sheet of candidates, scans their GitHub repos with AI agents,
scores predictions against true churn labels, and serves a live leaderboard dashboard.

---

## Architecture

```
Google Sheet (candidate_name, repo_url)
        ↓
  GitHubClient  (repo traversal, file download, rate-limit backoff)
        ↓
  PredictionAgent   NExtractorAgent   CodeReviewerAgent
  (find + normalize  (find recommended  (YAML rubric → scored
   predictions CSV)   outreach N)        review with justifications)
        ↓
  SQLite Cache (keyed by candidate + commit SHA — skip re-processing unchanged repos)
        ↓
  Scorer  ←── data/test_churn_labels.csv
        ↓
  Streamlit Dashboard (leaderboard | precision@N chart | review viewer)
```

**Core invariant**: a repo is processed at most once per commit SHA. Re-running the
pipeline only processes new candidates or candidates who pushed new commits.

---

## Setup

### 1. Clone and install

```bash
git clone <this-repo>
cd wellco-grader
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in required values
```

Required:
- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)
- `GOOGLE_SHEET_ID` — the ID from your Google Sheet URL
- `GITHUB_TOKEN` — recommended to avoid 60 req/hr anonymous limit

### 3. Add true labels

Place `test_churn_labels.csv` in the `data/` directory.
This file is gitignored and must never be committed to a public repo.

### 4. Create the Google Sheet

Create a public Google Sheet with columns: `candidate_name`, `repo_url`

| candidate_name | repo_url |
|---|---|
| Shay Shavit | https://github.com/SShayVi/wellco-churn-prediction-home-assignment |

Share → Anyone with the link → Viewer.

Copy the sheet ID from the URL:
`https://docs.google.com/spreadsheets/d/**SHEET_ID**/edit`

---

## Running

### Run the pipeline

Processes all candidates in the Google Sheet. Skips already-cached repos.

```bash
python -m grader
```

### Start the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard auto-refreshes every 60 seconds and reads from the SQLite cache.

---

## Testing

### Unit tests (no API keys needed)

```bash
pytest tests/unit/ -v
```

### Integration tests (real GitHub + Anthropic API)

```bash
pytest tests/ --integration -v
```

Integration tests run against Shay's repo as a live fixture and consume API credits.

---

## Customizing the Review Rubric

Edit `config/review_questions.yaml`. Each question has:
- `id`: unique identifier
- `text`: the question shown to the LLM reviewer
- `weight`: relative importance (higher = more impact on overall score)

Changes take effect on the next pipeline run (no code changes needed).

---

## Deploying the Dashboard Publicly

**Option A: Streamlit Community Cloud** (recommended for sharing)
1. Push this repo to GitHub (without `data/` — it's gitignored)
2. Go to share.streamlit.io and connect the repo
3. Set secrets in the Streamlit Cloud dashboard (same as `.env` vars)
4. SQLite is ephemeral on Streamlit Cloud — for persistence, swap `ResultCache`
   to use a hosted database (Supabase, PlanetScale, etc.)

**Option B: Local + ngrok**
```bash
streamlit run dashboard/app.py &
ngrok http 8501
```

---

## Project Structure

```
wellco-grader/
├── grader/
│   ├── agents/          # PredictionAgent, NExtractorAgent, CodeReviewerAgent
│   ├── sources/         # GoogleSheets + GitHub clients
│   ├── scoring/         # metrics.py (precision@N) + scorer.py
│   ├── storage/         # Pydantic models + SQLite cache
│   ├── pipeline.py      # Orchestrator
│   └── __main__.py      # CLI entry point
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── config/
│   ├── settings.py      # Pydantic Settings (env vars)
│   └── review_questions.yaml
├── tests/
│   ├── unit/            # Fast tests, no API calls
│   ├── integration/     # Real API calls, --integration flag
│   └── fixtures/csvs/   # Edge-case prediction CSVs
└── data/                # True labels (gitignored)
```
