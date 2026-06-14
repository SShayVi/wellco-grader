# Future: Agent-Based Grader

This folder archives the full agent-based grading system built in the first iteration.
It is **not active** — the live pipeline now uses a simpler direct-CSV approach.

## What's here

| Path | Purpose |
|---|---|
| `grader/agents/base.py` | `BaseAgent`: Claude API wrapper with retry + rate-limit backoff |
| `grader/agents/prediction_agent.py` | Scans a GitHub repo for prediction CSVs, maps columns via LLM + heuristics |
| `grader/agents/n_extractor.py` | Finds the candidate's recommended N across CSV / README / code / PDF |
| `grader/agents/code_reviewer.py` | Scores a repo against a YAML rubric (0/1/2 per question) |
| `grader/sources/github.py` | `GitHubClient`: recursive repo traversal + file download with rate-limit backoff |
| `config/review_questions.yaml` | 8-question rubric (sabotage checks + quality dimensions) |

## When to reactivate

Re-enable this system when you want:
- **Code review scores** alongside prediction scores
- **N extraction** (auto-detect recommended N instead of asking candidates to provide it)
- **GitHub-native workflow** (candidates submit repo URLs instead of CSV links)

## How to reactivate

1. Copy `grader/agents/` and `grader/sources/github.py` back into the main `grader/` tree.
2. Copy `config/review_questions.yaml` back to `config/`.
3. Restore the old `grader/pipeline.py` (see git history: the commit before the "simplify to direct-CSV" commit).
4. Update `config/settings.py` to add back `github_token`, `max_csv_candidates`, `max_repo_chars`.
5. Update `requirements.txt` to re-add `PyGithub>=2.3.0` and `PyMuPDF>=1.24.0`.
6. Update the Google Sheet columns back to `candidate_name`, `repo_url`.
7. Restore the **Code Review Viewer** section in `dashboard/app.py`.

## Design notes

- All three agents share `BaseAgent` and receive file contents from `GitHubClient` — they never call the GitHub API directly.
- Each agent uses Claude tool-use (`tool_choice: any`) for structured JSON output.
- The pipeline processes agents sequentially per candidate; N extraction and code review are **non-fatal** (a failure skips that signal but doesn't drop the candidate).
- Cache key was `(candidate_name, commit_sha)` — a repo is processed at most once per commit.
