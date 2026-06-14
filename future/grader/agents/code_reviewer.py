"""
Reviews a GitHub repo against a YAML-defined rubric.
Sends all repo text content to Claude in a single call and returns
per-question scores (0/1/2) with justifications.
"""
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from github.Repository import Repository

from grader.agents.base import BaseAgent
from grader.sources.github import GitHubClient, RepoFile
from grader.storage.models import ReviewQuestionResult, ReviewResult

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = [".py", ".ipynb", ".md", ".txt", ".rst", ".yaml", ".yml", ".r", ".rmd"]
_MAX_CHARS = 150_000
_MAX_FILE_CHARS = 20_000


def _load_questions(yaml_path: Path) -> list[dict]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])
    required_keys = {"id", "text", "weight"}
    for q in questions:
        missing = required_keys - set(q.keys())
        if missing:
            raise ValueError(f"Review question missing keys {missing}: {q}")
    return questions


def _flatten_notebook(content: bytes) -> str:
    """Extract all cell sources from a Jupyter notebook."""
    try:
        nb = json.loads(content)
        parts = []
        for cell in nb.get("cells", []):
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else source
            if text.strip():
                parts.append(f"# [{cell.get('cell_type', 'code')}]\n{text}")
        return "\n\n".join(parts)
    except Exception:
        return content.decode("utf-8", errors="ignore")


def _build_review_tool(questions: list[dict]) -> dict[str, Any]:
    question_ids = [q["id"] for q in questions]
    return {
        "name": "review_submission",
        "description": (
            "Score a WellCo churn prediction submission against a rubric. "
            "For each question, assign a score of 0 (not addressed), "
            "1 (partially addressed), or 2 (fully addressed). "
            "Be critical and evidence-based."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reviews": {
                    "type": "array",
                    "description": f"One entry per question. Must include all {len(question_ids)} questions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "enum": question_ids,
                                "description": "Question ID",
                            },
                            "score": {
                                "type": "integer",
                                "enum": [0, 1, 2],
                                "description": "0=not addressed, 1=partial, 2=full",
                            },
                            "justification": {
                                "type": "string",
                                "description": "Specific evidence from the code supporting this score",
                            },
                        },
                        "required": ["id", "score", "justification"],
                    },
                }
            },
            "required": ["reviews"],
        },
    }


class CodeReviewerAgent(BaseAgent):
    def __init__(
        self,
        api_key: str,
        github_client: GitHubClient,
        questions_path: Path,
        model: str = "claude-sonnet-4-6",
        max_chars: int = _MAX_CHARS,
    ) -> None:
        super().__init__(api_key, model)
        self._gh = github_client
        self._questions = _load_questions(questions_path)
        self._max_chars = max_chars
        self._review_tool = _build_review_tool(self._questions)

    def run(self, repo: Repository) -> ReviewResult:
        repo_text = self._collect_repo_text(repo)
        return self._review(repo_text)

    def _collect_repo_text(self, repo: Repository) -> str:
        files = self._gh.list_files(repo, extensions=_TEXT_EXTENSIONS)

        # Prioritize: root-level files first, then by path depth, then by size
        def priority(f: RepoFile) -> tuple:
            depth = f.path.count("/")
            is_readme = "readme" in f.path.lower()
            is_notebook = f.path.endswith(".ipynb")
            is_py = f.path.endswith(".py")
            return (not is_readme, not is_py, not is_notebook, depth, -f.size)

        files.sort(key=priority)

        parts: list[str] = []
        total_chars = 0

        for f in files:
            if total_chars >= self._max_chars:
                logger.info("Repo text truncated at %d chars (%d files)", total_chars, len(parts))
                break
            try:
                raw = self._gh.download_file(repo, f.path)
                if f.path.endswith(".ipynb"):
                    content = _flatten_notebook(raw)
                else:
                    content = raw.decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning("Cannot read %s: %s", f.path, e)
                continue

            content = content[:_MAX_FILE_CHARS]
            header = f"\n{'='*60}\nFILE: {f.path}\n{'='*60}\n"
            parts.append(header + content)
            total_chars += len(content)

        if not parts:
            return "[No readable text files found in repository]"

        truncation_note = ""
        if total_chars >= self._max_chars:
            truncation_note = f"\n[NOTE: Repository content truncated to {self._max_chars:,} chars]\n"

        return truncation_note + "\n".join(parts)

    def _review(self, repo_text: str) -> ReviewResult:
        questions_text = "\n".join(
            f"{i+1}. [{q['id']}] (weight={q['weight']}) {q['text']}"
            for i, q in enumerate(self._questions)
        )

        response = self._call(
            max_tokens=4096,
            tools=[self._review_tool],
            tool_choice={"type": "any"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a senior data scientist reviewing a WellCo churn prediction "
                        "home assignment submission. Be rigorous and evidence-based.\n\n"
                        f"REVIEW RUBRIC:\n{questions_text}\n\n"
                        f"REPOSITORY CONTENTS:\n{repo_text}\n\n"
                        "Score each question 0 (not done), 1 (partial), 2 (full). "
                        "Cite specific evidence from the code for each score."
                    ),
                }
            ],
        )

        raw = self._extract_tool_input(response)
        reviews_by_id = {r["id"]: r for r in raw.get("reviews", [])}

        question_results: list[ReviewQuestionResult] = []
        for q in self._questions:
            review = reviews_by_id.get(q["id"], {"score": 0, "justification": "Not reviewed"})
            question_results.append(
                ReviewQuestionResult(
                    id=q["id"],
                    score=review["score"],
                    justification=review["justification"],
                    weight=q["weight"],
                )
            )

        total_weight = sum(q["weight"] for q in self._questions)
        weighted_score = (
            sum(r.score * r.weight for r in question_results) / (2 * total_weight)
            if total_weight > 0
            else 0.0
        )

        return ReviewResult(questions=question_results, weighted_score=round(weighted_score, 4))
