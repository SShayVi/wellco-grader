"""
Searches a GitHub repo for the candidate's recommended outreach N.
Search priority: predictions CSV row count → explicit N column →
README/docs → Python/notebook code → PDF presentations → fallback.
"""
import json
import logging
import re
from io import BytesIO
from typing import Optional

import pandas as pd

from grader.agents.base import BaseAgent
from grader.sources.github import GitHubClient, RepoFile
from github.Repository import Repository
from grader.storage.models import NExtractionResult, NSource

logger = logging.getLogger(__name__)

_N_COL_HINTS = {"n_recommended", "n_outreach", "outreach_n", "recommended_n", "optimal_n"}
_N_WARNING_THRESHOLD = 5_000

_EXTRACT_N_TOOL = {
    "name": "extract_n",
    "description": (
        "Extract the recommended outreach N from text. "
        "N is the number of members the candidate recommends contacting."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "The recommended outreach N"},
            "confidence": {"type": "number", "description": "Confidence 0.0–1.0"},
            "explanation": {
                "type": "string",
                "description": "Why this value was chosen as the recommended N",
            },
        },
        "required": ["n", "confidence", "explanation"],
    },
}


def _try_flatten_notebook(content: bytes) -> Optional[str]:
    """Extract code cells from a Jupyter notebook."""
    try:
        nb = json.loads(content)
        cells = nb.get("cells", [])
        parts = []
        for cell in cells:
            if cell.get("cell_type") in ("code", "markdown"):
                source = cell.get("source", [])
                if isinstance(source, list):
                    parts.append("".join(source))
                else:
                    parts.append(source)
        return "\n\n".join(parts)
    except Exception:
        return None


class NExtractorAgent(BaseAgent):
    def __init__(
        self,
        api_key: str,
        github_client: GitHubClient,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        super().__init__(api_key, model)
        self._gh = github_client

    def run(
        self,
        repo: Repository,
        predictions_df: Optional[pd.DataFrame] = None,
        predictions_path: Optional[str] = None,
    ) -> NExtractionResult:
        """
        Search for recommended N in priority order.
        predictions_df is the already-normalized DataFrame if available.
        """
        # 1. Explicit N column in the predictions CSV
        if predictions_df is not None:
            lower_cols = {c.lower(): c for c in predictions_df.columns}
            for hint in _N_COL_HINTS:
                if hint in lower_cols:
                    col = lower_cols[hint]
                    val = predictions_df[col].dropna().iloc[0] if not predictions_df[col].dropna().empty else None
                    if val is not None:
                        n = int(val)
                        logger.info("N from explicit CSV column '%s': %d", col, n)
                        return self._make_result(n, NSource.CSV_EXPLICIT_COLUMN, 1.0)

        # 2. README / docs
        result = self._search_text_files(repo)
        if result:
            return result

        # 3. Python / notebook code files
        result = self._search_code_files(repo)
        if result:
            return result

        # 4. PDF presentations
        result = self._search_pdfs(repo)
        if result:
            return result

        # 5. Fallback: row count of predictions CSV
        if predictions_df is not None:
            n = len(predictions_df)
            logger.info("N inferred from predictions row count: %d", n)
            return self._make_result(n, NSource.INFERRED, 0.5)

        return self._make_result(1000, NSource.INFERRED, 0.1)

    def _search_text_files(self, repo: Repository) -> Optional[NExtractionResult]:
        """Search README and other docs for N mentions."""
        doc_files = self._gh.list_files(repo, extensions=[".md", ".txt", ".rst"])
        readme_files = [f for f in doc_files if "readme" in f.path.lower()]
        other_docs = [f for f in doc_files if "readme" not in f.path.lower()]

        for f in readme_files + other_docs[:3]:
            try:
                content = self._gh.download_file(repo, f.path).decode("utf-8", errors="ignore")
            except Exception:
                continue

            n = self._regex_search(content)
            if n:
                logger.info("N from regex in %s: %d", f.path, n)
                return self._make_result(n, NSource.README, 0.8)

            n = self._llm_extract_n(content[:8000], f.path)
            if n:
                return self._make_result(n, NSource.README, 0.75)

        return None

    def _search_code_files(self, repo: Repository) -> Optional[NExtractionResult]:
        """Search Python scripts and notebooks for N assignments near outreach logic."""
        code_files = self._gh.list_files(repo, extensions=[".py", ".ipynb"])
        outreach_keywords = re.compile(
            r"(outreach|churn|threshold|n_out|n_rec|optimal|elbow|top_n|top_k)",
            re.IGNORECASE,
        )

        candidates: list[tuple[str, str]] = []
        for f in code_files[:10]:
            try:
                raw = self._gh.download_file(repo, f.path)
                if f.path.endswith(".ipynb"):
                    content = _try_flatten_notebook(raw) or raw.decode("utf-8", errors="ignore")
                else:
                    content = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue

            if outreach_keywords.search(content):
                # Extract lines with n= or N= assignments near outreach keywords
                snippets = self._extract_n_snippets(content)
                if snippets:
                    candidates.append((f.path, snippets))

        if not candidates:
            return None

        combined = "\n\n---\n\n".join(
            f"# {path}\n{snippet}" for path, snippet in candidates[:5]
        )
        n = self._llm_extract_n(combined, "code files")
        if n:
            return self._make_result(n, NSource.CODE, 0.7)
        return None

    def _search_pdfs(self, repo: Repository) -> Optional[NExtractionResult]:
        """Extract text from PDF presentations and search for N."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed, skipping PDF search")
            return None

        pdf_files = self._gh.list_files(repo, extensions=[".pdf"])
        for f in pdf_files[:2]:
            try:
                raw = self._gh.download_file(repo, f.path)
                doc = fitz.open(stream=raw, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
            except Exception as e:
                logger.warning("Cannot read PDF %s: %s", f.path, e)
                continue

            n = self._regex_search(text)
            if n:
                return self._make_result(n, NSource.PDF, 0.75)
            n = self._llm_extract_n(text[:8000], f.path)
            if n:
                return self._make_result(n, NSource.PDF, 0.7)

        return None

    def _regex_search(self, text: str) -> Optional[int]:
        """
        Fast regex heuristic: look for patterns like 'N = 1234', 'top 1234 members',
        'outreach 1234', 'recommend 1234'. Requires a plausible range [50, 9999].
        """
        patterns = [
            r"\bN\s*[=:]\s*(\d+)",
            r"\bn_outreach\s*[=:]\s*(\d+)",
            r"\brecommend(?:ed)?\s+(\d+)\s*(?:members?|users?|people)?",
            r"\btop[- ](\d+)\s*(?:members?|users?)?",
            r"\boutreach\s+(\d+)\s*(?:members?|users?)?",
            r"\boptimal\s+N\s*[=:]\s*(\d+)",
            r"\b(\d+)\s*members?\s*(?:for\s+)?outreach",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                n = int(match.group(1))
                if 50 <= n <= 9_999:
                    return n
        return None

    def _extract_n_snippets(self, content: str) -> str:
        """Extract lines containing n= or N= assignments."""
        lines = content.splitlines()
        snippets = []
        for i, line in enumerate(lines):
            if re.search(r"\b[nN]\s*=\s*\d+", line) or re.search(
                r"(outreach|optimal|threshold|top_n|elbow)", line, re.IGNORECASE
            ):
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippets.append("\n".join(lines[start:end]))
        return "\n---\n".join(snippets[:20])

    def _llm_extract_n(self, text: str, source_desc: str) -> Optional[int]:
        """Use LLM to extract N from text when regex fails."""
        try:
            response = self._call(
                max_tokens=256,
                tools=[_EXTRACT_N_TOOL],
                tool_choice={"type": "any"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Source: {source_desc}\n\n"
                            f"{text}\n\n"
                            "Extract the recommended outreach N (number of members to contact) "
                            "from the above text. Only extract if clearly stated."
                        ),
                    }
                ],
            )
            result = self._extract_tool_input(response)
            if result["confidence"] >= 0.6 and 50 <= result["n"] <= 9_999:
                logger.debug("LLM extracted N=%d (conf=%.2f) from %s", result["n"], result["confidence"], source_desc)
                return result["n"]
        except Exception as e:
            logger.warning("LLM N extraction failed for %s: %s", source_desc, e)
        return None

    def _make_result(self, n: int, source: NSource, confidence: float) -> NExtractionResult:
        n = max(1, min(n, 10_000))
        return NExtractionResult(
            n=n,
            source=source,
            confidence=confidence,
            n_warning=n > _N_WARNING_THRESHOLD,
        )
