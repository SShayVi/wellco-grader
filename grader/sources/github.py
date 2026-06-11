import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional

from github import Github, GithubException, UnknownObjectException
from github.Repository import Repository

logger = logging.getLogger(__name__)

_RATE_LIMIT_WAIT_BASE = 5  # seconds, doubles each retry
_MAX_RETRIES = 5
_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB — skip binaries


@dataclass
class RepoFile:
    path: str
    size: int


class RepoUnavailableError(Exception):
    pass


class GitHubClient:
    def __init__(self, token: str = "") -> None:
        self._gh = Github(token) if token else Github()

    def _repo_path_from_url(self, url: str) -> str:
        """Parse 'https://github.com/owner/repo' → 'owner/repo'."""
        url = url.rstrip("/")
        if "github.com/" not in url:
            raise ValueError(f"Not a GitHub URL: {url}")
        return url.split("github.com/", 1)[1]

    def get_repo(self, url: str) -> Repository:
        path = self._repo_path_from_url(url)
        try:
            return self._gh.get_repo(path)
        except UnknownObjectException:
            raise RepoUnavailableError(f"Repo not found or private: {url}")
        except GithubException as e:
            raise RepoUnavailableError(f"GitHub error for {url}: {e}")

    def get_latest_sha(self, repo: Repository) -> str:
        branch = repo.default_branch
        return repo.get_branch(branch).commit.sha

    def list_files(
        self,
        repo: Repository,
        extensions: Optional[list[str]] = None,
    ) -> list[RepoFile]:
        """
        Recursively list all files in the repo.
        If extensions is given, only include files with those extensions.
        Files larger than _MAX_FILE_SIZE are excluded.
        """
        results: list[RepoFile] = []
        stack = [""]

        while stack:
            path = stack.pop()
            try:
                items = repo.get_contents(path)
            except GithubException as e:
                logger.warning("Cannot list %s: %s", path or "(root)", e)
                continue

            if not isinstance(items, list):
                items = [items]

            for item in items:
                if item.type == "dir":
                    stack.append(item.path)
                else:
                    if item.size > _MAX_FILE_SIZE:
                        continue
                    if extensions is None or any(
                        item.path.lower().endswith(ext) for ext in extensions
                    ):
                        results.append(RepoFile(path=item.path, size=item.size))

        return results

    def download_file(self, repo: Repository, path: str) -> bytes:
        """Download a single file with exponential-backoff retry."""
        for attempt in range(_MAX_RETRIES):
            try:
                content_file = repo.get_contents(path)
                if isinstance(content_file, list):
                    raise ValueError(f"{path} is a directory, not a file")
                return base64.b64decode(content_file.content)
            except GithubException as e:
                if e.status == 403 and "rate limit" in str(e).lower():
                    wait = _RATE_LIMIT_WAIT_BASE * (2 ** attempt)
                    logger.warning("Rate limit hit, waiting %ds (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                if e.status == 404:
                    raise FileNotFoundError(f"File not found in repo: {path}")
                raise

        raise RuntimeError(f"Failed to download {path} after {_MAX_RETRIES} attempts")
