"""Abstract version-control hosting interface + GitHub / GitLab clients."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class PullRequest:
    """Normalised pull/merge request."""

    number: int
    title: str
    body: str
    state: str
    head_ref: str
    base_ref: str
    author: str
    url: str
    draft: bool = False
    merged: bool = False


@dataclass
class Comment:
    """One PR / issue comment."""

    id: int
    author: str
    body: str
    created_at: str


@dataclass
class ReviewComment:
    """An inline review comment tied to a path + line."""

    id: int
    author: str
    body: str
    path: str
    line: int


@dataclass
class Issue:
    """Tracker issue."""

    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    assignees: list[str]
    url: str


@dataclass
class CICheck:
    """One CI check on a commit."""

    name: str
    status: str
    conclusion: str | None
    url: str
    details: str = ""


@dataclass
class RepoFile:
    """File entry returned by directory listings."""

    path: str
    type: str
    size: int


class VCSHostClient(ABC):
    """Abstract interface shared by GitHub and GitLab implementations."""

    @abstractmethod
    async def get_pull_request(self, repo: str, pr_number: int) -> PullRequest: ...
    @abstractmethod
    async def list_pull_requests(self, repo: str, state: str = "open", author: str | None = None) -> list[PullRequest]: ...
    @abstractmethod
    async def create_pull_request(self, repo: str, title: str, body: str, head: str, base: str, draft: bool = False) -> PullRequest: ...
    @abstractmethod
    async def update_pull_request(self, repo: str, pr_number: int, **kwargs: Any) -> PullRequest: ...
    @abstractmethod
    async def add_pr_comment(self, repo: str, pr_number: int, body: str) -> Comment: ...
    @abstractmethod
    async def add_pr_review_comment(self, repo: str, pr_number: int, path: str, line: int, body: str) -> ReviewComment: ...
    @abstractmethod
    async def get_pr_diff(self, repo: str, pr_number: int) -> str: ...
    @abstractmethod
    async def get_issue(self, repo: str, issue_number: int) -> Issue: ...
    @abstractmethod
    async def list_issues(self, repo: str, labels: list[str] | None = None, assignee: str | None = None, state: str = "open") -> list[Issue]: ...
    @abstractmethod
    async def create_issue(self, repo: str, title: str, body: str, labels: list[str] | None = None, assignees: list[str] | None = None) -> Issue: ...
    @abstractmethod
    async def close_issue(self, repo: str, issue_number: int, comment: str | None = None) -> None: ...
    @abstractmethod
    async def get_ci_status(self, repo: str, commit_sha: str) -> list[CICheck]: ...
    @abstractmethod
    async def list_repo_files(self, repo: str, path: str = "", ref: str | None = None) -> list[RepoFile]: ...
    @abstractmethod
    async def get_file_content(self, repo: str, path: str, ref: str | None = None) -> str: ...


# ---------- GitHub ----------


class GitHubClient(VCSHostClient):
    """REST GitHub client backed by ``httpx`` (no PyGithub dependency)."""

    def __init__(
        self,
        token: str | None = None,
        api_base: str = "https://api.github.com",
    ) -> None:
        """Initialise the GitHub client.

        Args:
            token: Personal-access token. Falls back to ``GITHUB_TOKEN`` env var.
            api_base: API base URL (override for GH Enterprise).
        """
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._api = api_base.rstrip("/")

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if extra:
            h.update(extra)
        return h

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx

        url = f"{self._api}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url, headers=self._headers(kwargs.pop("headers", None)), **kwargs)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            return resp.text

    async def get_pull_request(self, repo: str, pr_number: int) -> PullRequest:
        data = await self._call("GET", f"/repos/{repo}/pulls/{pr_number}")
        return _pr_from_gh(data)

    async def list_pull_requests(self, repo: str, state: str = "open", author: str | None = None) -> list[PullRequest]:
        params = {"state": state, "per_page": 50}
        data = await self._call("GET", f"/repos/{repo}/pulls", params=params)
        prs = [_pr_from_gh(d) for d in data or []]
        if author:
            prs = [p for p in prs if p.author == author]
        return prs

    async def create_pull_request(self, repo: str, title: str, body: str, head: str, base: str, draft: bool = False) -> PullRequest:
        payload = {"title": title, "body": body, "head": head, "base": base, "draft": draft}
        data = await self._call("POST", f"/repos/{repo}/pulls", json=payload)
        return _pr_from_gh(data)

    async def update_pull_request(self, repo: str, pr_number: int, **kwargs: Any) -> PullRequest:
        data = await self._call("PATCH", f"/repos/{repo}/pulls/{pr_number}", json=kwargs)
        return _pr_from_gh(data)

    async def add_pr_comment(self, repo: str, pr_number: int, body: str) -> Comment:
        data = await self._call("POST", f"/repos/{repo}/issues/{pr_number}/comments", json={"body": body})
        return Comment(
            id=int(data.get("id", 0)),
            author=(data.get("user") or {}).get("login", ""),
            body=data.get("body", ""),
            created_at=data.get("created_at", ""),
        )

    async def add_pr_review_comment(self, repo: str, pr_number: int, path: str, line: int, body: str) -> ReviewComment:
        pr = await self.get_pull_request(repo, pr_number)
        commit_sha = pr.head_ref  # NB: caller often passes the SHA via head_ref already
        payload = {"body": body, "commit_id": commit_sha, "path": path, "line": line, "side": "RIGHT"}
        data = await self._call("POST", f"/repos/{repo}/pulls/{pr_number}/comments", json=payload)
        return ReviewComment(
            id=int(data.get("id", 0)),
            author=(data.get("user") or {}).get("login", ""),
            body=data.get("body", ""),
            path=data.get("path", path),
            line=int(data.get("line", line) or line),
        )

    async def get_pr_diff(self, repo: str, pr_number: int) -> str:
        return await self._call(
            "GET", f"/repos/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )

    async def get_issue(self, repo: str, issue_number: int) -> Issue:
        data = await self._call("GET", f"/repos/{repo}/issues/{issue_number}")
        return _issue_from_gh(data)

    async def list_issues(self, repo: str, labels: list[str] | None = None, assignee: str | None = None, state: str = "open") -> list[Issue]:
        params: dict[str, Any] = {"state": state, "per_page": 50}
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee
        data = await self._call("GET", f"/repos/{repo}/issues", params=params)
        return [_issue_from_gh(d) for d in data or [] if "pull_request" not in d]

    async def create_issue(self, repo: str, title: str, body: str, labels: list[str] | None = None, assignees: list[str] | None = None) -> Issue:
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        data = await self._call("POST", f"/repos/{repo}/issues", json=payload)
        return _issue_from_gh(data)

    async def close_issue(self, repo: str, issue_number: int, comment: str | None = None) -> None:
        if comment:
            await self.add_pr_comment(repo, issue_number, comment)
        await self._call("PATCH", f"/repos/{repo}/issues/{issue_number}", json={"state": "closed"})

    async def get_ci_status(self, repo: str, commit_sha: str) -> list[CICheck]:
        data = await self._call("GET", f"/repos/{repo}/commits/{commit_sha}/check-runs")
        out: list[CICheck] = []
        for run in (data or {}).get("check_runs", []) or []:
            out.append(CICheck(
                name=run.get("name", ""),
                status=run.get("status", ""),
                conclusion=run.get("conclusion"),
                url=run.get("html_url", ""),
                details=run.get("output", {}).get("summary", "")[:1000],
            ))
        return out

    async def list_repo_files(self, repo: str, path: str = "", ref: str | None = None) -> list[RepoFile]:
        params = {"ref": ref} if ref else None
        data = await self._call("GET", f"/repos/{repo}/contents/{path}", params=params)
        if not isinstance(data, list):
            data = [data]
        return [
            RepoFile(path=d.get("path", ""), type=d.get("type", ""), size=int(d.get("size", 0) or 0))
            for d in data
        ]

    async def get_file_content(self, repo: str, path: str, ref: str | None = None) -> str:
        params = {"ref": ref} if ref else None
        data = await self._call("GET", f"/repos/{repo}/contents/{path}", params=params)
        import base64

        return base64.b64decode(data.get("content", "")).decode("utf-8", "replace")


def _pr_from_gh(data: dict[str, Any]) -> PullRequest:
    """Normalise a GitHub PR dict to :class:`PullRequest`."""
    return PullRequest(
        number=int(data.get("number", 0)),
        title=data.get("title", ""),
        body=data.get("body") or "",
        state=data.get("state", "open"),
        head_ref=(data.get("head") or {}).get("ref", ""),
        base_ref=(data.get("base") or {}).get("ref", ""),
        author=(data.get("user") or {}).get("login", ""),
        url=data.get("html_url", ""),
        draft=bool(data.get("draft", False)),
        merged=bool(data.get("merged", False)),
    )


def _issue_from_gh(data: dict[str, Any]) -> Issue:
    """Normalise a GitHub issue dict to :class:`Issue`."""
    return Issue(
        number=int(data.get("number", 0)),
        title=data.get("title", ""),
        body=data.get("body") or "",
        state=data.get("state", "open"),
        labels=[l.get("name", "") if isinstance(l, dict) else str(l) for l in (data.get("labels") or [])],
        assignees=[(a or {}).get("login", "") for a in (data.get("assignees") or [])],
        url=data.get("html_url", ""),
    )


# ---------- GitLab ----------


class GitLabClient(VCSHostClient):
    """GitLab REST client (supports self-hosted instances via ``api_base``)."""

    def __init__(
        self,
        token: str | None = None,
        api_base: str = "https://gitlab.com/api/v4",
    ) -> None:
        """Initialise the GitLab client.

        Args:
            token: Personal-access token (otherwise reads ``GITLAB_TOKEN``).
            api_base: API base URL.
        """
        self._token = token or os.environ.get("GITLAB_TOKEN")
        self._api = api_base.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"PRIVATE-TOKEN": self._token} if self._token else {}

    @staticmethod
    def _enc(repo: str) -> str:
        """URL-encode a ``namespace/project`` repo id."""
        import urllib.parse

        return urllib.parse.quote(repo, safe="")

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, f"{self._api}{path}", headers=self._headers(), **kwargs)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            return resp.text

    async def get_pull_request(self, repo: str, pr_number: int) -> PullRequest:
        data = await self._call("GET", f"/projects/{self._enc(repo)}/merge_requests/{pr_number}")
        return _pr_from_gl(data)

    async def list_pull_requests(self, repo: str, state: str = "open", author: str | None = None) -> list[PullRequest]:
        state_map = {"open": "opened", "closed": "closed", "merged": "merged", "all": "all"}
        params: dict[str, Any] = {"state": state_map.get(state, state), "per_page": 50}
        if author:
            params["author_username"] = author
        data = await self._call("GET", f"/projects/{self._enc(repo)}/merge_requests", params=params)
        return [_pr_from_gl(d) for d in data or []]

    async def create_pull_request(self, repo: str, title: str, body: str, head: str, base: str, draft: bool = False) -> PullRequest:
        payload = {"title": title, "description": body, "source_branch": head, "target_branch": base}
        if draft:
            payload["title"] = "Draft: " + title
        data = await self._call("POST", f"/projects/{self._enc(repo)}/merge_requests", json=payload)
        return _pr_from_gl(data)

    async def update_pull_request(self, repo: str, pr_number: int, **kwargs: Any) -> PullRequest:
        data = await self._call("PUT", f"/projects/{self._enc(repo)}/merge_requests/{pr_number}", json=kwargs)
        return _pr_from_gl(data)

    async def add_pr_comment(self, repo: str, pr_number: int, body: str) -> Comment:
        data = await self._call("POST", f"/projects/{self._enc(repo)}/merge_requests/{pr_number}/notes", json={"body": body})
        return Comment(
            id=int(data.get("id", 0)),
            author=(data.get("author") or {}).get("username", ""),
            body=data.get("body", ""),
            created_at=data.get("created_at", ""),
        )

    async def add_pr_review_comment(self, repo: str, pr_number: int, path: str, line: int, body: str) -> ReviewComment:
        payload = {
            "body": body,
            "position": {
                "base_sha": "",
                "start_sha": "",
                "head_sha": "",
                "position_type": "text",
                "new_path": path,
                "new_line": line,
            },
        }
        data = await self._call("POST", f"/projects/{self._enc(repo)}/merge_requests/{pr_number}/discussions", json=payload)
        note = ((data.get("notes") or [{}])[0]) if isinstance(data, dict) else {}
        return ReviewComment(
            id=int(note.get("id", 0)),
            author=(note.get("author") or {}).get("username", ""),
            body=note.get("body", body),
            path=path,
            line=line,
        )

    async def get_pr_diff(self, repo: str, pr_number: int) -> str:
        data = await self._call("GET", f"/projects/{self._enc(repo)}/merge_requests/{pr_number}/changes")
        diffs = (data or {}).get("changes", []) or []
        return "\n".join(d.get("diff", "") for d in diffs)

    async def get_issue(self, repo: str, issue_number: int) -> Issue:
        data = await self._call("GET", f"/projects/{self._enc(repo)}/issues/{issue_number}")
        return _issue_from_gl(data)

    async def list_issues(self, repo: str, labels: list[str] | None = None, assignee: str | None = None, state: str = "open") -> list[Issue]:
        params: dict[str, Any] = {"state": "opened" if state == "open" else state, "per_page": 50}
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee_username"] = assignee
        data = await self._call("GET", f"/projects/{self._enc(repo)}/issues", params=params)
        return [_issue_from_gl(d) for d in data or []]

    async def create_issue(self, repo: str, title: str, body: str, labels: list[str] | None = None, assignees: list[str] | None = None) -> Issue:
        payload: dict[str, Any] = {"title": title, "description": body}
        if labels:
            payload["labels"] = ",".join(labels)
        if assignees:
            payload["assignee_ids"] = []  # would require username→id lookup
        data = await self._call("POST", f"/projects/{self._enc(repo)}/issues", json=payload)
        return _issue_from_gl(data)

    async def close_issue(self, repo: str, issue_number: int, comment: str | None = None) -> None:
        if comment:
            await self._call(
                "POST",
                f"/projects/{self._enc(repo)}/issues/{issue_number}/notes",
                json={"body": comment},
            )
        await self._call("PUT", f"/projects/{self._enc(repo)}/issues/{issue_number}", json={"state_event": "close"})

    async def get_ci_status(self, repo: str, commit_sha: str) -> list[CICheck]:
        data = await self._call("GET", f"/projects/{self._enc(repo)}/repository/commits/{commit_sha}/statuses")
        out: list[CICheck] = []
        for s in data or []:
            out.append(CICheck(
                name=s.get("name", ""),
                status=s.get("status", ""),
                conclusion=s.get("status"),
                url=s.get("target_url", ""),
                details=s.get("description", ""),
            ))
        return out

    async def list_repo_files(self, repo: str, path: str = "", ref: str | None = None) -> list[RepoFile]:
        params = {"path": path}
        if ref:
            params["ref"] = ref
        data = await self._call("GET", f"/projects/{self._enc(repo)}/repository/tree", params=params)
        return [RepoFile(path=d.get("path", ""), type=d.get("type", ""), size=0) for d in data or []]

    async def get_file_content(self, repo: str, path: str, ref: str | None = None) -> str:
        import urllib.parse

        params = {"ref": ref or "HEAD"}
        data = await self._call(
            "GET",
            f"/projects/{self._enc(repo)}/repository/files/{urllib.parse.quote(path, safe='')}/raw",
            params=params,
        )
        return data if isinstance(data, str) else str(data)


def _pr_from_gl(data: dict[str, Any]) -> PullRequest:
    """Normalise a GitLab MR dict to :class:`PullRequest`."""
    return PullRequest(
        number=int(data.get("iid", 0)),
        title=data.get("title", ""),
        body=data.get("description") or "",
        state=data.get("state", "opened"),
        head_ref=data.get("source_branch", ""),
        base_ref=data.get("target_branch", ""),
        author=(data.get("author") or {}).get("username", ""),
        url=data.get("web_url", ""),
        draft=bool(data.get("draft") or data.get("work_in_progress")),
        merged=data.get("state") == "merged",
    )


def _issue_from_gl(data: dict[str, Any]) -> Issue:
    """Normalise a GitLab issue dict to :class:`Issue`."""
    return Issue(
        number=int(data.get("iid", 0)),
        title=data.get("title", ""),
        body=data.get("description") or "",
        state="open" if data.get("state") == "opened" else data.get("state", ""),
        labels=list(data.get("labels") or []),
        assignees=[(a or {}).get("username", "") for a in (data.get("assignees") or [])],
        url=data.get("web_url", ""),
    )


# ---------- registry ----------


def _default_repo() -> str:
    """Resolve the current ``owner/repo`` from ``GITHUB_REPOSITORY`` or git remote."""
    env = os.environ.get("GITHUB_REPOSITORY")
    if env:
        return env
    return os.environ.get("AGENT_REPO", "")


def _build_client() -> VCSHostClient:
    """Select GitHub or GitLab based on env / config."""
    if os.environ.get("GITLAB_TOKEN"):
        return GitLabClient()
    return GitHubClient()


@GLOBAL_REGISTRY.tool(
    description="Fetch a GitHub/GitLab PR with title, body, head/base refs, and URL.",
    side_effect="network",
    timeout=60.0,
)
async def gh_get_pr(pr_number: int, repo: str = "") -> dict:
    """Return one PR's metadata."""
    client = _build_client()
    pr = await client.get_pull_request(repo or _default_repo(), pr_number)
    return pr.__dict__


@GLOBAL_REGISTRY.tool(
    description="Create a new pull/merge request.",
    side_effect="destructive",
    timeout=60.0,
)
async def gh_create_pr(title: str, body: str, head: str, base: str = "main", draft: bool = False, repo: str = "") -> dict:
    """Create a pull request."""
    client = _build_client()
    pr = await client.create_pull_request(repo or _default_repo(), title, body, head, base, draft=draft)
    return pr.__dict__


@GLOBAL_REGISTRY.tool(
    description="Get the unified diff of a pull/merge request.",
    side_effect="network",
    timeout=60.0,
)
async def gh_pr_diff(pr_number: int, repo: str = "") -> dict:
    """Fetch a PR's diff."""
    client = _build_client()
    diff = await client.get_pr_diff(repo or _default_repo(), pr_number)
    return {"pr_number": pr_number, "diff": diff[:300_000]}


@GLOBAL_REGISTRY.tool(
    description="Fetch a tracker issue with title, body, labels, assignees.",
    side_effect="network",
    timeout=60.0,
)
async def gh_get_issue(issue_number: int, repo: str = "") -> dict:
    """Get one issue."""
    client = _build_client()
    issue = await client.get_issue(repo or _default_repo(), issue_number)
    return issue.__dict__


@GLOBAL_REGISTRY.tool(
    description="Fetch CI checks for a commit SHA.",
    side_effect="network",
    timeout=60.0,
)
async def gh_ci_status(commit_sha: str, repo: str = "") -> dict:
    """List CI checks for a commit."""
    client = _build_client()
    checks = await client.get_ci_status(repo or _default_repo(), commit_sha)
    failing = [c.__dict__ for c in checks if (c.conclusion or "").lower() in {"failure", "cancelled", "timed_out"}]
    return {
        "commit_sha": commit_sha,
        "checks": [c.__dict__ for c in checks],
        "failing": failing,
    }


@GLOBAL_REGISTRY.tool(
    description="Post a top-level comment on a PR / issue.",
    side_effect="destructive",
    timeout=60.0,
)
async def gh_comment(pr_number: int, body: str, repo: str = "") -> dict:
    """Add a comment to a PR / issue."""
    client = _build_client()
    cmt = await client.add_pr_comment(repo or _default_repo(), pr_number, body)
    return cmt.__dict__
