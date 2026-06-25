"""Tests for the VCS hosting clients (httpx mocked at the transport layer)."""

from __future__ import annotations

import json

import httpx
import pytest

from agent.tools.vcs_hosting import (
    GitHubClient,
    GitLabClient,
    _issue_from_gh,
    _issue_from_gl,
    _pr_from_gh,
    _pr_from_gl,
)


def _routes(handler):
    """Build a per-request httpx mock client by patching AsyncClient.send."""
    transport = httpx.MockTransport(handler)
    return transport


def test_pr_from_gh_normalisation() -> None:
    pr = _pr_from_gh({
        "number": 1, "title": "x", "body": "y", "state": "open",
        "head": {"ref": "feature"}, "base": {"ref": "main"},
        "user": {"login": "alice"}, "html_url": "u", "draft": False, "merged": False,
    })
    assert pr.number == 1 and pr.head_ref == "feature" and pr.author == "alice"


def test_pr_from_gl_normalisation() -> None:
    pr = _pr_from_gl({
        "iid": 7, "title": "mr", "description": "...", "state": "opened",
        "source_branch": "b", "target_branch": "main",
        "author": {"username": "bob"}, "web_url": "u",
    })
    assert pr.number == 7 and pr.author == "bob"


def test_issue_normalisation() -> None:
    issue_gh = _issue_from_gh({"number": 9, "title": "t", "body": "b", "state": "open", "labels": [{"name": "bug"}], "assignees": [{"login": "a"}], "html_url": "u"})
    assert issue_gh.labels == ["bug"] and issue_gh.assignees == ["a"]
    issue_gl = _issue_from_gl({"iid": 9, "title": "t", "description": "b", "state": "opened", "labels": ["bug"], "assignees": [{"username": "a"}], "web_url": "u"})
    assert issue_gl.labels == ["bug"]


@pytest.mark.asyncio
async def test_github_get_pull_request_uses_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "number": 5, "title": "t", "body": "b", "state": "open",
                "head": {"ref": "f"}, "base": {"ref": "main"},
                "user": {"login": "u"}, "html_url": "x",
            },
        )

    transport = httpx.MockTransport(handler)

    class _PatchedClient(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = transport
            super().__init__(*a, **kw)

    monkeypatch.setattr("httpx.AsyncClient", _PatchedClient)

    client = GitHubClient(token="t-123")
    pr = await client.get_pull_request("owner/repo", 5)
    assert pr.number == 5
    assert seen_headers.get("authorization") == "Bearer t-123"


@pytest.mark.asyncio
async def test_github_create_pr_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        captured["body"] = body
        return httpx.Response(
            201,
            json={
                "number": 11, "title": body["title"], "body": body["body"], "state": "open",
                "head": {"ref": body["head"]}, "base": {"ref": body["base"]},
                "user": {"login": "me"}, "html_url": "u",
            },
        )

    real_cls = httpx.AsyncClient

    def _factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_cls(*a, **kw)

    monkeypatch.setattr("httpx.AsyncClient", _factory)
    client = GitHubClient(token="t")
    pr = await client.create_pull_request("o/r", "title", "body", "feat", "main", draft=True)
    assert captured["body"]["draft"] is True
    assert pr.number == 11


@pytest.mark.asyncio
async def test_gitlab_get_pull_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "iid": 3, "title": "x", "description": "y", "state": "opened",
                "source_branch": "f", "target_branch": "main",
                "author": {"username": "ada"}, "web_url": "u",
            },
        )

    real_cls = httpx.AsyncClient

    def _factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_cls(*a, **kw)

    monkeypatch.setattr("httpx.AsyncClient", _factory)
    client = GitLabClient(token="t")
    pr = await client.get_pull_request("group/proj", 3)
    assert pr.number == 3
    assert pr.author == "ada"
