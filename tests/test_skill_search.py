import pytest

from services.memory import skill_search as ss


def _http(routes):
    def call(method, url, headers=None):
        for frag, resp in routes.items():
            if frag in url:
                return resp
        return (404, {})
    return call


def test_code_search_with_token_maps_items():
    routes = {"search/code": (200, {"items": [
        {"name": "SKILL.md", "path": "skills/foo/SKILL.md",
         "html_url": "https://github.com/o/r/blob/main/skills/foo/SKILL.md",
         "repository": {"full_name": "o/r", "description": "d", "stargazers_count": 5}}]})}
    out = ss.search_skills("foo", token="ghp_x", http=_http(routes))
    assert out[0]["source"] == "code"
    assert out[0]["repo"] == "o/r"
    assert out[0]["stars"] == 5
    assert out[0]["url"].endswith("/SKILL.md")


def test_repo_search_no_token_includes_only_repos_with_skill_md():
    routes = {
        "search/repositories": (200, {"items": [
            {"full_name": "o/has", "description": "d", "stargazers_count": 3, "default_branch": "main",
             "html_url": "https://github.com/o/has"},
            {"full_name": "o/no", "description": "d", "stargazers_count": 1, "default_branch": "main",
             "html_url": "https://github.com/o/no"}]}),
        "o/has/main/SKILL.md": (200, {}),
        "o/no/main/SKILL.md": (404, {}),
    }
    out = ss.search_skills("foo", token="", http=_http(routes))
    repos = [r["repo"] for r in out]
    assert "o/has" in repos and "o/no" not in repos
    assert out[0]["source"] == "repo"
    assert out[0]["url"] == "https://github.com/o/has/blob/main/SKILL.md"


def test_rate_limit_raises():
    routes = {"search/repositories": (403, {"message": "API rate limit exceeded"})}
    with pytest.raises(ss.SkillSearchError):
        ss.search_skills("foo", token="", http=_http(routes))


def test_blank_query_returns_empty():
    assert ss.search_skills("   ", token="", http=_http({})) == []
