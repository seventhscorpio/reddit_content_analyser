"""Tests for tag_reddit_content.py."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tag_reddit_content as trc

TEST_DIR = Path(__file__).resolve().parent
DEMO_RULES_CSV = TEST_DIR / "demo_coding_rules" / "coding_rules.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_post() -> Dict[str, Any]:
    """Minimal Reddit post dict using keywords from coding_rules.csv."""
    return {
        "id": "abc123",
        "subreddit": "testsub",
        "author": "alice",
        "created_utc": 1700000000,
        "title": "The AGI prophecy",
        "selftext": "Some believe AGI will be a divine creation.",
        "comments": [
            {
                "id": "c1",
                "author": "bob",
                "created_utc": 1700000001,
                "body": "AGI could be the supreme intelligence.",
            },
            {
                "id": "c2",
                "author": "eve",
                "created_utc": 1700000002,
                "body": "Nothing to see here.",
            },
        ],
    }


@pytest.fixture
def rules_csv() -> Path:
    """Path to the demo coding_rules.csv."""
    return DEMO_RULES_CSV


@pytest.fixture
def json_file(tmp_dir: Path, sample_post: Dict[str, Any]) -> Path:
    """Write sample post to a JSON file."""
    p = tmp_dir / "sample.json"
    p.write_text(json.dumps(sample_post), encoding="utf-8")
    return p


def _make_regex_chunks(
    keywords: List[str],
    case_sensitive: bool = False,
    whole_word: bool = False,
) -> list:
    return trc.build_regex_chunks(
        keywords, case_sensitive=case_sensitive, whole_word=whole_word
    )


# ---------------------------------------------------------------------------
# read_text_safe
# ---------------------------------------------------------------------------

class TestReadTextSafe:
    def test_reads_existing_file(self, tmp_dir: Path) -> None:
        p = tmp_dir / "hello.txt"
        p.write_text("hello world", encoding="utf-8")
        assert trc.read_text_safe(p) == "hello world"

    def test_returns_none_for_missing_file(self, tmp_dir: Path) -> None:
        assert trc.read_text_safe(tmp_dir / "nope.txt") is None

    def test_handles_utf8_with_bom(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bom.txt"
        p.write_bytes(b"\xef\xbb\xbfhello")
        result = trc.read_text_safe(p)
        assert result is not None
        assert "hello" in result


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_valid_json(self, tmp_dir: Path) -> None:
        p = tmp_dir / "data.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        assert trc.load_json(p) == {"a": 1}

    def test_invalid_json(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.json"
        p.write_text("{invalid", encoding="utf-8")
        assert trc.load_json(p) is None

    def test_missing_file(self, tmp_dir: Path) -> None:
        assert trc.load_json(tmp_dir / "missing.json") is None


# ---------------------------------------------------------------------------
# CSV dialect detection
# ---------------------------------------------------------------------------

class TestCsvDialect:
    def test_make_dialect_comma(self) -> None:
        d = trc._make_dialect(",")
        assert d.delimiter == ","

    def test_make_dialect_semicolon(self) -> None:
        d = trc._make_dialect(";")
        assert d.delimiter == ";"

    def test_detect_csv_dialect_comma(self) -> None:
        d = trc._detect_csv_dialect("a,b,c\n1,2,3")
        assert d.delimiter == ","

    def test_detect_csv_dialect_semicolon_only(self) -> None:
        d = trc._detect_csv_dialect("a;b;c")
        assert d.delimiter == ";"

    def test_detect_csv_dialect_tab(self) -> None:
        d = trc._detect_csv_dialect("a\tb\tc")
        assert d.delimiter == "\t"

    def test_detect_csv_dialect_fallback(self) -> None:
        # No delimiters at all → excel fallback
        d = trc._detect_csv_dialect("singleword")
        assert d.delimiter == ","


# ---------------------------------------------------------------------------
# load_keywords_csv
# ---------------------------------------------------------------------------

class TestLoadKeywordsCsv:
    def test_basic_load(self, rules_csv: Path) -> None:
        kws = trc.load_keywords_csv(rules_csv, case_sensitive=False)
        assert len(kws) == 48
        assert kws[0] == "god"
        assert "AGI" in kws
        assert "divine" in kws

    def test_deduplication_case_insensitive(self, tmp_dir: Path) -> None:
        p = tmp_dir / "dup.csv"
        p.write_text("keyword\nAI\nai\nAi\n", encoding="utf-8")
        kws = trc.load_keywords_csv(p, case_sensitive=False)
        assert len(kws) == 1

    def test_deduplication_case_sensitive(self, tmp_dir: Path) -> None:
        p = tmp_dir / "dup.csv"
        p.write_text("keyword\nAI\nai\n", encoding="utf-8")
        kws = trc.load_keywords_csv(p, case_sensitive=True)
        assert len(kws) == 2

    def test_strips_whitespace(self, tmp_dir: Path) -> None:
        p = tmp_dir / "ws.csv"
        p.write_text("keyword\n  hello  \n", encoding="utf-8")
        kws = trc.load_keywords_csv(p, case_sensitive=False)
        assert kws == ["hello"]

    def test_skips_empty_keywords(self, tmp_dir: Path) -> None:
        p = tmp_dir / "empty.csv"
        p.write_text("keyword\n\n  \nfoo\n", encoding="utf-8")
        kws = trc.load_keywords_csv(p, case_sensitive=False)
        assert kws == ["foo"]

    def test_missing_file(self, tmp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            trc.load_keywords_csv(tmp_dir / "nope.csv", False)

    def test_empty_file(self, tmp_dir: Path) -> None:
        p = tmp_dir / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            trc.load_keywords_csv(p, False)

    def test_no_keyword_column(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.csv"
        p.write_text("name\nfoo\n", encoding="utf-8")
        with pytest.raises(ValueError, match="keyword"):
            trc.load_keywords_csv(p, False)

    def test_no_valid_keywords(self, tmp_dir: Path) -> None:
        p = tmp_dir / "novalid.csv"
        p.write_text("keyword\n\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No valid keywords"):
            trc.load_keywords_csv(p, False)

    def test_semicolon_delimiter(self, tmp_dir: Path) -> None:
        p = tmp_dir / "semi.csv"
        p.write_text("id;keyword\n1;hello\n2;world\n", encoding="utf-8")
        kws = trc.load_keywords_csv(p, case_sensitive=False)
        assert "hello" in kws
        assert "world" in kws

    def test_utf8_sig_encoding(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbfkeyword\nfoo\n")
        kws = trc.load_keywords_csv(p, case_sensitive=False)
        assert kws == ["foo"]


# ---------------------------------------------------------------------------
# normalize_fullname
# ---------------------------------------------------------------------------

class TestNormalizeFullname:
    def test_t1_prefix(self) -> None:
        assert trc.normalize_fullname("t1_abc") == "abc"

    def test_t3_prefix(self) -> None:
        assert trc.normalize_fullname("t3_xyz") == "xyz"

    def test_no_prefix(self) -> None:
        assert trc.normalize_fullname("plain") == "plain"

    def test_non_string(self) -> None:
        assert trc.normalize_fullname(123) == ""
        assert trc.normalize_fullname(None) == ""


# ---------------------------------------------------------------------------
# to_iso_utc
# ---------------------------------------------------------------------------

class TestToIsoUtc:
    def test_epoch_seconds(self) -> None:
        result = trc.to_iso_utc(1700000000, "f.json", "$", None)
        assert result == "2023-11-14T22:13:20Z"

    def test_epoch_millis(self) -> None:
        result = trc.to_iso_utc(1700000000000, "f.json", "$", None)
        assert result == "2023-11-14T22:13:20Z"

    def test_string_published(self) -> None:
        result = trc.to_iso_utc(
            "2026-01-30T21:00:33+00:00", "f.json", "$", "published"
        )
        assert result == "2026-01-30T21:00:33+00:00"

    def test_string_non_published(self) -> None:
        result = trc.to_iso_utc("some-date", "f.json", "$", "created_utc")
        assert result == "some-date"

    def test_none_value(self) -> None:
        assert trc.to_iso_utc(None, "f.json", "$") == ""

    def test_none_suppress_missing(self) -> None:
        assert trc.to_iso_utc(None, "f.json", "$", suppress_missing=True) == ""

    def test_unknown_type(self) -> None:
        assert trc.to_iso_utc([], "f.json", "$") == ""


# ---------------------------------------------------------------------------
# get_value
# ---------------------------------------------------------------------------

class TestGetValue:
    def test_first_match(self) -> None:
        d = {"a": 1, "b": 2}
        val, key = trc.get_value(d, ["a", "b"])
        assert val == 1
        assert key == "a"

    def test_skips_none(self) -> None:
        d = {"a": None, "b": 2}
        val, key = trc.get_value(d, ["a", "b"])
        assert val == 2
        assert key == "b"

    def test_all_missing(self) -> None:
        val, key = trc.get_value({}, ["x", "y"])
        assert val is None
        assert key is None


# ---------------------------------------------------------------------------
# get_string_field
# ---------------------------------------------------------------------------

class TestGetStringField:
    def test_returns_string(self) -> None:
        d = {"author": "alice"}
        assert trc.get_string_field(
            d, ["author"], "f", "$", "author"
        ) == "alice"

    def test_nested_dict_with_name(self) -> None:
        d = {"author": {"name": "alice"}}
        assert trc.get_string_field(
            d, ["author"], "f", "$", "author"
        ) == "alice"

    def test_missing_returns_empty(self) -> None:
        assert trc.get_string_field(
            {}, ["author"], "f", "$", "author"
        ) == ""

    def test_non_string_coerced(self) -> None:
        d = {"score": 42}
        assert trc.get_string_field(
            d, ["score"], "f", "$", "score"
        ) == "42"


# ---------------------------------------------------------------------------
# extract_subreddit
# ---------------------------------------------------------------------------

class TestExtractSubreddit:
    def test_direct_name(self) -> None:
        assert trc.extract_subreddit(
            {"subreddit": "python"}, "f", "$"
        ) == "python"

    def test_from_url(self) -> None:
        d = {"url": "/r/singularity/comments/abc/title/"}
        assert trc.extract_subreddit(d, "f", "$") == "singularity"

    def test_from_url_no_trailing_slash(self) -> None:
        d = {"url": "/r/python"}
        assert trc.extract_subreddit(d, "f", "$") == "python"

    def test_prefixed(self) -> None:
        d = {"subreddit_name_prefixed": "r/python"}
        assert trc.extract_subreddit(d, "f", "$") == "python"

    def test_missing(self) -> None:
        assert trc.extract_subreddit({}, "f", "$") == ""


# ---------------------------------------------------------------------------
# extract_post_id
# ---------------------------------------------------------------------------

class TestExtractPostId:
    def test_direct_id(self) -> None:
        assert trc.extract_post_id(
            {"id": "abc"}, "f", "$", "fallback"
        ) == "abc"

    def test_fullname_stripped(self) -> None:
        assert trc.extract_post_id(
            {"id": "t3_abc"}, "f", "$", "fallback"
        ) == "abc"

    def test_from_url(self) -> None:
        d = {"url": "/r/sub/comments/xyz/some_title/"}
        assert trc.extract_post_id(d, "f", "$", "fallback") == "xyz"

    def test_from_permalink(self) -> None:
        d = {"permalink": "/r/sub/comments/xyz/some_title/"}
        assert trc.extract_post_id(d, "f", "$", "fallback") == "xyz"

    def test_url_without_comments_pattern(self) -> None:
        d = {"url": "/r/sub/some_page/"}
        assert trc.extract_post_id(
            d, "f", "$", "fallback"
        ) == "fallback"

    def test_fallback(self) -> None:
        assert trc.extract_post_id({}, "f", "$", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# extract_comment_id
# ---------------------------------------------------------------------------

class TestExtractCommentId:
    def test_direct_id(self) -> None:
        cid, fb = trc.extract_comment_id({"id": "c1"}, "f", "$", "fb")
        assert cid == "c1"
        assert fb is False

    def test_fullname_stripped(self) -> None:
        cid, fb = trc.extract_comment_id({"id": "t1_c1"}, "f", "$", "fb")
        assert cid == "c1"
        assert fb is False

    def test_fallback(self) -> None:
        cid, fb = trc.extract_comment_id({}, "f", "$", "fb")
        assert cid == "fb"
        assert fb is True


# ---------------------------------------------------------------------------
# extract_likes_dislikes
# ---------------------------------------------------------------------------

class TestExtractLikesDislikes:
    def test_score_dict(self) -> None:
        d = {"score": {"likes": 10, "dislikes": 2, "unvoted": 5}}
        likes, dislikes, unvoted = trc.extract_likes_dislikes(d)
        assert likes == "10"
        assert dislikes == "2"
        assert unvoted == "5"

    def test_score_dict_with_nones(self) -> None:
        d = {"score": {"likes": None, "dislikes": None, "unvoted": None}}
        likes, dislikes, unvoted = trc.extract_likes_dislikes(d)
        assert likes == ""
        assert dislikes == ""
        assert unvoted == ""

    def test_ups_downs(self) -> None:
        d = {"ups": 10, "downs": 3}
        likes, dislikes, unvoted = trc.extract_likes_dislikes(d)
        assert likes == "10"
        assert dislikes == "3"
        assert unvoted == ""

    def test_no_score(self) -> None:
        likes, dislikes, unvoted = trc.extract_likes_dislikes({})
        assert likes == ""
        assert dislikes == ""
        assert unvoted == ""


# ---------------------------------------------------------------------------
# post_text / comment_text
# ---------------------------------------------------------------------------

class TestPostText:
    def test_title_and_selftext(self) -> None:
        d = {"title": "Hello", "selftext": "World"}
        assert trc.post_text(d, "f", "$") == "Hello\n\nWorld"

    def test_title_only(self) -> None:
        d = {"title": "Hello"}
        assert trc.post_text(d, "f", "$") == "Hello"

    def test_selftext_only(self) -> None:
        d = {"selftext": "World"}
        assert trc.post_text(d, "f", "$") == "World"

    def test_text_field(self) -> None:
        d = {"text": "Alt body"}
        assert trc.post_text(d, "f", "$") == "Alt body"

    def test_empty(self) -> None:
        assert trc.post_text({}, "f", "$") == ""


class TestCommentText:
    def test_body(self) -> None:
        assert trc.comment_text({"body": "hey"}, "f", "$") == "hey"

    def test_text_field(self) -> None:
        assert trc.comment_text({"text": "hey"}, "f", "$") == "hey"

    def test_empty(self) -> None:
        assert trc.comment_text({}, "f", "$") == ""


# ---------------------------------------------------------------------------
# collect_nodes
# ---------------------------------------------------------------------------

class TestCollectNodes:
    def test_flat_dict(self) -> None:
        d = {"a": 1}
        nodes = list(trc.collect_nodes(d))
        assert any(n is d for n, _, _ in nodes)

    def test_nested_list(self) -> None:
        obj = [{"x": 1}, {"y": 2}]
        nodes = list(trc.collect_nodes(obj))
        assert len([n for n, _, _ in nodes if isinstance(n, dict)]) == 2

    def test_reddit_kind_t3(self) -> None:
        obj = {"kind": "t3", "data": {"title": "Hi"}}
        nodes = list(trc.collect_nodes(obj))
        # Should yield data dict with kind hint
        assert any(
            n.get("title") == "Hi" and kind == "t3"
            for n, _, kind in nodes
        )

    def test_no_infinite_loop(self) -> None:
        """Circular references should not cause infinite recursion."""
        d: Dict[str, Any] = {"a": 1}
        d["self"] = d
        nodes = list(trc.collect_nodes(d))
        assert len(nodes) >= 1


# ---------------------------------------------------------------------------
# score_post / score_comment
# ---------------------------------------------------------------------------

class TestScorePost:
    def test_t3_hint(self) -> None:
        assert trc.score_post({}, "t3") >= 5

    def test_title_boosts(self) -> None:
        assert trc.score_post({"title": "Hi"}, None) > 0

    def test_body_without_title_penalized(self) -> None:
        assert trc.score_post({"body": "x"}, None) < 0

    def test_empty_dict(self) -> None:
        assert trc.score_post({}, None) == 0


class TestScoreComment:
    def test_t1_hint(self) -> None:
        assert trc.score_comment({}, "t1") >= 5

    def test_body_boosts(self) -> None:
        assert trc.score_comment({"body": "x"}, None) > 0

    def test_title_penalizes(self) -> None:
        s = trc.score_comment({"title": "x"}, None)
        assert s < 0

    def test_empty_dict(self) -> None:
        assert trc.score_comment({}, None) == 0


# ---------------------------------------------------------------------------
# find_post_and_comments
# ---------------------------------------------------------------------------

class TestFindPostAndComments:
    def test_simple_post_with_comments(
        self, sample_post: Dict[str, Any]
    ) -> None:
        post, comments = trc.find_post_and_comments(sample_post)
        assert post is not None
        assert post[0]["title"] == "The AGI prophecy"
        assert len(comments) >= 2

    def test_no_post(self) -> None:
        obj = [{"body": "comment only"}]
        post, comments = trc.find_post_and_comments(obj)
        assert post is None
        assert len(comments) >= 1

    def test_reddit_listing(self) -> None:
        """Reddit API listing format with kind/data wrappers."""
        obj = [
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "post1",
                                "title": "Post",
                                "selftext": "content",
                                "subreddit": "test",
                            },
                        }
                    ]
                },
            },
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c1",
                                "body": "comment body",
                                "parent_id": "t3_post1",
                            },
                        }
                    ]
                },
            },
        ]
        post, comments = trc.find_post_and_comments(obj)
        assert post is not None
        assert post[0].get("title") == "Post"
        assert len(comments) >= 1


# ---------------------------------------------------------------------------
# build_phrase_pattern
# ---------------------------------------------------------------------------

class TestBuildPhrasePattern:
    def test_no_word_boundary(self) -> None:
        p = trc.build_phrase_pattern("AI", whole_word=False)
        assert p == "AI"

    def test_word_boundary(self) -> None:
        p = trc.build_phrase_pattern("AI", whole_word=True)
        assert r"(?<!\w)" in p
        assert r"(?!\w)" in p

    def test_special_chars_escaped(self) -> None:
        p = trc.build_phrase_pattern("C++", whole_word=False)
        assert r"\+" in p

    def test_word_boundary_non_alnum_edges(self) -> None:
        # Phrase starting/ending with non-alnum: no boundaries added
        p = trc.build_phrase_pattern("(test)", whole_word=True)
        assert r"(?<!\w)" not in p


# ---------------------------------------------------------------------------
# build_regex_chunks
# ---------------------------------------------------------------------------

class TestBuildRegexChunks:
    def test_single_chunk(self) -> None:
        chunks = trc.build_regex_chunks(["AI", "test"], False, False)
        assert len(chunks) == 1
        regex, mapping = chunks[0]
        assert len(mapping) == 2

    def test_multiple_chunks(self) -> None:
        kws = [f"kw{i}" for i in range(600)]
        chunks = trc.build_regex_chunks(kws, False, False, chunk_size=500)
        assert len(chunks) == 2

    def test_case_insensitive(self) -> None:
        import re
        chunks = trc.build_regex_chunks(["Hello"], False, False)
        regex, _ = chunks[0]
        assert regex.flags & re.IGNORECASE

    def test_case_sensitive(self) -> None:
        import re
        chunks = trc.build_regex_chunks(["Hello"], True, False)
        regex, _ = chunks[0]
        assert not (regex.flags & re.IGNORECASE)


# ---------------------------------------------------------------------------
# find_keyword_matches
# ---------------------------------------------------------------------------

class TestFindKeywordMatches:
    def test_match_found(self) -> None:
        chunks = _make_regex_chunks(["AI"])
        matches = trc.find_keyword_matches("AI is great", chunks)
        assert 1 in matches
        assert matches[1] == "AI"

    def test_no_match(self) -> None:
        chunks = _make_regex_chunks(["python"])
        matches = trc.find_keyword_matches("AI is great", chunks)
        assert len(matches) == 0

    def test_multiple_keywords(self) -> None:
        chunks = _make_regex_chunks(["AI", "great"])
        matches = trc.find_keyword_matches("AI is great", chunks)
        assert len(matches) == 2

    def test_empty_text(self) -> None:
        chunks = _make_regex_chunks(["AI"])
        assert trc.find_keyword_matches("", chunks) == {}

    def test_case_insensitive(self) -> None:
        chunks = _make_regex_chunks(["ai"], case_sensitive=False)
        matches = trc.find_keyword_matches("AI is here", chunks)
        assert len(matches) == 1

    def test_case_sensitive_no_match(self) -> None:
        chunks = _make_regex_chunks(["ai"], case_sensitive=True)
        matches = trc.find_keyword_matches("AI is here", chunks)
        assert len(matches) == 0

    def test_whole_word(self) -> None:
        chunks = _make_regex_chunks(["art"], whole_word=True)
        # "art" should not match inside "artificial"
        assert trc.find_keyword_matches("artificial", chunks) == {}
        assert len(trc.find_keyword_matches("modern art", chunks)) == 1


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------

class TestProcessFile:
    def test_basic_processing(
        self, json_file: Path, rules_csv: Path
    ) -> None:
        kws = trc.load_keywords_csv(rules_csv, False)
        chunks = _make_regex_chunks(kws)
        ok, posts, comments, fb, rows, hits = trc.process_file(
            json_file, chunks
        )
        assert ok is True
        assert posts == 1
        assert comments == 2
        assert len(rows) > 0
        # "AGI" should match in post and first comment
        agi_rows = [r for r in rows if r["keyword"] == "AGI"]
        assert len(agi_rows) >= 2

    def test_missing_file(self, tmp_dir: Path) -> None:
        chunks = _make_regex_chunks(["AI"])
        ok, posts, comments, fb, rows, hits = trc.process_file(
            tmp_dir / "nope.json", chunks
        )
        assert ok is False
        assert rows == []

    def test_invalid_json(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.json"
        p.write_text("{bad", encoding="utf-8")
        chunks = _make_regex_chunks(["AI"])
        ok, _, _, _, rows, _ = trc.process_file(p, chunks)
        assert ok is False

    def test_post_without_comments(self, tmp_dir: Path) -> None:
        post = {
            "id": "p1",
            "subreddit": "test",
            "title": "Hello AI",
            "selftext": "",
        }
        p = tmp_dir / "nocomments.json"
        p.write_text(json.dumps(post), encoding="utf-8")
        chunks = _make_regex_chunks(["AI"])
        ok, posts, comments, _, rows, _ = trc.process_file(p, chunks)
        assert ok is True
        assert posts == 1
        assert comments == 0

    def test_fallback_comment_ids(self, tmp_dir: Path) -> None:
        post = {
            "id": "p1",
            "subreddit": "test",
            "title": "Hello AI",
            "comments": [
                {"body": "AI comment without id"},
            ],
        }
        p = tmp_dir / "noid.json"
        p.write_text(json.dumps(post), encoding="utf-8")
        chunks = _make_regex_chunks(["AI"])
        ok, _, _, fallback_count, rows, _ = trc.process_file(p, chunks)
        assert ok is True
        assert fallback_count == 1

    def test_rows_have_expected_fields(
        self, json_file: Path, rules_csv: Path
    ) -> None:
        kws = trc.load_keywords_csv(rules_csv, False)
        chunks = _make_regex_chunks(kws)
        _, _, _, _, rows, _ = trc.process_file(json_file, chunks)
        expected_fields = {
            "root_post_id",
            "subreddit",
            "content_type",
            "content_id",
            "author",
            "content_text",
            "created_at",
            "likes",
            "dislikes",
            "unvoted",
            "keyword_id",
            "keyword",
        }
        for row in rows:
            assert set(row.keys()) == expected_fields

    def test_content_types(
        self, json_file: Path, rules_csv: Path
    ) -> None:
        kws = trc.load_keywords_csv(rules_csv, False)
        chunks = _make_regex_chunks(kws)
        _, _, _, _, rows, _ = trc.process_file(json_file, chunks)
        types = {r["content_type"] for r in rows}
        assert "post" in types
        assert "comment" in types


# ---------------------------------------------------------------------------
# write_tagged_csv / write_phrases_csv
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_write_tagged_csv(self, tmp_dir: Path) -> None:
        rows = [
            {
                "root_post_id": "p1",
                "subreddit": "test",
                "content_type": "post",
                "content_id": "p1",
                "author": "alice",
                "content_text": "hello",
                "created_at": "2023-01-01T00:00:00Z",
                "likes": "5",
                "dislikes": "1",
                "unvoted": "",
                "keyword_id": "1",
                "keyword": "hello",
            }
        ]
        out = tmp_dir / "tagged.csv"
        trc.write_tagged_csv(out, rows)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "root_post_id" in text
        assert "hello" in text

    def test_write_phrases_csv(self, tmp_dir: Path) -> None:
        keywords = ["AI", "test"]
        kw_posts: Dict[int, Set[str]] = {1: {"p1", "p2"}}
        kw_contents: Dict[int, Set[str]] = {1: {"p1", "c1"}}
        out = tmp_dir / "phrases.csv"
        trc.write_phrases_csv(out, keywords, kw_posts, kw_contents)
        assert out.exists()
        with out.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == [
                "id", "keyword", "tagged_post_count", "tagged_content_count"
            ]
            row1 = next(reader)
            assert row1[1] == "AI"
            assert row1[2] == "2"  # 2 posts
            assert row1[3] == "2"  # 2 content items
            row2 = next(reader)
            assert row2[1] == "test"
            assert row2[2] == "0"

    def test_creates_parent_dirs(self, tmp_dir: Path) -> None:
        out = tmp_dir / "a" / "b" / "tagged.csv"
        trc.write_tagged_csv(out, [])
        assert out.exists()


# ---------------------------------------------------------------------------
# run (integration)
# ---------------------------------------------------------------------------

class TestRun:
    def _make_args(
        self,
        tmp_dir: Path,
        data_dir: str,
        rules: str,
        out_tagged: str | None = None,
        out_phrases: str | None = None,
        **kwargs: Any,
    ) -> argparse.Namespace:
        defaults = {
            "data_dir": data_dir,
            "rules": rules,
            "out_tagged": out_tagged,
            "out_phrases": out_phrases,
            "out_xlsx": str(tmp_dir / "results.xlsx"),
            "output_dir": None,
            "add_timestamp_prefix": False,
            "add_timestamp_suffix": False,
            "case_sensitive": False,
            "whole_word": False,
            "threads": 1,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_end_to_end(
        self,
        tmp_dir: Path,
        sample_post: Dict[str, Any],
    ) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "post.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        out_tagged = tmp_dir / "tagged.csv"
        out_phrases = tmp_dir / "phrases.csv"

        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            str(out_tagged),
            str(out_phrases),
        )
        code = trc.run(args)
        assert code == 0
        assert out_tagged.exists()
        assert out_phrases.exists()
        xlsx_path = tmp_dir / "results.xlsx"
        assert xlsx_path.exists()

        # Verify tagged CSV content
        with out_tagged.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 2  # post + at least 1 comment

        # Verify XLSX sheets
        import pandas as pd  # type: ignore
        sheets = pd.ExcelFile(xlsx_path).sheet_names
        assert "tagged_content" in sheets
        assert "phrases_found" in sheets
        assert "coding_rules" in sheets
        rules_df = pd.read_excel(xlsx_path, sheet_name="coding_rules")
        assert "keyword" in rules_df.columns
        assert len(rules_df) == 48

    def test_logs_output_file_paths(
        self,
        tmp_dir: Path,
        sample_post: Dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "post.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        out_tagged = tmp_dir / "tagged.csv"
        out_phrases = tmp_dir / "phrases.csv"
        out_xlsx = tmp_dir / "results.xlsx"

        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            str(out_tagged),
            str(out_phrases),
            out_xlsx=str(out_xlsx),
        )
        caplog.set_level(logging.INFO, logger=trc.LOG.name)

        code = trc.run(args)
        assert code == 0

        info_messages = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelno == logging.INFO
        ]
        assert any(
            msg == f"Tagged CSV written: {out_tagged}" for msg in info_messages
        )
        assert any(
            msg == f"Phrases CSV written: {out_phrases}" for msg in info_messages
        )
        assert any(
            msg == f"XLSX written: {out_xlsx}" for msg in info_messages
        )

    def test_missing_data_dir(self, tmp_dir: Path) -> None:
        args = self._make_args(
            tmp_dir,
            str(tmp_dir / "missing"),
            str(DEMO_RULES_CSV),
            str(tmp_dir / "t.csv"),
            str(tmp_dir / "p.csv"),
        )
        assert trc.run(args) == 1

    def test_no_json_files(self, tmp_dir: Path) -> None:
        data_dir = tmp_dir / "empty_data"
        data_dir.mkdir()
        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            str(tmp_dir / "t.csv"),
            str(tmp_dir / "p.csv"),
        )
        assert trc.run(args) == 1

    def test_invalid_threads(self, tmp_dir: Path) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "x.json").write_text("{}", encoding="utf-8")
        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            str(tmp_dir / "t.csv"),
            str(tmp_dir / "p.csv"),
            threads=0,
        )
        assert trc.run(args) == 1

    def test_multithreaded(
        self,
        tmp_dir: Path,
        sample_post: Dict[str, Any],
    ) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "a.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        (data_dir / "b.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        out_tagged = tmp_dir / "tagged.csv"
        out_phrases = tmp_dir / "phrases.csv"

        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            str(out_tagged),
            str(out_phrases),
            threads=2,
        )
        code = trc.run(args)
        assert code == 0

    def test_output_dir(
        self,
        tmp_dir: Path,
        sample_post: Dict[str, Any],
    ) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "post.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        out_dir = tmp_dir / "output"

        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            out_xlsx=None,
            output_dir=str(out_dir),
        )
        code = trc.run(args)
        assert code == 0
        assert (out_dir / "tagged_content.csv").exists()
        assert (out_dir / "phrases_found.csv").exists()
        assert (out_dir / "results.xlsx").exists()

    def test_output_dir_with_timestamp_prefix(
        self,
        tmp_dir: Path,
        sample_post: Dict[str, Any],
    ) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "post.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        out_dir = tmp_dir / "output"

        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            out_xlsx=None,
            output_dir=str(out_dir),
            add_timestamp_prefix=True,
        )
        code = trc.run(args)
        assert code == 0

        tagged = [p.name for p in out_dir.glob("*_tagged_content.csv")]
        assert len(tagged) == 1
        match = re.fullmatch(r"(\d{8}-\d{6})_tagged_content\.csv", tagged[0])
        assert match is not None
        timestamp = match.group(1)
        assert (out_dir / f"{timestamp}_phrases_found.csv").exists()
        assert (out_dir / f"{timestamp}_results.xlsx").exists()
        assert not (out_dir / "tagged_content.csv").exists()
        assert not (out_dir / "phrases_found.csv").exists()
        assert not (out_dir / "results.xlsx").exists()

    def test_custom_output_paths_with_timestamp_suffix(
        self,
        tmp_dir: Path,
        sample_post: Dict[str, Any],
    ) -> None:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / "post.json").write_text(
            json.dumps(sample_post), encoding="utf-8"
        )
        out_tagged = tmp_dir / "custom_tagged.csv"
        out_phrases = tmp_dir / "custom_phrases.csv"
        out_xlsx = tmp_dir / "custom_results.xlsx"

        args = self._make_args(
            tmp_dir,
            str(data_dir),
            str(DEMO_RULES_CSV),
            str(out_tagged),
            str(out_phrases),
            out_xlsx=str(out_xlsx),
            add_timestamp_suffix=True,
        )
        code = trc.run(args)
        assert code == 0

        tagged = [p.name for p in tmp_dir.glob("custom_tagged_*.csv")]
        assert len(tagged) == 1
        match = re.fullmatch(r"custom_tagged_(\d{8}-\d{6})\.csv", tagged[0])
        assert match is not None
        timestamp = match.group(1)
        assert (tmp_dir / f"custom_phrases_{timestamp}.csv").exists()
        assert (tmp_dir / f"custom_results_{timestamp}.xlsx").exists()
        assert not out_tagged.exists()
        assert not out_phrases.exists()
        assert not out_xlsx.exists()


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------

class TestBuildArgParser:
    def test_required_rules(self) -> None:
        parser = trc.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_minimal_args(self) -> None:
        parser = trc.build_arg_parser()
        args = parser.parse_args(["--rules", "rules.csv"])
        assert args.rules == "rules.csv"
        assert args.add_timestamp_prefix is False
        assert args.add_timestamp_suffix is False
        assert args.threads == 1
        assert args.case_sensitive is False
        assert args.whole_word is False

    def test_all_flags(self) -> None:
        parser = trc.build_arg_parser()
        args = parser.parse_args([
            "--rules", "r.csv",
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--out-tagged", "t.csv",
            "--out-phrases", "p.csv",
            "--out-xlsx", "o.xlsx",
            "--add-timestamp-prefix",
            "--case-sensitive",
            "--whole-word",
            "--threads", "4",
        ])
        assert args.data_dir == "/data"
        assert args.output_dir == "/out"
        assert args.out_tagged == "t.csv"
        assert args.out_phrases == "p.csv"
        assert args.out_xlsx == "o.xlsx"
        assert args.add_timestamp_prefix is True
        assert args.add_timestamp_suffix is False
        assert args.case_sensitive is True
        assert args.whole_word is True
        assert args.threads == 4

    def test_timestamp_flags_mutually_exclusive(self) -> None:
        parser = trc.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--rules", "r.csv",
                "--add-timestamp-prefix",
                "--add-timestamp-suffix",
            ])


# ---------------------------------------------------------------------------
# run_self_test
# ---------------------------------------------------------------------------

class TestSelfTest:
    def test_self_test_passes(self) -> None:
        trc.setup_logging()
        assert trc.run_self_test() == 0


# ---------------------------------------------------------------------------
# Edge cases / real-world formats
# ---------------------------------------------------------------------------

class TestRealWorldFormats:
    def test_published_string_timestamp(self, tmp_dir: Path) -> None:
        """Format seen in demo data: published as millis, comments
        as ISO strings."""
        post = {
            "title": "AI news",
            "author": "user",
            "published": 1770143683000,
            "url": "/r/singularity/comments/abc/title/",
            "comments": [
                {
                    "author": "commenter",
                    "text": "I like AI",
                    "published": "2026-02-03T18:40:02+00:00",
                    "score": {
                        "likes": None,
                        "dislikes": None,
                        "unvoted": None,
                    },
                    "comments": [],
                }
            ],
        }
        p = tmp_dir / "real.json"
        p.write_text(json.dumps(post), encoding="utf-8")
        chunks = _make_regex_chunks(["AI"])
        ok, posts, comments, _, rows, _ = trc.process_file(p, chunks)
        assert ok is True
        assert posts == 1
        assert comments >= 1
        # Check that both post and comment matched "AI"
        assert len(rows) >= 2

    def test_nested_comments(self, tmp_dir: Path) -> None:
        """Comments nested inside comments."""
        post = {
            "id": "p1",
            "subreddit": "test",
            "title": "Hello",
            "selftext": "AI world",
            "comments": [
                {
                    "id": "c1",
                    "body": "AI reply",
                    "comments": [
                        {
                            "id": "c2",
                            "body": "Nested AI reply",
                            "comments": [],
                        }
                    ],
                }
            ],
        }
        p = tmp_dir / "nested.json"
        p.write_text(json.dumps(post), encoding="utf-8")
        chunks = _make_regex_chunks(["AI"])
        ok, _, comments, _, rows, _ = trc.process_file(p, chunks)
        assert ok is True
        assert comments >= 2
        ai_rows = [r for r in rows if r["keyword"] == "AI"]
        # post + c1 + c2 = at least 3 matches
        assert len(ai_rows) >= 3

    def test_text_field_as_post_body(self, tmp_dir: Path) -> None:
        """Some formats use 'text' instead of 'selftext'."""
        post = {
            "title": "Title here",
            "text": "AI content in text field",
            "url": "/r/test/comments/x1/title/",
        }
        p = tmp_dir / "textfield.json"
        p.write_text(json.dumps(post), encoding="utf-8")
        chunks = _make_regex_chunks(["AI"])
        ok, _, _, _, rows, _ = trc.process_file(p, chunks)
        assert ok is True
        assert any(r["keyword"] == "AI" for r in rows)
