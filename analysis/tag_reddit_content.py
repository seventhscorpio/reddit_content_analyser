#!/usr/bin/env python
"""
Tag Reddit posts and comments by matching keyword phrases.

Input:
- JSON files in a directory (one file per post + its comments).
- coding_rules.csv with a column named "keyword" (UTF-8).

Output:
- tagged_content.csv with one row per matched phrase per content item.
- phrases_found.csv with per-phrase summary counts.

Expected JSON (flexible, heuristics-based):
- Post-like dicts usually include: title, selftext/text/body, url/permalink, author, published/created_utc, comments.
- Comment-like dicts usually include: body/text, author, published/created_utc, score, nested comments.
The parser walks the whole JSON and selects best candidates by heuristic scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast


LOG = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console logging used by the CLI pipeline.

    Workflow role:
        Called at startup by ``main`` (and self-test mode) so every pipeline
        stage emits uniform diagnostics.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_text_safe(path: Path) -> Optional[str]:
    """Read UTF-8 text from disk with Windows long-path fallback.

    Workflow role:
        First IO step for JSON ingestion; isolates filesystem quirks before
        parsing begins.

    Args:
        path: File path to read.

    Returns:
        File contents, or ``None`` when the file is missing or unreadable.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        if os.name == "nt":
            try:
                abs_path = str(path.resolve())
            except Exception:
                abs_path = str(path)
            if not abs_path.startswith("\\\\?\\"):
                abs_path = "\\\\?\\" + abs_path
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except (FileNotFoundError, OSError):
                return None
        return None


def load_json(path: Path) -> Optional[Any]:
    """Load JSON from disk while keeping per-file failures non-fatal.

    Workflow role:
        Entry point for per-file parsing in ``process_file``. Returning ``None``
        lets batch processing continue when one file is broken.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON value, or ``None`` when reading/parsing fails.
    """
    text = read_text_safe(path)
    if text is None:
        LOG.warning("Unreadable JSON file: %s", path.name)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        LOG.warning("Invalid JSON file: %s", path.name)
        return None


def _make_dialect(delimiter: str) -> csv.Dialect:
    """Build a CSV dialect instance for delimiter fallback.

    Workflow role:
        Helper used by rules ingestion when ``csv.Sniffer`` cannot infer format.

    Args:
        delimiter: Delimiter character to use.

    Returns:
        Dialect instance compatible with ``csv.reader``.
    """
    attrs = {
        "delimiter": delimiter,
        "quotechar": '"',
        "doublequote": True,
        "skipinitialspace": False,
        "lineterminator": "\n",
        "quoting": csv.QUOTE_MINIMAL,
    }
    dialect_type = type("CustomDialect", (csv.Dialect,), attrs)
    return cast(csv.Dialect, dialect_type())


def _detect_csv_dialect(sample: str) -> csv.Dialect:
    """Detect CSV dialect from a text sample.

    Workflow role:
        Keeps rules loading resilient to comma/semicolon/tab variants.

    Args:
        sample: Sample text from the rules file.

    Returns:
        Detected dialect. Falls back to semicolon/tab/excel heuristics.
    """
    try:
        return cast(csv.Dialect, csv.Sniffer().sniff(sample, delimiters=",;\t"))
    except csv.Error:
        if ";" in sample and "," not in sample:
            return _make_dialect(";")
        if "\t" in sample:
            return _make_dialect("\t")
        return cast(csv.Dialect, csv.get_dialect("excel"))


def load_keywords_csv(path: Path, case_sensitive: bool) -> List[str]:
    """Load cleaned keyword phrases from a rules CSV.

    Workflow role:
        Produces the canonical keyword list used to compile regex chunks and to
        preserve stable keyword IDs in outputs.

    Args:
        path: Path to ``coding_rules.csv``.
        case_sensitive: Whether duplicate detection is case-sensitive.

    Returns:
        Ordered list of unique keyword phrases.

    Raises:
        FileNotFoundError: If the rules file does not exist.
        ValueError: If the file is empty, missing ``keyword`` column, or has no
            valid keywords after cleaning.
    """
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    sample_text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not sample_text.strip():
        raise ValueError("Rules file is empty.")
    sample_lines = "\n".join(sample_text.splitlines()[:5])
    dialect = _detect_csv_dialect(sample_lines)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        rows = list(reader)
    if not rows:
        raise ValueError("Rules file is empty.")
    # Locate the canonical "keyword" column case-insensitively.
    header = rows[0]
    keyword_idx = None
    for idx, name in enumerate(header):
        if name.strip().lower() == "keyword":
            keyword_idx = idx
            break
    if keyword_idx is None:
        raise ValueError('Rules file must contain a "keyword" column.')

    # Preserve file order while deduplicating by matching mode.
    seen: Set[str] = set()
    keywords: List[str] = []
    for row in rows[1:]:
        if keyword_idx >= len(row):
            continue
        kw = row[keyword_idx].strip()
        if not kw:
            continue
        key = kw if case_sensitive else kw.casefold()
        if key in seen:
            LOG.warning("Duplicate keyword skipped: %s", kw)
            continue
        seen.add(key)
        keywords.append(kw)
    if not keywords:
        raise ValueError("No valid keywords after cleaning.")
    return keywords


def normalize_fullname(value: Any) -> str:
    """Convert Reddit fullname values (``t1_``/``t3_``) to raw IDs.

    Workflow role:
        Shared normalizer for post/comment ID extraction so downstream grouping
        uses stable identifiers.

    Args:
        value: Candidate fullname value.

    Returns:
        Raw identifier when possible, otherwise an empty string or original text.
    """
    if not isinstance(value, str):
        return ""
    if value.startswith("t1_") or value.startswith("t3_"):
        return value.split("_", 1)[1]
    return value


def to_iso_utc(
    value: Any,
    file_name: str,
    json_path: str,
    label: str | None = None,
    suppress_missing: bool = False,
) -> str:
    """Convert timestamps to normalized ISO-8601 UTC strings.

    Workflow role:
        Central timestamp normalizer used when building both post and comment
        output rows.

    Args:
        value: Timestamp value (epoch seconds, epoch milliseconds, or text).
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.
        label: Matched source field name.
        suppress_missing: Whether to suppress missing-value warnings.

    Returns:
        ISO-8601 timestamp string, passthrough text, or an empty string.
    """
    if value is None:
        if not suppress_missing:
            LOG.warning("Missing created_at in %s at %s", file_name, json_path)
        return ""
    if isinstance(value, (int, float)):
        ts = float(value)
        # Treat very large epoch values as milliseconds.
        if ts > 1e12:
            ts = ts / 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0)
        return dt.isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        if label == "published":
            return value
        LOG.warning("Non-epoch created_at in %s at %s", file_name, json_path)
        return value
    LOG.warning("Unknown created_at type in %s at %s", file_name, json_path)
    return ""


def get_value(d: Dict[str, Any], keys: Iterable[str]) -> Tuple[Any, Optional[str]]:
    """Return the first non-null value for candidate keys.

    Workflow role:
        Core lookup primitive for schema-flexible extraction across the parser.

    Args:
        d: Source dictionary.
        keys: Candidate keys in lookup priority order.

    Returns:
        Tuple of ``(value, key_name)``; ``(None, None)`` if nothing matched.
    """
    for key in keys:
        if key in d and d[key] is not None:
            return d[key], key
    return None, None


def get_string_field(
    d: Dict[str, Any],
    keys: Iterable[str],
    file_name: str,
    json_path: str,
    field_label: str,
    warn_missing: bool = True,
) -> str:
    """Extract a best-effort string field from flexible JSON structures.

    Workflow role:
        Normalizes heterogeneous author/title/body field shapes into consistent
        output-ready strings.

    Args:
        d: Source dictionary.
        keys: Candidate keys in lookup priority order.
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.
        field_label: Human-readable field label for warnings.
        warn_missing: Whether to log warnings for missing values.

    Returns:
        Extracted string value, or an empty string if not found.
    """
    value, key = get_value(d, keys)
    if value is None:
        if warn_missing:
            LOG.warning("Missing %s in %s at %s", field_label, file_name, json_path)
        return ""
    if isinstance(value, dict):
        nested_val, _ = get_value(value, ["name", "username", "id"])
        if nested_val is not None:
            value = nested_val
    if isinstance(value, str):
        return value
    LOG.warning("Non-string %s in %s at %s.%s", field_label, file_name, json_path, key)
    return str(value)


def extract_subreddit(
    d: Dict[str, Any], file_name: str, json_path: str
) -> str:
    """Extract subreddit name from direct fields or URL-like values.

    Workflow role:
        Supplies the ``subreddit`` column for tagged output rows, even when the
        source schema only exposes permalink-like fields.

    Args:
        d: Source dictionary.
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.

    Returns:
        Subreddit name or an empty string when unavailable.
    """
    value, _ = get_value(
        d,
        ["subreddit", "subreddit_name_prefixed",
         "community", "url", "permalink"],
    )
    if value is None:
        LOG.warning(
            "Missing subreddit in %s at %s", file_name, json_path
        )
        return ""
    if isinstance(value, str):
        if "/r/" in value:
            m = re.search(r"/r/([^/]+)(?:/|$)", value)
            if m:
                return m.group(1)
        if value.startswith("r/"):
            return value[2:]
        return value
    return str(value)


def extract_post_id(
    d: Dict[str, Any],
    file_name: str,
    json_path: str,
    fallback: str,
) -> str:
    """Extract a stable post ID from fields or Reddit permalink.

    Workflow role:
        Defines the root identifier used to group all comment matches under one
        post in both detailed and aggregate outputs.

    Args:
        d: Source dictionary.
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.
        fallback: Fallback ID (usually file stem).

    Returns:
        Post ID string.
    """
    id_keys = ["id", "post_id", "name", "full_id", "link_id"]
    value, _ = get_value(d, id_keys)
    post_id = normalize_fullname(value)
    if not post_id:
        # Recover IDs from canonical Reddit comment-thread URLs.
        url_value, _ = get_value(d, ["url", "permalink"])
        if isinstance(url_value, str):
            m = re.search(r"/comments/([^/]+)/", url_value)
            if m:
                post_id = m.group(1)
    if not post_id:
        LOG.warning(
            "Missing post_id in %s at %s; using file stem",
            file_name,
            json_path,
        )
        return fallback
    return post_id


def extract_comment_id(
    d: Dict[str, Any],
    file_name: str,
    json_path: str,
    fallback: str,
) -> Tuple[str, bool]:
    """Extract a comment ID and indicate whether fallback was used.

    Workflow role:
        Generates stable ``content_id`` values so comment-level matches can be
        counted and deduplicated reliably.

    Args:
        d: Source dictionary.
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.
        fallback: Generated fallback content ID.

    Returns:
        Tuple of ``(comment_id, used_fallback)``.
    """
    value, _ = get_value(d, ["id", "comment_id", "name", "full_id"])
    comment_id = normalize_fullname(value)
    if not comment_id:
        return fallback, True
    return comment_id, False


def extract_likes_dislikes(d: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract vote metrics from heterogeneous score payloads.

    Workflow role:
        Normalizes score-related fields into consistent output columns.

    Args:
        d: Source dictionary.

    Returns:
        Tuple of ``(likes, dislikes, unvoted)`` as strings.
    """
    score = d.get("score")
    if isinstance(score, dict):
        likes = score.get("likes")
        dislikes = score.get("dislikes")
        unvoted = score.get("unvoted")
        likes_str = "" if likes is None else str(likes)
        dislikes_str = "" if dislikes is None else str(dislikes)
        unvoted_str = "" if unvoted is None else str(unvoted)
        return likes_str, dislikes_str, unvoted_str
    likes, _ = get_value(d, ["ups", "upvotes", "likes"])
    dislikes, _ = get_value(d, ["downs", "downvotes", "dislikes"])
    likes_str = "" if likes is None else str(likes)
    dislikes_str = "" if dislikes is None else str(dislikes)
    return likes_str, dislikes_str, ""


def post_text(d: Dict[str, Any], file_name: str, json_path: str) -> str:
    """Build searchable post text from title/body fields.

    Workflow role:
        Creates the exact post text blob scanned by keyword matching.

    Args:
        d: Post-like dictionary.
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.

    Returns:
        Combined post text, prioritizing ``title + selftext``.
    """
    title = get_string_field(d, ["title"], file_name, json_path, "title")
    selftext = get_string_field(
        d,
        ["selftext", "selftext_html", "text", "body"],
        file_name,
        json_path,
        "selftext",
        warn_missing=False,
    )
    if title and selftext:
        # Prefer full context: title + selftext.
        return f"{title}\n\n{selftext}"
    if selftext:
        return selftext
    if title:
        return title
    LOG.warning("Missing post text in %s at %s", file_name, json_path)
    return ""


def comment_text(d: Dict[str, Any], file_name: str, json_path: str) -> str:
    """Extract searchable comment text from common body keys.

    Workflow role:
        Creates the exact comment text blob scanned by keyword matching.

    Args:
        d: Comment-like dictionary.
        file_name: Source JSON filename for diagnostics.
        json_path: JSON path for diagnostics.

    Returns:
        Comment text, or an empty string when unavailable.
    """
    body = get_string_field(d, ["body", "text", "comment"], file_name, json_path, "body")
    if not body:
        LOG.warning("Missing comment text in %s at %s", file_name, json_path)
    return body


def collect_nodes(
    obj: Any,
    path: str = "$",
    seen: Optional[Set[int]] = None,
) -> Iterable[Tuple[Dict[str, Any], str, Optional[str]]]:
    """Traverse JSON and yield dict nodes with path and optional kind hint.

    Workflow role:
        Structural discovery pass used before scoring to support arbitrary JSON
        layouts (including nested Reddit listing wrappers).

    Args:
        obj: Arbitrary JSON-compatible value.
        path: Current JSON path, used for diagnostics.
        seen: Set of visited object IDs to avoid recursion cycles.

    Yields:
        Tuples of ``(node_dict, json_path, kind_hint)``.
    """
    if seen is None:
        seen = set()
    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        kind = obj.get("kind")
        data = obj.get("data")
        if kind in ("t1", "t3") and isinstance(data, dict):
            # Reddit listings often store payload in `{"kind": "...", "data": ...}`.
            yield data, f"{path}.data", kind
            for item in collect_nodes(data, f"{path}.data", seen):
                yield item
            for k, v in obj.items():
                if k == "data":
                    continue
                for item in collect_nodes(v, f"{path}.{k}", seen):
                    yield item
            return
        yield obj, path, None
        for k, v in obj.items():
            for item in collect_nodes(v, f"{path}.{k}", seen):
                yield item
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            for item in collect_nodes(v, f"{path}[{i}]", seen):
                yield item


def score_post(d: Dict[str, Any], kind_hint: Optional[str]) -> int:
    """Score how likely a node is to represent a Reddit post.

    Workflow role:
        Heuristic ranking input for selecting one canonical post object per file.

    Args:
        d: Candidate dictionary node.
        kind_hint: Optional Reddit listing kind hint.

    Returns:
        Heuristic score where higher means more post-like.
    """
    score = 0
    if kind_hint == "t3":
        score += 5
    if "title" in d:
        score += 3
    if "selftext" in d or "selftext_html" in d or "text" in d:
        score += 2
    if "comments" in d and isinstance(d.get("comments"), list):
        score += 2
    if "is_self" in d:
        score += 1
    if "body" in d and "title" not in d:
        score -= 2
    return score


def score_comment(d: Dict[str, Any], kind_hint: Optional[str]) -> int:
    """Score how likely a node is to represent a Reddit comment.

    Workflow role:
        Heuristic filter for collecting comment candidates from noisy payloads.

    Args:
        d: Candidate dictionary node.
        kind_hint: Optional Reddit listing kind hint.

    Returns:
        Heuristic score where higher means more comment-like.
    """
    score = 0
    if kind_hint == "t1":
        score += 5
    if "body" in d or "text" in d:
        score += 3
    if "parent_id" in d or "link_id" in d:
        score += 2
    if "title" in d:
        score -= 2
    return score


def find_post_and_comments(
    obj: Any,
) -> Tuple[
    Optional[Tuple[Dict[str, Any], str]],
    List[Tuple[Dict[str, Any], str]],
]:
    """Find the best post candidate and all comment candidates in JSON data.

    Workflow role:
        Bridges structural traversal and text matching by deciding which nodes
        become the post plus comment stream for ``process_file``.

    Args:
        obj: Parsed JSON object from one source file.

    Returns:
        Tuple ``(post_info, comments)`` where ``post_info`` is
        ``(post_node, post_path)`` or ``None``, and ``comments`` is a list of
        ``(comment_node, comment_path)``.
    """
    candidates: List[Tuple[int, Dict[str, Any], str]] = []
    comments: List[Tuple[Dict[str, Any], str]] = []
    for node, path, kind_hint in collect_nodes(obj):
        pscore = score_post(node, kind_hint)
        cscore = score_comment(node, kind_hint)
        if pscore > 0:
            candidates.append((pscore, node, path))
        if cscore > 0:
            comments.append((node, path))
    candidates.sort(key=lambda x: x[0], reverse=True)
    post = (candidates[0][1], candidates[0][2]) if candidates else None
    if post:
        post_id = id(post[0])
        comments = [(n, p) for (n, p) in comments if id(n) != post_id]
    return post, comments


def build_phrase_pattern(phrase: str, whole_word: bool) -> str:
    """Build a regex-safe pattern for one keyword phrase.

    Workflow role:
        Converts one rule into a safe regex fragment used in chunk compilation.

    Args:
        phrase: Keyword phrase to escape.
        whole_word: Whether to enforce token boundaries when possible.

    Returns:
        Regex pattern string for the phrase.
    """
    escaped = re.escape(phrase)
    if whole_word and phrase and phrase[0].isalnum() and phrase[-1].isalnum():
        return r"(?<!\w)" + escaped + r"(?!\w)"
    return escaped


def build_regex_chunks(
    keywords: List[str],
    case_sensitive: bool,
    whole_word: bool,
    chunk_size: int = 500,
) -> List[Tuple[re.Pattern, List[Tuple[int, str]]]]:
    """Compile keyword regex chunks and preserve keyword-ID mapping.

    Workflow role:
        Precomputes matchers once per run so file processing reuses compiled
        patterns instead of recompiling per content item.

    Args:
        keywords: Ordered keyword list.
        case_sensitive: Whether matching is case-sensitive.
        whole_word: Whether to enforce word boundaries.
        chunk_size: Max keywords per compiled regex chunk.

    Returns:
        List of ``(compiled_regex, mapping)`` tuples. Mapping stores
        ``(keyword_id, keyword_text)`` in capturing-group order.
    """
    flags = re.UNICODE
    if not case_sensitive:
        flags |= re.IGNORECASE
    compiled: List[Tuple[re.Pattern, List[Tuple[int, str]]]] = []
    for i in range(0, len(keywords), chunk_size):
        chunk = keywords[i : i + chunk_size]
        mappings: List[Tuple[int, str]] = []
        parts: List[str] = []
        # Capturing-group order is used to map matches back to keyword IDs.
        for j, kw in enumerate(chunk, start=i + 1):
            mappings.append((j, kw))
            parts.append(f"({build_phrase_pattern(kw, whole_word)})")
        pattern = "|".join(parts)
        compiled.append((re.compile(pattern, flags), mappings))
    return compiled


def find_keyword_matches(
    text: str,
    regex_chunks: List[Tuple[re.Pattern, List[Tuple[int, str]]]],
) -> Dict[int, str]:
    """Find keyword matches for a single text value.

    Workflow role:
        Executes matching for one post/comment text and returns unique keyword
        hits for row emission and aggregate counting.

    Args:
        text: Content text to scan.
        regex_chunks: Compiled regex chunks with keyword mapping metadata.

    Returns:
        Mapping from keyword ID to keyword text for unique matches.
    """
    matched: Dict[int, str] = {}
    if not text:
        return matched
    for regex, mapping in regex_chunks:
        for m in regex.finditer(text):
            idx = m.lastindex
            if not idx:
                continue
            kw_id, kw = mapping[idx - 1]
            matched[kw_id] = kw
    return matched


def process_file(
    path: Path,
    regex_chunks: List[Tuple[re.Pattern, List[Tuple[int, str]]]],
) -> Tuple[
    bool,
    int,
    int,
    int,
    List[Dict[str, str]],
    Dict[int, Tuple[Set[str], Set[str]]],
]:
    """Process one JSON file into tagged rows and aggregate hit sets.

    Workflow role:
        Core per-file worker: parse JSON, detect post/comments, extract fields,
        match keywords, emit rows, and accumulate per-keyword hit sets.

    Args:
        path: JSON file path.
        regex_chunks: Compiled keyword regex chunks.

    Returns:
        Tuple ``(ok, post_count, comment_count, fallback_comment_count, rows, keyword_hits)``.
        ``keyword_hits`` maps keyword IDs to ``(post_ids, content_ids)`` sets.
    """
    data = load_json(path)
    if data is None:
        return False, 0, 0, 0, [], {}

    # Identify post and comment structures within the JSON.
    post_info, comments = find_post_and_comments(data)
    post_node = post_info[0] if post_info else None
    post_path = post_info[1] if post_info else "$"

    root_post_id = ""
    subreddit = ""
    if post_node:
        # Use post metadata for shared fields and root id.
        root_post_id = extract_post_id(post_node, path.name, post_path, path.stem)
        subreddit = extract_subreddit(post_node, path.name, post_path)
    else:
        LOG.warning("No post object detected in %s; using file stem as root_post_id", path.name)
        root_post_id = path.stem

    tagged_rows: List[Dict[str, str]] = []
    keyword_hits: Dict[int, Tuple[Set[str], Set[str]]] = {}

    def record_hit(kw_id: int, content_id: str) -> None:
        """Track aggregate hit sets used later for phrases summary output."""
        if kw_id not in keyword_hits:
            keyword_hits[kw_id] = (set(), set())
        keyword_hits[kw_id][0].add(root_post_id)
        keyword_hits[kw_id][1].add(content_id)

    post_count = 1 if post_node else 0
    comment_count = 0
    fallback_comment_ids = 0

    if post_node:
        # Tag the post text itself.
        content_id = root_post_id
        author = get_string_field(
            post_node,
            ["author", "user", "username"],
            path.name,
            post_path,
            "author",
            warn_missing=False,
        )
        created_raw, created_key = get_value(
            post_node,
            ["created_utc", "created", "created_at",
             "createdAt", "date", "timestamp", "published"],
        )
        created_at = to_iso_utc(
            created_raw, path.name, post_path, created_key
        )
        likes, dislikes, unvoted = extract_likes_dislikes(post_node)
        text = post_text(post_node, path.name, post_path)
        matches = find_keyword_matches(text, regex_chunks)
        for kw_id, kw in matches.items():
            tagged_rows.append(
                {
                    "root_post_id": root_post_id,
                    "subreddit": subreddit,
                    "content_type": "post",
                    "content_id": content_id,
                    "author": author,
                    "content_text": text,
                    "created_at": created_at,
                    "likes": likes,
                    "dislikes": dislikes,
                    "unvoted": unvoted,
                    "keyword_id": str(kw_id),
                    "keyword": kw,
                }
            )
            record_hit(kw_id, content_id)

    for idx, (comment, cpath) in enumerate(comments, start=1):
        # Tag each comment independently, with fallback ids when missing.
        comment_count += 1
        fallback_id = f"{root_post_id}_comment_{idx}"
        content_id, used_fallback = extract_comment_id(
            comment, path.name, cpath, fallback_id
        )
        if used_fallback:
            fallback_comment_ids += 1
        author = get_string_field(
            comment,
            ["author", "user", "username"],
            path.name,
            cpath,
            "author",
            warn_missing=False,
        )
        created_raw, created_key = get_value(
            comment,
            ["created_utc", "created", "created_at",
             "createdAt", "date", "timestamp", "published"],
        )
        created_at = to_iso_utc(
            created_raw, path.name, cpath, created_key,
            suppress_missing=True,
        )
        likes, dislikes, unvoted = extract_likes_dislikes(comment)
        text = comment_text(comment, path.name, cpath)
        matches = find_keyword_matches(text, regex_chunks)
        for kw_id, kw in matches.items():
            tagged_rows.append(
                {
                    "root_post_id": root_post_id,
                    "subreddit": subreddit,
                    "content_type": "comment",
                    "content_id": content_id,
                    "author": author,
                    "content_text": text,
                    "created_at": created_at,
                    "likes": likes,
                    "dislikes": dislikes,
                    "unvoted": unvoted,
                    "keyword_id": str(kw_id),
                    "keyword": kw,
                }
            )
            record_hit(kw_id, content_id)

    return True, post_count, comment_count, fallback_comment_ids, tagged_rows, keyword_hits


def write_tagged_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write the detailed match table to ``tagged_content.csv`` format.

    Workflow role:
        Persists row-level match details consumed by downstream analysis.

    Args:
        path: Output CSV path.
        rows: Tagged content rows, one per ``(content, keyword)`` match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_phrases_csv(
    path: Path,
    keywords: List[str],
    keyword_posts: Dict[int, Set[str]],
    keyword_contents: Dict[int, Set[str]],
) -> None:
    """Write phrase-level summary metrics to ``phrases_found.csv`` format.

    Workflow role:
        Persists per-keyword aggregates derived from merged per-file hit sets.

    Args:
        path: Output CSV path.
        keywords: Ordered keyword list.
        keyword_posts: Keyword ID to set of matched root post IDs.
        keyword_contents: Keyword ID to set of matched content IDs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "keyword", "tagged_post_count", "tagged_content_count"])
        for idx, kw in enumerate(keywords, start=1):
            post_count = len(keyword_posts.get(idx, set()))
            content_count = len(keyword_contents.get(idx, set()))
            writer.writerow([idx, kw, post_count, content_count])


def _read_rules_csv(rules_path: Path) -> list:
    """Read rules CSV rows for XLSX export without dropping columns.

    Workflow role:
        Supports report traceability by embedding original rules rows in XLSX.

    Args:
        rules_path: Path to rules CSV.

    Returns:
        List of raw row dictionaries as read by ``csv.DictReader``.
    """
    sample_text = rules_path.read_text(
        encoding="utf-8-sig", errors="replace"
    )
    sample_lines = "\n".join(sample_text.splitlines()[:5])
    dialect = _detect_csv_dialect(sample_lines)
    with rules_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def write_xlsx(
    path: Path,
    tagged_rows: List[Dict[str, str]],
    keywords: List[str],
    keyword_posts: Dict[int, Set[str]],
    keyword_contents: Dict[int, Set[str]],
    rules_path: Path,
    run_metadata: Dict[str, str],
) -> None:
    """Write multi-sheet XLSX output for detailed and summary reporting.

    Workflow role:
        Produces a single report artifact combining detailed rows, aggregates,
        source rules, and run metadata.

    Args:
        path: Output XLSX path.
        tagged_rows: Detailed tagged content rows.
        keywords: Ordered keyword list.
        keyword_posts: Keyword ID to set of matched root post IDs.
        keyword_contents: Keyword ID to set of matched content IDs.
        rules_path: Source rules CSV path.
        run_metadata: Runtime metadata values for the metadata sheet.

    Raises:
        RuntimeError: If pandas cannot be imported for XLSX writing.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pandas is required to write XLSX output."
        ) from exc

    # Build phrase summary rows in deterministic keyword-ID order.
    phrases_rows: List[Dict[str, Any]] = []
    for idx, kw in enumerate(keywords, start=1):
        phrases_rows.append(
            {
                "id": idx,
                "keyword": kw,
                "tagged_post_count": len(
                    keyword_posts.get(idx, set())
                ),
                "tagged_content_count": len(
                    keyword_contents.get(idx, set())
                ),
            }
        )

    rules_rows = _read_rules_csv(rules_path)
    metadata_rows = [
        {"key": "command", "value": run_metadata.get("command", "")},
        {
            "key": "working_directory",
            "value": run_metadata.get("working_directory", ""),
        },
        {"key": "run_timestamp", "value": run_metadata.get("run_timestamp", "")},
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep sheet names stable; the self-test validates this contract.
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(tagged_rows).to_excel(
            writer, sheet_name="tagged_content", index=False
        )
        pd.DataFrame(phrases_rows).to_excel(
            writer, sheet_name="phrases_found", index=False
        )
        pd.DataFrame(rules_rows).to_excel(
            writer, sheet_name="coding_rules", index=False
        )
        pd.DataFrame(metadata_rows, columns=["key", "value"]).to_excel(
            writer, sheet_name="metadata", index=False
        )


def format_command_line(argv: List[str]) -> str:
    """Format argv as a shell-like command string for metadata.

    Workflow role:
        Captures reproducibility metadata for the XLSX ``metadata`` sheet.

    Args:
        argv: Raw argument vector.

    Returns:
        Platform-appropriate command-line string.
    """
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    try:
        import shlex

        return shlex.join(argv)
    except Exception:
        return " ".join(argv)


def with_timestamp_affix(
    path: Path,
    timestamp: str,
    add_prefix: bool,
    add_suffix: bool,
) -> Path:
    """Apply optional timestamp prefix/suffix naming to a file path.

    Workflow role:
        Implements output naming policy so multiple runs can coexist cleanly.

    Args:
        path: Base output path.
        timestamp: Timestamp token in ``YYYYMMDD-HHMMSS`` format.
        add_prefix: Whether to prepend timestamp to filename.
        add_suffix: Whether to append timestamp to filename stem.

    Returns:
        Path with adjusted filename.

    Raises:
        ValueError: If both prefix and suffix flags are set.
    """
    if add_prefix and add_suffix:
        raise ValueError("Timestamp prefix and suffix are mutually exclusive.")
    if add_prefix:
        return path.with_name(f"{timestamp}_{path.name}")
    if add_suffix:
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    return path


def run(args: argparse.Namespace) -> int:
    """Execute the end-to-end tagging pipeline for parsed CLI args.

    Workflow role:
        Orchestrator for the full run lifecycle: validate inputs, compile
        patterns, process files, merge aggregates, and write outputs.

    Args:
        args: Parsed command-line namespace.

    Returns:
        Process exit code (``0`` on success, ``1`` on failure).
    """
    data_dir = Path(args.data_dir).resolve()
    rules_path = Path(args.rules)
    run_metadata = {
        "command": format_command_line(sys.argv),
        "working_directory": str(Path.cwd()),
        "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    output_dir_value = getattr(args, "output_dir", None)
    output_dir = Path(output_dir_value) if output_dir_value else None
    # Explicit output paths override output-dir defaults.
    out_tagged = Path(args.out_tagged) if args.out_tagged else (
        (output_dir / "tagged_content.csv") if output_dir else Path("tagged_content.csv")
    )
    out_phrases = Path(args.out_phrases) if args.out_phrases else (
        (output_dir / "phrases_found.csv") if output_dir else Path("phrases_found.csv")
    )
    out_xlsx_value = getattr(args, "out_xlsx", None)
    out_xlsx = Path(out_xlsx_value) if out_xlsx_value else (
        (output_dir / "results.xlsx") if output_dir
        else Path("results.xlsx")
    )
    add_timestamp_prefix = bool(getattr(args, "add_timestamp_prefix", False))
    add_timestamp_suffix = bool(getattr(args, "add_timestamp_suffix", False))
    if add_timestamp_prefix and add_timestamp_suffix:
        LOG.error(
            "--add-timestamp-prefix and --add-timestamp-suffix are mutually exclusive."
        )
        return 1
    if add_timestamp_prefix or add_timestamp_suffix:
        # One shared timestamp keeps all output filenames aligned.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_tagged = with_timestamp_affix(
            out_tagged, timestamp, add_timestamp_prefix, add_timestamp_suffix
        )
        out_phrases = with_timestamp_affix(
            out_phrases, timestamp, add_timestamp_prefix, add_timestamp_suffix
        )
        out_xlsx = with_timestamp_affix(
            out_xlsx, timestamp, add_timestamp_prefix, add_timestamp_suffix
        )

    LOG.info("Data directory: %s", data_dir)
    if not data_dir.is_dir():
        LOG.error("Data directory does not exist: %s", data_dir)
        return 1
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        LOG.error("No .json files found in: %s", data_dir)
        return 1
    if args.threads < 1:
        LOG.error("--threads must be >= 1")
        return 1

    # Compile regex chunks once, then reuse for every file.
    keywords = load_keywords_csv(rules_path, args.case_sensitive)
    regex_chunks = build_regex_chunks(
        keywords, case_sensitive=args.case_sensitive, whole_word=args.whole_word
    )

    # Aggregate outputs across files.
    all_rows: List[Dict[str, str]] = []
    keyword_posts: Dict[int, Set[str]] = {}
    keyword_contents: Dict[int, Set[str]] = {}
    files_ok = 0
    posts_total = 0
    comments_total = 0
    fallback_comments_total = 0

    if args.threads == 1:
        # Sequential processing.
        for path in json_files:
            ok, post_count, comment_count, fallback_comments, rows, hits = process_file(
                path, regex_chunks
            )
            if ok:
                files_ok += 1
            posts_total += post_count
            comments_total += comment_count
            fallback_comments_total += fallback_comments
            all_rows.extend(rows)
            for kw_id, (post_ids, content_ids) in hits.items():
                keyword_posts.setdefault(kw_id, set()).update(post_ids)
                keyword_contents.setdefault(kw_id, set()).update(content_ids)
    else:
        # Parallel I/O-safe processing.
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = {ex.submit(process_file, path, regex_chunks): path for path in json_files}
            for fut in as_completed(futures):
                try:
                    ok, post_count, comment_count, fallback_comments, rows, hits = fut.result()
                except Exception as exc:
                    LOG.warning("Worker failed for %s: %s", futures[fut].name, exc)
                    continue
                if ok:
                    files_ok += 1
                posts_total += post_count
                comments_total += comment_count
                fallback_comments_total += fallback_comments
                all_rows.extend(rows)
                for kw_id, (post_ids, content_ids) in hits.items():
                    keyword_posts.setdefault(kw_id, set()).update(post_ids)
                    keyword_contents.setdefault(kw_id, set()).update(content_ids)

    write_tagged_csv(out_tagged, all_rows)
    LOG.info("Tagged CSV written: %s", out_tagged)
    write_phrases_csv(out_phrases, keywords, keyword_posts, keyword_contents)
    LOG.info("Phrases CSV written: %s", out_phrases)
    try:
        write_xlsx(
            out_xlsx, all_rows, keywords,
            keyword_posts, keyword_contents,
            rules_path,
            run_metadata,
        )
        LOG.info("XLSX written: %s", out_xlsx)
    except Exception as exc:
        LOG.error("Failed to write XLSX: %s", exc)
        return 1

    found_phrases = sum(1 for idx in range(1, len(keywords) + 1) if keyword_posts.get(idx))

    LOG.info("Files total: %d", len(json_files))
    LOG.info("Files processed: %d", files_ok)
    LOG.info("Posts found: %d", posts_total)
    LOG.info("Comments found: %d", comments_total)
    LOG.info("Comments with fallback id: %d", fallback_comments_total)
    LOG.info("Tagged rows: %d", len(all_rows))
    LOG.info("Phrases matched: %d", found_phrases)
    return 0


def run_self_test() -> int:
    """Run an integration self-test on temporary synthetic data.

    Workflow role:
        Validates the full production pipeline contract (outputs and schema)
        using a deterministic in-memory fixture written to temp files.

    Returns:
        Process exit code (``0`` on success, ``1`` on failure).
    """
    LOG.info("Running self-test...")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_dir = base / "demo_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        rules_path = base / "coding_rules.csv"
        out_tagged = base / "tagged_content.csv"
        out_phrases = base / "phrases_found.csv"
        out_xlsx = base / "results.xlsx"

        # Minimal fixture with one post and two comments; only two texts mention "AI".
        sample = {
            "id": "abc123",
            "subreddit": "testsub",
            "author": "alice",
            "created_utc": 1700000000,
            "title": "Hello World",
            "selftext": "This is a test post about AI.",
            "comments": [
                {"id": "c1", "author": "bob", "created_utc": 1700000001, "body": "AI is here."},
                {"id": "c2", "author": "eve", "created_utc": 1700000002, "body": "Nothing to see."},
            ],
        }
        (data_dir / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
        rules_path.write_text("keyword\nAI\n test \nAI\n", encoding="utf-8")

        # Reuse the normal runtime pipeline with temporary paths.
        args = argparse.Namespace(
            data_dir=str(data_dir),
            rules=str(rules_path),
            out_tagged=str(out_tagged),
            out_phrases=str(out_phrases),
            out_xlsx=str(out_xlsx),
            add_timestamp_prefix=False,
            add_timestamp_suffix=False,
            case_sensitive=False,
            whole_word=False,
            threads=1,
        )
        code = run(args)
        if code != 0:
            LOG.error("Self-test pipeline failed with code %d", code)
            return code

        # Verify all declared output artifacts are present.
        for output_path in (out_tagged, out_phrases, out_xlsx):
            if not output_path.exists():
                LOG.error("Self-test missing output: %s", output_path)
                return 1

        with out_tagged.open("r", encoding="utf-8", newline="") as f:
            tagged_rows = list(csv.DictReader(f))
        if len(tagged_rows) < 2:
            LOG.error(
                "Self-test expected at least 2 tagged rows, got %d",
                len(tagged_rows),
            )
            return 1
        if not any(row.get("keyword") == "AI" for row in tagged_rows):
            LOG.error("Self-test expected keyword 'AI' in tagged_content.csv")
            return 1

        with out_phrases.open("r", encoding="utf-8", newline="") as f:
            phrases_rows = list(csv.DictReader(f))
        ai_rows = [row for row in phrases_rows if row.get("keyword") == "AI"]
        if not ai_rows:
            LOG.error("Self-test expected keyword 'AI' in phrases_found.csv")
            return 1
        ai_row = ai_rows[0]
        if ai_row.get("tagged_post_count") != "1":
            LOG.error(
                "Self-test expected AI tagged_post_count=1, got %s",
                ai_row.get("tagged_post_count"),
            )
            return 1
        if ai_row.get("tagged_content_count") != "2":
            LOG.error(
                "Self-test expected AI tagged_content_count=2, got %s",
                ai_row.get("tagged_content_count"),
            )
            return 1

        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            LOG.error(
                "Self-test failed importing pandas for XLSX validation: %s",
                exc,
            )
            return 1

        # Confirm report sheet contract used by downstream users.
        sheets: set[str] = {
            str(sheet_name)
            for sheet_name in pd.ExcelFile(out_xlsx).sheet_names
        }
        required_sheets = {
            "tagged_content",
            "phrases_found",
            "coding_rules",
            "metadata",
        }
        missing_sheets = sorted(required_sheets - sheets)
        if missing_sheets:
            LOG.error(
                "Self-test missing XLSX sheets: %s",
                ", ".join(missing_sheets),
            )
            return 1

        metadata_df = pd.read_excel(out_xlsx, sheet_name="metadata")
        if list(metadata_df.columns) != ["key", "value"]:
            LOG.error(
                "Self-test expected metadata columns ['key', 'value'], got %s",
                list(metadata_df.columns),
            )
            return 1
        metadata = {
            str(row["key"]): str(row["value"])
            for _, row in metadata_df.iterrows()
        }
        for key in ("command", "working_directory", "run_timestamp"):
            if not metadata.get(key):
                LOG.error("Self-test metadata key missing or empty: %s", key)
                return 1
        try:
            run_dt = datetime.fromisoformat(metadata["run_timestamp"])
        except ValueError:
            LOG.error(
                "Self-test run_timestamp is not valid ISO format: %s",
                metadata["run_timestamp"],
            )
            return 1
        if run_dt.tzinfo is None:
            LOG.error(
                "Self-test run_timestamp missing timezone: %s",
                metadata["run_timestamp"],
            )
            return 1

        LOG.info("Self-test outputs: %s, %s", out_tagged, out_phrases)
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Workflow role:
        Defines the external interface consumed by ``main`` before dispatching
        into ``run``.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Tag Reddit posts/comments by matching keyword phrases."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory with JSON files.",
    )
    parser.add_argument("--rules", required=True, help="Path to coding_rules.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Base directory for default output files. Used for files not explicitly set "
            "via --out-tagged / --out-phrases / --out-xlsx."
        ),
    )
    parser.add_argument(
        "--out-tagged",
        default=None,
        help=(
            "Output tagged_content.csv path. Default: tagged_content.csv "
            "(or <output-dir>/tagged_content.csv when --output-dir is set)."
        ),
    )
    parser.add_argument(
        "--out-phrases",
        default=None,
        help=(
            "Output phrases_found.csv path. Default: phrases_found.csv "
            "(or <output-dir>/phrases_found.csv when --output-dir is set)."
        ),
    )
    parser.add_argument(
        "--out-xlsx",
        default=None,
        help=(
            "Output XLSX path. Default: results.xlsx "
            "(or <output-dir>/results.xlsx when --output-dir "
            "is set)."
        ),
    )
    timestamp_group = parser.add_mutually_exclusive_group()
    timestamp_group.add_argument(
        "--add-timestamp-prefix",
        action="store_true",
        help=(
            "Prefix output filenames with current timestamp: "
            "YYYYMMDD-HHMMSS_filename.ext."
        ),
    )
    timestamp_group.add_argument(
        "--add-timestamp-suffix",
        action="store_true",
        help=(
            "Suffix output filenames with current timestamp: "
            "filename_YYYYMMDD-HHMMSS.ext."
        ),
    )
    parser.add_argument("--case-sensitive", action="store_true", help="Use case-sensitive matching.")
    parser.add_argument("--whole-word", action="store_true", help="Respect word boundaries.")
    parser.add_argument("--threads", type=int, default=1, help="Number of worker threads.")
    parser.add_argument("--self-test", action="store_true", help="Run a minimal self-test.")
    return parser


def main() -> int:
    """Parse CLI args, initialize logging, and run the selected mode.

    Workflow role:
        Thin entrypoint that routes either to ``run_self_test`` or the normal
        ``run`` pipeline.

    Returns:
        Process exit code.
    """
    if "--self-test" in sys.argv:
        setup_logging()
        return run_self_test()
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging()
    return run(args)


# Workflow (high-level):
# 1) Load keywords and build regex chunks.
# 2) For each JSON file, detect post + comments heuristically.
# 3) Match phrases in each content text and emit rows per match.
# 4) Aggregate per-phrase counts and write CSV/XLSX outputs.


if __name__ == "__main__":
    raise SystemExit(main())
