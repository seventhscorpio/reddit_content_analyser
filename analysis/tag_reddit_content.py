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
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast


LOG = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure basic console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_text_safe(path: Path) -> Optional[str]:
    """Read text safely, including Windows long-path handling."""
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
    """Load JSON with safe text handling and warnings on failure."""
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
    """Create a csv.Dialect instance for a given delimiter."""
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
    """Detect CSV dialect from a sample, with safe fallbacks."""
    try:
        return cast(csv.Dialect, csv.Sniffer().sniff(sample, delimiters=",;\t"))
    except csv.Error:
        if ";" in sample and "," not in sample:
            return _make_dialect(";")
        if "\t" in sample:
            return _make_dialect("\t")
        return cast(csv.Dialect, csv.get_dialect("excel"))


def load_keywords_csv(path: Path, case_sensitive: bool) -> List[str]:
    """Load and clean keywords from CSV (dedupe, trim, skip empty)."""
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
    header = rows[0]
    keyword_idx = None
    for idx, name in enumerate(header):
        if name.strip().lower() == "keyword":
            keyword_idx = idx
            break
    if keyword_idx is None:
        raise ValueError('Rules file must contain a "keyword" column.')

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
    """Normalize Reddit fullname (t1_/t3_) to raw id."""
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
    """Convert timestamps to ISO-8601 UTC or pass through strings."""
    if value is None:
        if not suppress_missing:
            LOG.warning("Missing created_at in %s at %s", file_name, json_path)
        return ""
    if isinstance(value, (int, float)):
        ts = float(value)
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
    """Return the first non-null value and its key from a dict."""
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
    """Fetch a string field with optional missing-field warning."""
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


def extract_subreddit(d: Dict[str, Any], file_name: str, json_path: str) -> str:
    """Extract subreddit name from fields or URL/permalink."""
    value, _ = get_value(d, ["subreddit", "subreddit_name_prefixed", "community", "url", "permalink"])
    if value is None:
        LOG.warning("Missing subreddit in %s at %s", file_name, json_path)
        return ""
    if isinstance(value, str):
        if "/r/" in value:
            m = re.search(r"/r/([^/]+)/", value)
            if m:
                return m.group(1)
        if value.startswith("r/"):
            return value[2:]
        return value
    return str(value)


def extract_post_id(d: Dict[str, Any], file_name: str, json_path: str, fallback: str) -> str:
    """Extract post id from fields or URL/permalink, with fallback."""
    value, _ = get_value(d, ["id", "post_id", "name", "full_id", "link_id", "url", "permalink"])
    post_id = normalize_fullname(value)
    if not post_id and isinstance(value, str):
        m = re.search(r"/comments/([^/]+)/", value)
        if m:
            post_id = m.group(1)
    if not post_id:
        LOG.warning("Missing post_id in %s at %s; using file stem", file_name, json_path)
        return fallback
    return post_id


def extract_comment_id(
    d: Dict[str, Any],
    file_name: str,
    json_path: str,
    fallback: str,
) -> Tuple[str, bool]:
    """Extract comment id from fields, returning (id, used_fallback)."""
    value, _ = get_value(d, ["id", "comment_id", "name", "full_id"])
    comment_id = normalize_fullname(value)
    if not comment_id:
        return fallback, True
    return comment_id, False


def extract_likes_dislikes(d: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract likes/dislikes/unvoted from score-like fields."""
    if isinstance(d.get("score"), dict):
        score = d.get("score") or {}
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
    """Build post text from title/selftext with safe fallbacks."""
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
    """Extract comment text from common fields."""
    body = get_string_field(d, ["body", "text", "comment"], file_name, json_path, "body")
    if not body:
        LOG.warning("Missing comment text in %s at %s", file_name, json_path)
    return body


def collect_nodes(
    obj: Any,
    path: str = "$",
    seen: Optional[Set[int]] = None,
) -> Iterable[Tuple[Dict[str, Any], str, Optional[str]]]:
    """Walk arbitrary JSON structures and yield dict nodes with paths."""
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
    """Heuristically score a dict as a post candidate."""
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
    """Heuristically score a dict as a comment candidate."""
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


def find_post_and_comments(obj: Any) -> Tuple[Optional[Tuple[Dict[str, Any], str]], List[Tuple[Dict[str, Any], str]]]:
    """Select the best post candidate and collect comment candidates."""
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
    """Build a regex pattern for a phrase with optional word boundaries."""
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
    """Compile keywords into regex chunks to avoid huge patterns."""
    flags = re.UNICODE
    if not case_sensitive:
        flags |= re.IGNORECASE
    compiled: List[Tuple[re.Pattern, List[Tuple[int, str]]]] = []
    for i in range(0, len(keywords), chunk_size):
        chunk = keywords[i : i + chunk_size]
        mappings: List[Tuple[int, str]] = []
        parts: List[str] = []
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
    """Find matched keyword ids for a single text."""
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
    """Process one JSON file and return tagged rows plus per-keyword hit sets."""
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

    def record_hit(kw_id: int, kw: str, content_id: str) -> None:
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
            ["created_utc", "created", "created_at", "createdAt", "date", "timestamp", "published"],
        )
        created_at = to_iso_utc(created_raw, path.name, post_path, created_key)
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
            record_hit(kw_id, kw, content_id)

    for idx, (comment, cpath) in enumerate(comments, start=1):
        # Tag each comment independently, with fallback ids when missing.
        comment_count += 1
        fallback_id = f"{root_post_id}_comment_{idx}"
        content_id, used_fallback = extract_comment_id(comment, path.name, cpath, fallback_id)
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
            ["created_utc", "created", "created_at", "createdAt", "date", "timestamp", "published"],
        )
        created_at = to_iso_utc(created_raw, path.name, cpath, created_key, suppress_missing=True)
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
            record_hit(kw_id, kw, content_id)

    return True, post_count, comment_count, fallback_comment_ids, tagged_rows, keyword_hits


def write_tagged_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write tagged content rows to CSV."""
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
    """Write per-keyword aggregate counts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "keyword", "tagged_post_count", "tagged_content_count"])
        for idx, kw in enumerate(keywords, start=1):
            post_count = len(keyword_posts.get(idx, set()))
            content_count = len(keyword_contents.get(idx, set()))
            writer.writerow([idx, kw, post_count, content_count])


def write_xlsx(
    path: Path,
    tagged_rows: List[Dict[str, str]],
    keywords: List[str],
    keyword_posts: Dict[int, Set[str]],
    keyword_contents: Dict[int, Set[str]],
) -> None:
    """Write tagged content and phrases to a single XLSX workbook."""
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to write XLSX output.") from exc

    phrases_rows: List[Dict[str, Any]] = []
    for idx, kw in enumerate(keywords, start=1):
        phrases_rows.append(
            {
                "id": idx,
                "keyword": kw,
                "tagged_post_count": len(keyword_posts.get(idx, set())),
                "tagged_content_count": len(keyword_contents.get(idx, set())),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(tagged_rows).to_excel(writer, sheet_name="tagged_content", index=False)
        pd.DataFrame(phrases_rows).to_excel(writer, sheet_name="phrases_found", index=False)


def resolve_default_data_dir() -> Path:
    """Resolve default data directory relative to this script."""
    # Default relative to this script's location: ..\Data\Demo data
    return (Path(__file__).parent / ".." / "Data" / "Demo data").resolve()


def run(args: argparse.Namespace) -> int:
    """Main processing pipeline."""
    data_dir = Path(args.data_dir).resolve() if args.data_dir else resolve_default_data_dir()
    rules_path = Path(args.rules)
    out_tagged = Path(args.out_tagged)
    out_phrases = Path(args.out_phrases)
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else None

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
    write_phrases_csv(out_phrases, keywords, keyword_posts, keyword_contents)
    if out_xlsx:
        try:
            write_xlsx(out_xlsx, all_rows, keywords, keyword_posts, keyword_contents)
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
    """Run a minimal self-test with synthetic data."""
    LOG.info("Running self-test...")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_dir = base / "demo_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        rules_path = base / "coding_rules.csv"
        out_tagged = base / "tagged_content.csv"
        out_phrases = base / "phrases_found.csv"

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

        args = argparse.Namespace(
            data_dir=str(data_dir),
            rules=str(rules_path),
            out_tagged=str(out_tagged),
            out_phrases=str(out_phrases),
            case_sensitive=False,
            whole_word=False,
            threads=1,
        )
        code = run(args)
        LOG.info("Self-test outputs: %s, %s", out_tagged, out_phrases)
        return code


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Tag Reddit posts/comments by matching keyword phrases."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with JSON files (default: ..\\Data\\Demo data relative to script).",
    )
    parser.add_argument("--rules", required=True, help="Path to coding_rules.csv.")
    parser.add_argument("--out-tagged", required=True, help="Output tagged_content.csv.")
    parser.add_argument("--out-phrases", required=True, help="Output phrases_found.csv.")
    parser.add_argument(
        "--out-xlsx",
        help="Optional XLSX output containing tagged_content and phrases_found sheets.",
    )
    parser.add_argument("--case-sensitive", action="store_true", help="Use case-sensitive matching.")
    parser.add_argument("--whole-word", action="store_true", help="Respect word boundaries.")
    parser.add_argument("--threads", type=int, default=1, help="Number of worker threads.")
    parser.add_argument("--self-test", action="store_true", help="Run a minimal self-test.")
    return parser


def main() -> int:
    """CLI entry point."""
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
