# Reddit Content Tagger

`tag_reddit_content.py` tags Reddit posts and comments by matching phrases
from a coding rules CSV.

It scans JSON files, finds post/comment structures heuristically, matches
keywords with regex, and writes:

- `tagged_content.csv` (one row per matched keyword per content item)
- `phrases_found.csv` (per-keyword aggregate counts)
- `results.xlsx` (multi-sheet workbook with all outputs + metadata)

## Features

- Flexible Reddit JSON parsing (supports nested and mixed structures).
- Keyword loading from CSV (`keyword` column required).
- Case-sensitive or case-insensitive matching.
- Optional whole-word matching.
- Single-threaded or multithreaded processing.
- Optional timestamp prefix/suffix for output filenames.
- XLSX export with sheets:
  - `tagged_content`
  - `phrases_found`
  - `coding_rules`
  - `metadata`

## Inputs

1. Data directory with `.json` files (`--data-dir`).
2. Rules CSV (`--rules`) with a `keyword` column.

### `--data-dir` expected contents

- Directory containing one or more `.json` files.
- Each file should represent a Reddit post and its comments (recommended: one
  post thread per file).
- Non-JSON files are ignored.
- Invalid/unreadable JSON files are skipped with warnings.

### Expected JSON format

The parser is heuristic and flexible. JSON may be a dict or list, and may
include wrapper objects (for example Reddit API-style `kind`/`data`).

Post-like objects are usually identified by fields such as:

- `title`
- `selftext`, `selftext_html`, `text`, or `body`
- `id` / `post_id` / `name` / `full_id`
- `subreddit` / `subreddit_name_prefixed` / `community` / `url` / `permalink`
- `author` / `user` / `username`
- `created_utc` / `created` / `created_at` / `createdAt` / `date` /
  `timestamp` / `published`
- `comments` (list)

Comment-like objects are usually identified by fields such as:

- `body`, `text`, or `comment`
- `id` / `comment_id` / `name` / `full_id`
- `parent_id` or `link_id`
- `author` / `user` / `username`
- `created_utc` / `created` / `created_at` / `createdAt` / `date` /
  `timestamp` / `published`
- `score` (or `ups`/`downs`)
- nested `comments` (supported)

### Example JSON file

```json
{
  "id": "abc123",
  "subreddit": "singularity",
  "author": "alice",
  "created_utc": 1700000000,
  "title": "AI progress this week",
  "selftext": "AGI discussion and model updates.",
  "comments": [
    {
      "id": "c1",
      "author": "bob",
      "created_utc": 1700000100,
      "body": "Interesting AGI timeline.",
      "ups": 5,
      "downs": 0
    },
    {
      "id": "c2",
      "author": "eve",
      "created_utc": 1700000200,
      "body": "Thanks for sharing."
    }
  ]
}
```

Example `coding_rules.csv`:

```csv
keyword
AGI
divine
singularity
```

## Outputs

### `tagged_content.csv`

Columns:

- `root_post_id`
- `subreddit`
- `content_type` (`post` or `comment`)
- `content_id`
- `author`
- `content_text`
- `created_at`
- `likes`
- `dislikes`
- `unvoted`
- `keyword_id`
- `keyword`

### `phrases_found.csv`

Columns:

- `id`
- `keyword`
- `tagged_post_count`
- `tagged_content_count`

### `results.xlsx`

Contains sheets:

- `tagged_content`
- `phrases_found`
- `coding_rules`
- `metadata`

`metadata` contains:

- `command`: full command line used to run the script
- `working_directory`: current working directory at run time
- `run_timestamp`: run date/time with seconds and timezone

## Usage

```bash
python tag_reddit_content.py --rules test/demo_coding_rules/coding_rules.csv --data-dir test/demo_data
```

### CLI options

- `--data-dir`: directory with input JSON files (required)
- `--rules`: path to coding rules CSV (required)
- `--output-dir`: base directory for default outputs
- `--out-tagged`: explicit path for tagged CSV
- `--out-phrases`: explicit path for phrases CSV
- `--out-xlsx`: explicit path for XLSX
- `--add-timestamp-prefix`: write `YYYYMMDD-HHMMSS_filename.ext`
- `--add-timestamp-suffix`: write `filename_YYYYMMDD-HHMMSS.ext`
- `--case-sensitive`: case-sensitive keyword matching
- `--whole-word`: only match whole words when applicable
- `--threads`: number of worker threads (default `1`)
- `--self-test`: run built-in synthetic self-test

`--add-timestamp-prefix` and `--add-timestamp-suffix` are mutually exclusive.

## Examples

Write default outputs into `test/out`:

```bash
python tag_reddit_content.py ^
  --data-dir test/demo_data ^
  --rules test/demo_coding_rules/coding_rules.csv ^
  --output-dir test/out
```

Timestamped filenames with suffix:

```bash
python tag_reddit_content.py ^
  --data-dir test/demo_data ^
  --rules test/demo_coding_rules/coding_rules.csv ^
  --output-dir test/out ^
  --add-timestamp-suffix
```

## Logging

The script logs progress and outputs, including:

- data directory
- output paths for CSV and XLSX files
- counts for processed files, posts/comments, tagged rows, and matched phrases

## Dependencies

- Python 3.10+ (recommended)
- `pandas` (required for XLSX output)
- `pytest` (for tests)

Install dependencies:

```bash
python -m pip install pandas pytest
```

## Tests

Run tests:

```bash
python -m pytest
```
