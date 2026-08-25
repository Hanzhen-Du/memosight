#!/usr/bin/env python3
"""Bulk-download diverse high-resolution images from Pexels by keyword. Data collection for
gatekeeper training.

The gatekeeper is a binary classifier and needs diverse training data:
  - positive:        useful text on a screen (slides, projector, code screen, document)
  - negative_noise:  text that is not a launch scene (signage, book spines, packaging,
                     lock screens)
  - negative_clean:  no text at all (landscapes, portraits, objects)

This script only fetches full-resolution originals and files them by category and keyword. It
does not downscale; greyscale conversion and resizing are handled by
scripts/extract_frames.py.

Keywords live in scripts/keywords.json, so adding or removing them needs no code change.

Image source: Pexels, free with registration. Documentation:
https://www.pexels.com/api/documentation/
The API key is read from the PEXELS_API_KEY environment variable, or from a .env file at the
project root (.env is gitignored). The .env format is one KEY=VALUE per line:
    PEXELS_API_KEY=your_key

Output layout, rooted at data/raw by default:
    data/raw/<category>/<keyword-slug>/<keyword-slug>_<seq>_<pexels-image-id>.jpg

The filename carries the Pexels image id for two reasons:
  - Global dedup across keywords and categories. At startup the output root is scanned
    recursively to build a global set of ids, so the same Pexels image is downloaded only once
    across the whole dataset. This eliminates at source the failure where one image is fetched
    under several keywords, scattered across subclasses, and then leaks across splits
    (root cause in docs/data-leakage-audit.md).
  - The incrementing sequence number prevents accidental overwrites.

Dependencies: requests (see requirements.txt).

Examples:
    python3 scripts/download_images.py                      # every keyword in keywords.json
    python3 scripts/download_images.py --category positive  # positives only
    python3 scripts/download_images.py --dry-run            # print the plan, download nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit(
        "Missing dependency: requests. Install it with: pip install requests\n"
        "(It is listed in requirements.txt; this script will not install it for you.)"
    )

# Pexels search endpoint. At most 80 results per page, per the official documentation.
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
PEXELS_MAX_PER_PAGE = 80

# Default output root (data/ is not committed).
DEFAULT_OUTPUT_ROOT = "data/raw"
# Default keyword config, alongside this script.
DEFAULT_CONFIG = Path(__file__).resolve().parent / "keywords.json"

# Rate limiting: default gap in seconds between two API or download requests. The Pexels free
# tier allows 200 requests per hour and 20000 per month, so a delay is both polite and avoids
# tripping the limit.
DEFAULT_DELAY = 1.0
# Default retry count and backoff base in seconds for a failed request.
DEFAULT_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
# Request timeout in seconds.
REQUEST_TIMEOUT = 30

# Keys starting with an underscore in the config are comments, not categories.
_COMMENT_PREFIX = "_"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk-download diverse high-resolution images from Pexels by keyword, filed by "
                    "category and keyword. Data collection for gatekeeper training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="keyword config file (JSON).",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="output root; subfolders are created per category and keyword.",
    )
    parser.add_argument(
        "--category",
        action="append",
        metavar="NAME",
        help="download only these categories; may be repeated. Defaults to every category in the config.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="seconds to wait between requests (rate limiting).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="how many times to retry a failed request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the download plan only; make no requests and download nothing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="reduce progress output.",
    )
    return parser.parse_args(argv)


def load_env_file(env_path: Path) -> None:
    """Minimal .env reader: put KEY=VALUE into os.environ without overwriting existing
    variables.

    Supports one KEY=VALUE per line, treats lines starting with # as comments, and strips
    quotes around the value. Adds no dependency.
    """
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key(project_root: Path) -> str:
    """Read PEXELS_API_KEY from the environment or from .env at the project root, with a clear
    error if it is missing."""
    load_env_file(project_root / ".env")
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        sys.exit(
            "No Pexels API key found.\n"
            "Request one free at https://www.pexels.com/api/ and then set it either way:\n"
            "  1) environment variable:  export PEXELS_API_KEY=your_key\n"
            f"  2) create a .env file at the project root (it is gitignored) containing:\n"
            "       PEXELS_API_KEY=your_key\n"
            "Until you have a key, --dry-run will still show the download plan."
        )
    return key


def slugify(query: str) -> str:
    """Turn a search term into a slug safe for folder and file names: lowercase, with
    non-alphanumerics replaced by underscores."""
    s = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return s or "query"


def load_config(config_path: Path, only_categories: list[str] | None) -> list[dict]:
    """Read the config and expand it into a list of {category, query, count} tasks. Keys
    beginning with an underscore are comments and are skipped."""
    if not config_path.is_file():
        sys.exit(f"config file does not exist: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"config file is not valid JSON: {config_path}\n{exc}")

    tasks: list[dict] = []
    for category, entries in data.items():
        if category.startswith(_COMMENT_PREFIX):
            continue
        if only_categories and category not in only_categories:
            continue
        if not isinstance(entries, list):
            sys.exit(f"category {category!r} should map to a list, got {type(entries).__name__}.")
        for entry in entries:
            query = entry.get("query")
            count = entry.get("count")
            if not query or not isinstance(count, int) or count <= 0:
                sys.exit(f"category {category!r} contains an invalid entry "
                         f"(needs query plus a positive integer count): {entry}")
            tasks.append({"category": category, "query": query, "count": count})

    if only_categories:
        known = [c for c in data if not c.startswith(_COMMENT_PREFIX)]
        missing = [c for c in only_categories if c not in known]
        if missing:
            sys.exit(f"these categories are not in the config: {missing}\navailable: {known}")
    if not tasks:
        sys.exit("nothing to download: the config is empty, or --category filtered everything out.")
    return tasks


def existing_photo_ids(dest_dir: Path) -> set[str]:
    """Scan one folder for already-downloaded images and extract the Pexels image id from the
    trailing _<id>.<ext> in each filename."""
    ids: set[str] = set()
    if not dest_dir.is_dir():
        return ids
    for p in dest_dir.iterdir():
        if not p.is_file():
            continue
        m = re.search(r"_(\d+)$", p.stem)
        if m:
            ids.add(m.group(1))
    return ids


def all_existing_photo_ids(output_root: Path) -> set[str]:
    """Recursively scan the whole output root for the Pexels image ids of every downloaded
    image, across categories and keywords.

    This is what makes global dedup possible. The same Pexels image is often returned by
    several keywords, and deduplicating within a single folder would let it scatter across
    subclasses under different filenames, which then leaks across splits when the data is
    divided (see docs/data-leakage-audit.md).
    """
    ids: set[str] = set()
    if not output_root.is_dir():
        return ids
    for p in output_root.rglob("*"):
        if not p.is_file():
            continue
        m = re.search(r"_(\d+)$", p.stem)
        if m:
            ids.add(m.group(1))
    return ids


def request_with_retries(
    method: str,
    url: str,
    retries: int,
    quiet: bool,
    **kwargs,
) -> requests.Response | None:
    """Request with retries and exponential backoff. Returns the Response even on 4xx or 5xx,
    leaving the caller to decide, and returns None if every retry hit a network error."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            # 429 (rate limited) and 5xx are worth retrying; anything else goes back to the caller.
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                if not quiet:
                    print(f"    HTTP {resp.status_code}, retrying in {wait:.0f}s ({attempt}/{retries})...")
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** (attempt - 1)
            if not quiet:
                print(f"    request failed: {exc}; retrying in {wait:.0f}s ({attempt}/{retries})...")
            time.sleep(wait)
    return None


def search_photos(
    api_key: str,
    query: str,
    want: int,
    delay: float,
    retries: int,
    quiet: bool,
) -> list[dict]:
    """Search Pexels, paging until `want` results are collected, and return a list of
    {id, url} where url is the original."""
    headers = {"Authorization": api_key}
    photos: list[dict] = []
    page = 1
    while len(photos) < want:
        per_page = min(PEXELS_MAX_PER_PAGE, want - len(photos))
        params = {"query": query, "per_page": per_page, "page": page}
        resp = request_with_retries(
            "GET", PEXELS_SEARCH_URL, retries, quiet, headers=headers, params=params
        )
        if resp is None:
            print(f"    search for {query!r} failed on page {page} (network error, retries "
                      "exhausted); skipping the rest of this keyword.")
            break
        if resp.status_code == 401:
            sys.exit("Pexels returned 401 Unauthorized: the API key is invalid or expired. "
                     "Check PEXELS_API_KEY.")
        if resp.status_code != 200:
            print(f"    search for {query!r} returned HTTP {resp.status_code}; skipping the "
                      "rest of this keyword.")
            break

        batch = resp.json().get("photos", [])
        if not batch:
            if not quiet:
                print(f"    {query!r} has no more results after page {page}; {len(photos)} "
                          "available in total.")
            break
        for ph in batch:
            src = ph.get("src", {}).get("original")
            if src:
                photos.append({"id": str(ph["id"]), "url": src})
        page += 1
        time.sleep(delay)
    return photos[:want]


def url_extension(url: str) -> str:
    """Infer the image extension from the URL path, defaulting to .jpg."""
    path = url.split("?", 1)[0]
    ext = Path(path).suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def download_photo(
    url: str,
    dest_path: Path,
    retries: int,
    quiet: bool,
) -> bool:
    """Download one image to dest_path. Writes to a temporary file and renames, so a truncated
    file cannot masquerade as a completed download."""
    resp = request_with_retries("GET", url, retries, quiet, stream=True)
    if resp is None or resp.status_code != 200:
        code = "network error" if resp is None else f"HTTP {resp.status_code}"
        if not quiet:
            print(f"    download failed ({code}): {url}")
        return False
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(dest_path)
        return True
    except OSError as exc:
        if not quiet:
            print(f"    failed to write file: {exc}")
        tmp_path.unlink(missing_ok=True)
        return False


def run_task(
    task: dict,
    api_key: str,
    output_root: Path,
    seen_ids: set[str],
    delay: float,
    retries: int,
    dry_run: bool,
    quiet: bool,
) -> tuple[int, int]:
    """Run one (category, query, count) task. Returns (newly downloaded, skipped).

    seen_ids is the global set of already-downloaded image ids across categories and keywords,
    updated in place. Each Pexels image id is downloaded once for the whole dataset, which
    prevents cross-split leakage at source.
    """
    category, query, want = task["category"], task["query"], task["count"]
    slug = slugify(query)
    dest_dir = output_root / category / slug

    already = existing_photo_ids(dest_dir)  # what this folder already has, to continue numbering
    print(f"[{category}/{slug}] target {want}, this folder already has {len(already)}, "
          f"{len(seen_ids)} ids known globally")

    if dry_run:
        print(f"    (dry-run) would search {query!r} and download up to {want} into {dest_dir}")
        return 0, 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    photos = search_photos(api_key, query, want, delay, retries, quiet)
    if not quiet:
        print(f"    {len(photos)} candidates found")

    downloaded = 0
    skipped = 0
    seq = len(already)  # continue from the existing sequence
    for ph in photos:
        # Global dedup: skip the id if it appears anywhere in the dataset, under any keyword
        # or category.
        if ph["id"] in seen_ids:
            skipped += 1
            continue
        seq += 1
        ext = url_extension(ph["url"])
        fname = f"{slug}_{seq:04d}_{ph['id']}{ext}"
        if download_photo(ph["url"], dest_dir / fname, retries, quiet):
            downloaded += 1
            seen_ids.add(ph["id"])  # add to the global set at once so later keywords skip it
            if not quiet and downloaded % 20 == 0:
                print(f"    downloaded {downloaded}...")
        else:
            seq -= 1  # download failed, give the sequence number back
        time.sleep(delay)

    print(f"    done: {downloaded} downloaded, {skipped} skipped as existing or duplicate "
          f"-> {dest_dir}")
    return downloaded, skipped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root).expanduser()
    config_path = Path(args.config).expanduser()

    tasks = load_config(config_path, args.category)

    total_want = sum(t["count"] for t in tasks)
    print(f"config: {config_path}")
    print(f"output root: {output_root.resolve()}")
    print(f"tasks: {len(tasks)}, planned download ceiling: {total_want}")
    if args.category:
        print(f"categories restricted to: {args.category}")
    print("-" * 50)

    # A dry run needs no key; only a real download does.
    api_key = "" if args.dry_run else get_api_key(project_root)

    # Global set of downloaded ids across categories and keywords. This is what makes
    # cross-keyword dedup possible and stops leakage at source.
    seen_ids = all_existing_photo_ids(output_root)
    print(f"{len(seen_ids)} image ids known globally (the cross-keyword dedup baseline)")
    print("-" * 50)

    total_dl = 0
    total_skip = 0
    for task in tasks:
        dl, sk = run_task(
            task, api_key, output_root, seen_ids,
            args.delay, args.retries, args.dry_run, args.quiet
        )
        total_dl += dl
        total_skip += sk

    print("-" * 50)
    if args.dry_run:
        print("(dry-run) plan printed; no files downloaded.")
    else:
        print(f"all done: {total_dl} downloaded, {total_skip} skipped as existing "
              f"-> {output_root.resolve()}")


if __name__ == "__main__":
    main()
