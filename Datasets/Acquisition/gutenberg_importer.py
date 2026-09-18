"""Bulk acquisition from Project Gutenberg -- the largest legally clean
source of book text for a training corpus (public domain, ~75k texts).

IMPORTANT, please read before running at scale:
Project Gutenberg explicitly asks that people NOT scrape it via repeated
individual HTTP requests (see https://www.gutenberg.org/policy/robot_access.html)
-- their servers rate-limit and can block IPs that do. Two supported paths:

  1. rsync mirror (recommended for "huge" -- this is what their own docs
     recommend for bulk access): `download_via_rsync()` below shells out to
     the `rsync` binary against one of their official mirrors. This is a
     full mirror sync, not per-file requests, and is the sanctioned way to
     get the whole catalog or a large subset efficiently.
  2. Catalog-driven HTTP fallback (`download_via_http()`) for when rsync
     isn't available -- downloads a specific list of book IDs with
     conservative rate-limiting and a descriptive User-Agent, using their
     documented per-book URL pattern. Much slower than rsync for "huge"
     and should be used for hundreds of books, not tens of thousands.

NEITHER of these has been run/tested from this environment -- gutenberg.org
is outside this container's network allowlist. The URL patterns and rsync
invocation below match Project Gutenberg's own current documentation as of
this module's writing, but please verify against
https://www.gutenberg.org/help/mirroring.html before a large run, since
mirror hostnames and paths do occasionally change.
"""
from __future__ import annotations

import csv
import io
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "BraegnBrainCorpusBuilder/1.0 (contact: set-your-contact-info-here)"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
# Official rsync mirrors -- see https://www.gutenberg.org/help/mirroring.html
# for the current authoritative list; this one is commonly cited but pick
# whichever mirror's docs you're actually reading at run time.
DEFAULT_RSYNC_MIRROR = "rsync://aleph.gutenberg.org/gutenberg/"


@dataclass
class CatalogEntry:
    gutenberg_id: str
    title: str
    language: str
    subjects: str
    bookshelves: str
    type: str  # "Text", "Sound", ... -- filter to "Text"


def fetch_catalog() -> list[CatalogEntry]:
    """Downloads Project Gutenberg's official catalog CSV (one request,
    not per-book) so you can filter by language/type BEFORE downloading
    any book content. This is the documented, sanctioned way to enumerate
    what's available -- see https://www.gutenberg.org/ebooks/search/ and
    the catalog feed docs linked from the Gutenberg help pages.
    """
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    entries = []
    for row in csv.DictReader(io.StringIO(raw)):
        entries.append(CatalogEntry(
            gutenberg_id=row.get("Text#", "").strip(),
            title=row.get("Title", "").strip(),
            language=row.get("Language", "").strip(),
            subjects=row.get("Subjects", "").strip(),
            bookshelves=row.get("Bookshelves", "").strip(),
            type=row.get("Type", "").strip(),
        ))
    return entries


def download_via_rsync(
    output_dir: str | Path,
    mirror: str = DEFAULT_RSYNC_MIRROR,
    include_pattern: str = "*.txt",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """The recommended path for a "huge" download. Requires the `rsync`
    binary on PATH. include_pattern defaults to plain-text files only
    (skips HTML/EPUB/images) since that's what a text corpus needs.

    Runs synchronously and can take a very long time for the full mirror
    (the complete catalog is well over 100GB across all formats) -- filter
    with include_pattern, or point at a specific subdirectory of the
    mirror, rather than syncing everything.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-avm", "--del",
        f"--include={include_pattern}", "--include=*/",
        "--exclude=*",
        mirror, str(output_dir),
    ]
    if extra_args:
        cmd[1:1] = extra_args
    return subprocess.run(cmd, capture_output=True, text=True)


def download_via_http(
    gutenberg_ids: list[str],
    output_dir: str | Path,
    requests_per_second: float = 0.5,
) -> list[Path]:
    """Fallback for when rsync isn't available. Conservative default rate
    (1 request per 2 seconds) since this hits Gutenberg's own web servers
    directly, unlike the rsync mirror path. Use for hundreds of specific
    books (e.g. a curated subject list from fetch_catalog()), not for
    trying to pull the whole catalog this way.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    delay = 1.0 / requests_per_second if requests_per_second > 0 else 0.0

    for gid in gutenberg_ids:
        # This is Gutenberg's documented stable per-book plain-text URL
        # pattern (the "pg<id>.txt" cache path) -- verify against a live
        # book page if it ever 404s, since older/newer texts sometimes use
        # a slightly different suffix (e.g. -0.txt, -8.txt for encoding
        # variants) that this simple pattern doesn't try to disambiguate.
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        dest = output_dir / f"pg{gid}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            written.append(dest)
        except Exception as e:  # noqa: BLE001 -- log and continue a bulk job
            print(f"[gutenberg] failed to fetch id {gid}: {e}")
        time.sleep(delay)

    return written
