from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path

from brainlab.data.corpus import synthetic_docs

DEFAULT_REPO = "salesforce/wikitext"


def repo_text_files(repo: str, prefix: str | None = None, revision: str = "main") -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=repo, revision=revision)
    keep = [f for f in files if f.lower().endswith(".txt")]
    if prefix:
        keep = [f for f in keep if f.startswith(prefix)]
    return sorted(keep)


def download_text_files(repo: str, out_dir: Path, prefix: str | None = None, revision: str = "main") -> list[Path]:
    from huggingface_hub import hf_hub_download

    names = repo_text_files(repo, prefix, revision)
    if not names:
        raise SystemExit(f"no .txt files found in {repo} with prefix {prefix!r}; pass --list to inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        cached = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision))
        flat = name.replace("/", "_")
        target = out_dir / flat
        shutil.copyfile(cached, target)
        written.append(target)
        print(f"{name} -> {target} ({target.stat().st_size:,} B)")
    return written


def write_synthetic(out_dir: Path, tokens: int, seed: int = 0, docs_per_file: int = 200) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = synthetic_docs(tokens, seed=seed)
    paths: list[Path] = []
    for file_index in range(0, max(len(docs), 1), docs_per_file):
        chunk = docs[file_index : file_index + docs_per_file]
        target = out_dir / f"synthetic-{file_index // docs_per_file:04d}.txt"
        target.write_text("\n\n".join(chunk) + "\n", encoding="utf-8")
        paths.append(target)
    print(f"{len(docs)} synthetic docs -> {len(paths)} files under {out_dir}")
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-data", description="Get a text corpus into data/corpus.")
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--synthetic", type=int, default=0, help="write N words of a deterministic corpus instead of downloading")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for name in repo_text_files(args.repo, args.prefix, args.revision):
            print(name)
        return 0
    if args.synthetic:
        write_synthetic(args.out, args.synthetic, args.seed)
        return 0
    download_text_files(args.repo, args.out, args.prefix, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
