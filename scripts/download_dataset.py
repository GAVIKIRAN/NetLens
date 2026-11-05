"""Download and install the CIC-IDS2017 flow CSVs into ``data/raw``.

The dataset is published by the Canadian Institute for Cybersecurity (UNB).
This script fetches the ~224 MB ``MachineLearningCSV.zip`` archive, verifies it,
and extracts the eight CSV files into the directory the training pipeline reads.

    python scripts\\download_dataset.py                 # download + install
    python scripts\\download_dataset.py --zip C:\\Users\\me\\Downloads\\MachineLearningCSV.zip
    python scripts\\download_dataset.py --labelled      # GeneratedLabelledFlows (has Protocol)
    python scripts\\download_dataset.py --links         # just print the URLs
    python scripts\\download_dataset.py --verify        # check what is already installed

If the CIC servers are slow or unreachable from your network (they often are
outside North America), use ``--links``, download the zip in a browser or with a
download manager, and then re-run with ``--zip <path>``.  That path does no
networking at all.
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from _bootstrap import bootstrap

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# The CIC publishes the same archives on a bare-IP host and behind a hostname.
# Both are tried in order; the IP host serves a certificate that does not match
# its own address, which is why it needs the relaxed TLS context below.
ARCHIVES: dict[str, dict[str, object]] = {
    "MachineLearningCSV.zip": {
        "size_mb": 224,
        "description": "8 flow CSVs, ~2.83 M rows. This is what the trainer wants.",
        "urls": [
            "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/MachineLearningCSV.zip",
            "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/MachineLearningCSV.zip",
            "https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/MachineLearningCSV.zip",
        ],
    },
    "GeneratedLabelledFlows.zip": {
        "size_mb": 271,
        "description": (
            "Same flows plus Flow ID / IPs / Protocol / Timestamp columns. "
            "Only useful here because it carries the Protocol column, which "
            "MachineLearningCSV omits."
        ),
        "urls": [
            "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/GeneratedLabelledFlows.zip",
            "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/GeneratedLabelledFlows.zip",
            "https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/GeneratedLabelledFlows.zip",
        ],
    },
}

LANDING_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
KAGGLE_MIRRORS = (
    ("chethuhn/network-intrusion-dataset", "the eight original CSVs, unmodified"),
    ("dhoogla/cicids2017", "a cleaned/de-duplicated parquet rebuild (different row counts)"),
)

CITATION = (
    "Sharafaldin, I., Habibi Lashkari, A. and Ghorbani, A.A. (2018). "
    "'Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic "
    "Characterization'. ICISSP 2018, Portugal."
)


# ---------------------------------------------------------------------------
#  Networking
# ---------------------------------------------------------------------------
def _ssl_context(url: str) -> ssl.SSLContext | None:
    """Relaxed TLS only for the bare-IP CIC host, which mis-serves its cert."""
    if not url.lower().startswith("https"):
        return None
    if "205.174.165.80" in url:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return None


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:,.1f} {unit}"
        num /= 1024.0
    return f"{num:,.1f} TB"


def download(url: str, destination: Path, timeout: float = 60.0) -> bool:
    """Stream ``url`` to ``destination``, resuming a partial file if present.

    Returns ``True`` on success. Never raises for network problems - the caller
    moves on to the next mirror.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
        print(f"    resuming from {_human(existing)}")

    try:
        context = _ssl_context(url)
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            status = getattr(response, "status", 200)
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                print("    server returned an HTML page, not the archive "
                      "(the CIC host redirects some clients) - trying the next mirror")
                return False

            total = response.headers.get("Content-Length")
            total = int(total) + (existing if status == 206 else 0) if total else 0
            mode = "ab" if (existing and status == 206) else "wb"
            if mode == "wb":
                existing = 0

            downloaded = existing
            started = time.time()
            last_report = 0.0
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report > 0.5:
                        last_report = now
                        rate = downloaded / max(0.001, now - started)
                        if total:
                            pct = 100.0 * downloaded / total
                            sys.stdout.write(
                                f"\r    {pct:5.1f}%  {_human(downloaded)} / "
                                f"{_human(total)}  at {_human(rate)}/s   "
                            )
                        else:
                            sys.stdout.write(
                                f"\r    {_human(downloaded)} at {_human(rate)}/s   "
                            )
                        sys.stdout.flush()
            sys.stdout.write("\r" + " " * 78 + "\r")
    except urllib.error.HTTPError as exc:
        print(f"    HTTP {exc.code} {exc.reason}")
        return False
    except urllib.error.URLError as exc:
        print(f"    unreachable: {exc.reason}")
        return False
    except TimeoutError:
        print("    timed out")
        return False
    except Exception as exc:  # pragma: no cover - network is unpredictable
        print(f"    failed: {type(exc).__name__}: {exc}")
        return False

    if not zipfile.is_zipfile(partial):
        print("    the downloaded file is not a valid zip archive")
        return False

    partial.replace(destination)
    print(f"    saved {destination.name} ({_human(destination.stat().st_size)})")
    return True


def fetch_archive(name: str, target_dir: Path) -> Path | None:
    """Try every mirror for ``name`` until one yields a valid zip."""
    spec = ARCHIVES[name]
    destination = target_dir / name
    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"  {name} is already downloaded ({_human(destination.stat().st_size)})")
        return destination

    print(f"  Downloading {name} (~{spec['size_mb']} MB)")
    for index, url in enumerate(spec["urls"], start=1):  # type: ignore[index]
        print(f"    [{index}/{len(spec['urls'])}] {url}")  # type: ignore[arg-type]
        if download(url, destination):
            return destination
    return None


# ---------------------------------------------------------------------------
#  Installation
# ---------------------------------------------------------------------------
def install(archive: Path, raw_dir: Path) -> list[Path]:
    """Extract every CSV in ``archive`` directly into ``raw_dir`` (flattened)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []

    with zipfile.ZipFile(archive) as bundle:
        members = [
            info for info in bundle.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".csv")
        ]
        if not members:
            print(f"  [!] {archive.name} contains no CSV files.")
            return installed

        print(f"  Extracting {len(members)} CSV file(s) into {raw_dir}")
        for info in members:
            # Flatten: the archives nest the CSVs one folder deep, and the
            # loader does not care about the layout, but a flat folder is
            # easier to eyeball.
            name = Path(info.filename).name
            target = raw_dir / name
            with bundle.open(info) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            installed.append(target)
            print(f"    {name}  ({_human(target.stat().st_size)})")
    return installed


def verify(raw_dir: Path) -> int:
    """Report what is installed and confirm the loader can read it."""
    from src.data.loader import detect_label_column, discover_csv_files, read_header
    from src.utils.errors import DatasetError

    try:
        files = discover_csv_files(raw_dir)
    except DatasetError as exc:
        print(f"\n  [!] {exc}\n")
        return 1

    print(f"\n  {len(files)} CSV file(s) in {raw_dir}\n")
    total_bytes = 0
    ok = True
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        try:
            columns = read_header(path, "latin-1")
            label = detect_label_column(columns)
            note = f"{len(columns)} columns, label='{label}'"
            if "protocol" in columns:
                note += ", has Protocol"
        except DatasetError as exc:
            note = f"UNREADABLE: {exc}"
            ok = False
        print(f"    {path.name:<62s} {_human(size):>10s}   {note}")

    print(f"\n  Total: {_human(total_bytes)}")
    if ok:
        print("\n  Ready. Next:  python scripts\\profile_dataset.py")
        print("          then:  python scripts\\train_model.py\n")
        return 0
    print("\n  Some files could not be read - see above.\n")
    return 1


def print_links() -> None:
    print("\n" + "=" * 78)
    print("  CIC-IDS2017 download links")
    print("=" * 78)
    print(f"\n  Official landing page: {LANDING_PAGE}\n")
    for name, spec in ARCHIVES.items():
        print(f"  {name}  (~{spec['size_mb']} MB)")
        print(f"    {spec['description']}")
        for url in spec["urls"]:  # type: ignore[index]
            print(f"      {url}")
        print()
    print("  Kaggle mirrors (need a free account + API token):")
    for slug, note in KAGGLE_MIRRORS:
        print(f"      kaggle datasets download -d {slug}")
        print(f"        {note}")
    print()
    print("  Downloaded it manually? Install it with:")
    print("      python scripts\\download_dataset.py --zip <path-to-zip>")
    print()
    print(f"  Cite as: {CITATION}")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and install the CIC-IDS2017 flow CSVs."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--zip", dest="zip_path", default=None,
                        help="install an already-downloaded archive (no networking)")
    parser.add_argument("--labelled", action="store_true",
                        help="fetch GeneratedLabelledFlows.zip instead (includes Protocol)")
    parser.add_argument("--both", action="store_true", help="fetch both archives")
    parser.add_argument("--links", action="store_true", help="print the URLs and exit")
    parser.add_argument("--verify", action="store_true",
                        help="report what is already installed and exit")
    parser.add_argument("--keep-zip", action="store_true",
                        help="keep the downloaded archive after extracting")
    parser.add_argument("--dest", default=None, help="override paths.raw_data_dir")
    args = parser.parse_args()

    if args.links:
        print_links()
        return 0

    settings = bootstrap(args.config)
    raw_dir = Path(args.dest) if args.dest else settings.path("raw_data_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return verify(raw_dir)

    print("\n" + "=" * 78)
    print("  CIC-IDS2017 installer")
    print("=" * 78)
    print(f"  Destination: {raw_dir}\n")

    installed: list[Path] = []

    # ------------------------------------------------ local archive path --
    if args.zip_path:
        archive = Path(args.zip_path).expanduser()
        if not archive.exists():
            print(f"  [!] File not found: {archive}\n")
            return 2
        if not zipfile.is_zipfile(archive):
            print(f"  [!] Not a zip archive: {archive}\n")
            return 2
        installed = install(archive, raw_dir)
    # ---------------------------------------------------- download path ---
    else:
        wanted = list(ARCHIVES) if args.both else (
            ["GeneratedLabelledFlows.zip"] if args.labelled else ["MachineLearningCSV.zip"]
        )
        downloads_dir = raw_dir.parent / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)

        for name in wanted:
            archive = fetch_archive(name, downloads_dir)
            if archive is None:
                print(f"\n  [!] Could not download {name} from any mirror.")
                print("      The CIC servers are frequently slow or blocked outside")
                print("      North America. Download it in a browser instead:\n")
                print_links()
                return 3
            installed.extend(install(archive, raw_dir))
            if not args.keep_zip:
                archive.unlink(missing_ok=True)
                print(f"  Removed {archive.name} (pass --keep-zip to keep it)")

    if not installed:
        print("\n  [!] Nothing was installed.\n")
        return 3

    print(f"\n  Installed {len(installed)} CSV file(s).")
    print(f"\n  Cite as: {CITATION}")
    return verify(raw_dir)


if __name__ == "__main__":
    raise SystemExit(main())
