"""Download immutable, SHA-verified model files without an HF token or extra SDK."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parent
    info = json.loads((root / 'HF_WEIGHTS.json').read_text())
    variants = sorted({Path(n).parts[1] for n in info['files']})
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=variants, default='base-fp16')
    ap.add_argument('--all', action='store_true', help='Download all variants, about5GB')
    args = ap.parse_args()
    assert len(info['revision']) == 40 and all(c in '0123456789abcdef' for c in info['revision'])
    for name, proof in info['files'].items():
        rel = Path(name)
        assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == 'weights'
        if not args.all and rel.parts[1] != args.variant:
            continue
        target = root / rel
        assert target.resolve().is_relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == proof['bytes'] and digest(target) == proof['sha256']:
            print(f'Verified existing: {name}', flush=True)
            continue
        url = f"https://huggingface.co/{info['repo']}/resolve/{info['revision']}/{name}"
        temp = target.with_suffix(target.suffix + '.partial')
        sha = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as f:
            for block in iter(lambda: response.read(4 * 1024**2), b''):
                sha.update(block)
                f.write(block)
        if sha.hexdigest() != proof['sha256'] or temp.stat().st_size != proof['bytes']:
            raise ValueError(f'Checksum mismatch; partial file retained: {name}')
        temp.replace(target)
        print(f'Downloaded and verified: {name}', flush=True)


if __name__ == '__main__':
    main()
