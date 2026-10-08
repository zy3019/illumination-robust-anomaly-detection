"""Read-only gate for accepted numerical tables and report assets (stdlib only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_report_inputs(root=ROOT, manifest=None):
    root = Path(root).resolve()
    manifest = Path(manifest) if manifest else root/'records/report_inputs_v1.json'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if record['status'] != 'accepted_inputs':
        raise ValueError('Report inputs are not accepted')
    for relative, expected in record['files_sha256'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Report input escapes root: ' + relative)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Accepted report input changed: ' + relative)
    return record


if __name__ == '__main__':
    print('Verified', len(verify_report_inputs()['files_sha256']), 'accepted report inputs')
