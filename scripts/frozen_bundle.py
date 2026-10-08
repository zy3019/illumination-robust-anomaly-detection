"""Export/restore the exact frozen inference tree, without datasets or final outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe bundle path: ' + relative)
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes destination')
    return path


def export_bundle(source, output, digest):
    source, output = (Path(os.path.abspath(p)).resolve() for p in (source,output))
    protocol_path = source/'records/formal_v1/protocol.json'
    if sha(protocol_path) != digest:
        raise ValueError('Protocol digest mismatch')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    files = dict(protocol['source_sha256'])
    files.update(protocol['assets_sha256'])
    for job in protocol['jobs']:
        files.update(job['frozen_files'])
        for name in ['scores.csv', 'metrics.csv']:
            relative = job['source_output'] + '/' + name
            files[relative] = sha(safe(source, relative))
    files['records/formal_v1/protocol.json'] = digest
    for name in ['records/formal_v1/protocol.sha256','records/requirements-w1-lock.txt',
                 'docs/formal_execution.md','scripts/download_assets.py','.gitattributes']:
        files[name] = sha(safe(source, name))
    for relative, expected in files.items():
        if sha(safe(source, relative)) != expected:
            raise ValueError('Frozen input changed: ' + relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    manifest = dict(schema=1, protocol_sha256=digest, files_sha256=files,
                    excludes=['dataset images/masks', 'final evaluation outputs'],
                    purpose='Restore historical frozen inference; not a new preregistration')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for relative in sorted(files):
            archive.write(safe(source, relative), relative)
        archive.writestr('BUNDLE.json', json.dumps(manifest, indent=2))
    checksum = sha(output)
    output.with_suffix(output.suffix+'.sha256').write_text(checksum+'\n', encoding='ascii')
    return dict(archive=str(output), sha256=checksum, files=len(files), bytes=output.stat().st_size)


def restore_bundle(archive_path, digest, target):
    archive_path, target = (Path(os.path.abspath(p)).resolve() for p in (archive_path,target))
    if target.exists():
        raise FileExistsError('Restore requires an absent destination: ' + str(target))
    if sha(archive_path) != digest:
        raise ValueError('Archive digest mismatch')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bundle-', dir=target.parent) as temporary:
        staging = Path(temporary)/'tree'
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read('BUNDLE.json'))
            expected = manifest['files_sha256']
            if len(names) != len(set(names)) or set(names) != set(expected)|{'BUNDLE.json'}:
                raise ValueError('Duplicate, missing, or extra archive files')
            for relative, digest_file in expected.items():
                path = safe(staging, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(relative) as incoming, path.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if sha(path) != digest_file:
                    raise ValueError('Restored file digest mismatch: ' + relative)
            (staging/'BUNDLE.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        staging.rename(target)
    return dict(restored=str(target), files=len(expected), protocol_sha256=manifest['protocol_sha256'])


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--digest', required=True)
    restore = commands.add_parser('restore')
    restore.add_argument('--archive', type=Path, required=True)
    restore.add_argument('--sha256', required=True)
    restore.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    result = export_bundle(args.source,args.output,args.digest) if args.command=='export' else restore_bundle(args.archive,args.sha256,args.target)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
