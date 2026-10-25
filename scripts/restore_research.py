"""Restore and verify published snapshot, then relocate report paths safely.

Uses only Python standard library. Never rewrites dataset JSONL or checkpoints.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def restore(root,verify_only=False):
    root=Path(root).resolve();artifacts=root/'research_artifacts'
    manifest=json.loads((artifacts/'manifest.json').read_text())
    for entry in manifest['archives']:
        with tempfile.TemporaryFile() as combined:
            d=hashlib.sha256()
            for part in entry['parts']:
                path=artifacts/part['path']
                if path.stat().st_size!=part['bytes'] or sha256(path)!=part['sha256']:
                    raise ValueError(f'Part checksum mismatch: {path}')
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):combined.write(chunk);d.update(chunk)
            if d.hexdigest()!=entry['archive_sha256']:raise ValueError('Archive checksum mismatch')
            combined.seek(0);expected={r['path']:r for r in entry['files']};seen=set()
            with tarfile.open(fileobj=combined,mode='r:gz') as tar:
                for member in tar:
                    target=(root/member.name).resolve()
                    if not target.is_relative_to(root) or not member.isfile() or member.name not in expected:
                        raise ValueError(f'Unexpected archive member {member.name}')
                    data=tar.extractfile(member);digest=hashlib.sha256()
                    destination=None
                    if not verify_only:
                        if target.exists():raise FileExistsError(f'{target}: use a clean clone to restore')
                        target.parent.mkdir(parents=True,exist_ok=True);destination=target.open('wb')
                    try:
                        for chunk in iter(lambda:data.read(1024*1024),b''):
                            digest.update(chunk)
                            if destination:destination.write(chunk)
                    finally:
                        if destination:destination.close()
                    if digest.hexdigest()!=expected[member.name]['sha256']:raise ValueError(f'File checksum mismatch {member.name}')
                    seen.add(member.name)
            if seen!=set(expected):raise ValueError('Archive file list mismatch')
        print(('Verified' if verify_only else 'Restored'),entry['name'],len(seen),'files',flush=True)
    if not verify_only:
        for alias in manifest['aliases']:
            src=root/alias['source'];dst=root/alias['destination']
            if dst.exists():raise FileExistsError(dst)
            shutil.copytree(src,dst)
        # Rebuild redundant aggregate source files without modifying split bytes.
        for name in ('oracle_lane_v1','v4_world300_inside13'):
            directory=root/'outputs'/name
            with (directory/'samples.jsonl').open('wb') as out:
                for split in ('train','val','test'):
                    with (directory/'audit'/f'{split}.jsonl').open('rb') as inp:shutil.copyfileobj(inp,out)
        prefix='/home/ygy/PathGenModel'
        changed=0
        for path in (root/'outputs').rglob('*.json'):
            # Keep source dataset/config/cache manifests byte-exact for resume fingerprints.
            if (path.name=='manifest.json' and path.parent.name!='fixed_val_supplements') or path.name=='config.json':continue
            if 'worlds' in path.parts or 'routes' in path.parts:continue
            text=path.read_text()
            if prefix in text:path.write_text(text.replace(prefix,root.as_posix()));changed+=1
        print('Rebuilt aggregate samples, restored shared Worlds/routes; relocated',changed,'JSON reports')
    print('All snapshot checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();restore(a.root,a.verify_only)
