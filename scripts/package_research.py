"""Build checksum-verified, Git-sized research snapshot parts; no credentials."""
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):d.update(chunk)
    return d.hexdigest()


def files_under(names):
    files=[]
    for name in names:
        p=ROOT/name
        if not p.exists():raise FileNotFoundError(p)
        files.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    return sorted(set(files))


def build():
    out=ROOT/'research_artifacts';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Snapshot already exists')
    oracle=files_under(['outputs/oracle_lane_v1'])
    oracle=[p for p in oracle if p.name not in ('samples.jsonl','world.json')]
    real=files_under(['outputs/v4_world300_inside13/audit','outputs/v4_world300_inside13/config.json',
        'outputs/v4_world300_inside13/accepted_world_seeds.json','outputs/v4_world300_inside13/rejected_worlds.json',
        'outputs/v4_world300_inside13/preview.png'])
    runs=files_under(['outputs/training_oracle_v1','outputs/context_comparison_inside13'])
    # Archive small historic reports/plots at their original paths, without old datasets/checkpoints/caches.
    historic=[]
    for d in ('outputs/context_comparison','outputs/training_v1','outputs/training_v1_world300_fixedguide'):
        for p in (ROOT/d).rglob('*'):
            if p.is_file() and p.suffix in ('.md','.json','.csv','.html','.png') and p.stat().st_size<10*1024*1024:
                historic.append(p)
    groups={'oracle_dataset':oracle,'inside13_inputs':real,'current_runs':runs,'historical_reports':sorted(historic)}
    archives=[];part_size=8*1024*1024
    for name,paths in groups.items():
        archive=out/f'{name}.tar.gz'
        records=[]
        with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
            for p in paths:
                if p.is_symlink():raise ValueError(f'Symlink not allowed: {p}')
                before=p.stat();digest=sha256(p);rel=p.relative_to(ROOT).as_posix()
                tar.add(p,arcname=rel,recursive=False)
                after=p.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError(f'File changed: {p}')
                records.append({'path':rel,'bytes':before.st_size,'sha256':digest})
        parts=[]
        with archive.open('rb') as f:
            i=0
            for chunk in iter(lambda:f.read(part_size),b''):
                p=out/f'{name}.tar.gz.part{i:03d}';p.write_bytes(chunk)
                parts.append({'path':p.name,'bytes':len(chunk),'sha256':sha256(p)});i+=1
        entry={'name':name,'archive_sha256':sha256(archive),'archive_bytes':archive.stat().st_size,'parts':parts,'files':records}
        archives.append(entry)
        print(name,len(paths),'files',round(archive.stat().st_size/1024/1024,2),'MiB',len(parts),'parts',flush=True)
    manifest={'version':1,'snapshot_date':'2026-10-25','archives':archives,
        'aliases':[{'source':'outputs/oracle_lane_v1/worlds','destination':'outputs/v4_world300_inside13/worlds'},
                   {'source':'outputs/oracle_lane_v1/routes','destination':'outputs/v4_world300_inside13/routes'}],
        'omitted':['duplicate samples.jsonl (rebuild from inherited splits)',
                   'older duplicate datasets and old model checkpoints',
                   'training FOV caches (rebuild with documented commands)',
                   'Python/IDE caches and machine credentials'],
        'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy','matplotlib','shapely','pytest','tensorboard')},
        'python':platform.python_version()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Total compressed MiB',round(sum(a['archive_bytes'] for a in archives)/1024/1024,2))


if __name__=='__main__':build()
