"""Local BEV vector editor. Run: python scene_editor.py --port 8765."""
from __future__ import annotations
import argparse
from datetime import datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import webbrowser
from threading import Lock

_PREVIEW_LOCK = Lock()
EDITOR_VERSION = 6

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'tools' / 'scene_editor'
KINDS = {'guide', 'gt', 'line', 'curb', 'stop', 'cross'}
FIELDS = {'line': 'lane_markings', 'curb': 'road_boundaries',
          'stop': 'stop_lines', 'cross': 'crosswalks'}


def points(value, minimum=2):
    if not isinstance(value, list) or not minimum <= len(value) <= 8192:
        raise ValueError(f'控制点数量必须为 {minimum}–8192')
    result = []
    for p in value:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError('每个点必须为 [x, y]')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or abs(v) > 10000 for v in p):
            raise ValueError('坐标必须是有限的米制数值，绝对值不超过 10000')
        result.append([float(v) for v in p])
    return result


def curve_points(entity):
    p = points(entity.get('points'), 3 if entity.get('kind') in ('stop', 'cross') else 2)
    mode = entity.get('mode', 'polyline')
    if mode not in ('bezier', 'polyline', 'polygon'):
        raise ValueError('不支持的线形')
    if entity.get('kind') in ('stop', 'cross') and mode != 'polygon':
        raise ValueError('停止线和斑马线需为闭合多边形')
    if mode == 'bezier':
        if len(p) != 4:
            raise ValueError('平滑曲线需要起点、两个形状控制点和终点')
        length = sum(math.dist(a, b) for a, b in zip(p, p[1:]))
        n = min(8192, max(256, math.ceil(length / .1)))
        p = [[sum(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)[i]*p[i][k]
                  for i in range(4)) for k in range(2)] for t in (j/n for j in range(n+1))]
    if len({tuple(x) for x in p}) < (3 if entity.get('kind') in ('stop', 'cross') else 2):
        raise ValueError('线不能退化为一个点；多边形至少有三个不同顶点')
    return p


def sample_guide(p, label='Guide'):
    """Same station convention as infer.py: first 80 m, no stretch/extrapolation."""
    import numpy as np
    a = np.asarray(p, dtype=float)
    a = a[np.r_[True, np.linalg.norm(np.diff(a, axis=0), axis=1) > 1e-9]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
    if arc[-1] < 80-1e-6:
        raise ValueError(f'{label} 长度只有 {arc[-1]:.2f} m，需要至少 80 m。请延长终点，不会自动拉伸。')
    return np.column_stack([np.interp(np.arange(81), arc, a[:, k]) for k in range(2)]).tolist()


@lru_cache(maxsize=1)
def encoder():
    from infer import prepare_sample, encode_input
    return prepare_sample, encode_input


def export_project(project):
    if not isinstance(project, dict) or project.get('version') != 1:
        raise ValueError('不支持的编辑文件版本')
    nav = project.get('navigation')
    if nav not in ('left', 'straight', 'right'):
        raise ValueError('请选择 left / straight / right 导航')
    entities = project.get('entities')
    if not isinstance(entities, list) or len(entities) > 256:
        raise ValueError('最多支持 256 个元素')
    if sum(e.get('kind') == 'guide' for e in entities if isinstance(e, dict)) != 1:
        raise ValueError('场景需要且只能有一条 Guide')
    gt_count = sum(e.get('kind') == 'gt' for e in entities if isinstance(e, dict))
    if gt_count > 1 or gt_count and project.get('gt_path') is not None:
        raise ValueError('场景最多只能有一条 GT，不能同时提供旧 gt_path 和 GT 元素')
    name = project.get('name', 'custom_scene')
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise ValueError('场景名称不能为空，且不超过 128 个字符')
    row = {'id': name, 'navigation': nav, 'coordinate_system': 'ego_at_start_x_forward_y_left_m',
           **{field: [] for field in FIELDS.values()}}
    ids = set()
    for e in entities:
        if not isinstance(e, dict) or e.get('kind') not in KINDS:
            raise ValueError('元素类型无效')
        eid = e.get('id')
        if not isinstance(eid, str) or not eid or len(eid) > 128 or eid in ids:
            raise ValueError('元素 ID 不能为空或重复，长度不超过 128')
        ids.add(eid)
        p = curve_points(e)
        if e['kind'] == 'guide':
            row['reference_line'] = sample_guide(p)
            continue
        if e['kind'] == 'gt':
            if math.hypot(*p[0]) > 1e-6:
                raise ValueError('GT 起点必须为自车原点 (0,0)，请调整起点')
            row['gt_path'] = sample_guide(p, 'GT')
            continue
        confidence = e.get('confidence', 1.)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('置信度必须在 0–1 之间')
        obj = {'id': eid, 'points': p, 'confidence': confidence}
        if e['kind'] == 'line':
            color, style = e.get('color', 'white'), e.get('line_style', 'dashed')
            if color not in ('white', 'yellow', 'unknown') or style not in ('solid', 'dashed', 'unknown'):
                raise ValueError('无效的车道线颜色或线型')
            if color != 'unknown': obj['color'] = color
            if style != 'unknown': obj['line_style'] = style
        row[FIELDS[e['kind']]].append(obj)
    if project.get('gt_path') is not None:
        row['gt_path'] = points(project['gt_path'])
    prepare, encode = encoder()
    row = prepare(row)
    encoded = encode(row, 32, 32)
    summary = {'guide_points': 81, 'bev_shape': list(encoded['bev_points'].shape),
               'gt_points': 81 if 'gt_path' in row else 0,
               'supplied': sum(len(row[k]) for k in FIELDS.values()),
               'retained': int(encoded['polyline_mask'].sum())}
    return row, encoded, summary


def import_project(raw):
    if not isinstance(raw, dict):
        raise ValueError('JSON 顶层必须是对象')
    if raw.get('version') == 1 and isinstance(raw.get('entities'), list):
        # Upgrade old read-only GT into an editable entity, without changing the source.
        if raw.get('gt_path') is not None:
            import copy
            raw = copy.deepcopy(raw)
            if any(e.get('kind') == 'gt' for e in raw['entities']):
                raise ValueError('旧 gt_path 与 GT 元素重复')
            eid = 'gt_1'
            while any(e.get('id') == eid for e in raw['entities']): eid += '_1'
            raw['entities'].append({'id': eid, 'kind': 'gt', 'mode': 'polyline',
                                    'points': points(raw.pop('gt_path'))})
        # Allow incomplete drafts (e.g. a Guide still shorter than 80 m).
        if len(raw['entities']) > 256: raise ValueError('最多支持256个元素')
        ids = set()
        for e in raw['entities']:
            if e.get('kind') not in KINDS or not isinstance(e.get('id'), str) or e['id'] in ids:
                raise ValueError('无效元素类型或重复 ID')
            ids.add(e['id']);curve_points(e)
        if raw.get('navigation') not in ('left', 'straight', 'right'):
            raise ValueError('无效导航')
        if sum(e['kind'] == 'gt' for e in raw['entities']) > 1:
            raise ValueError('场景最多只能有一条 GT')
        return raw
    project = {'version': 1, 'name': str(raw.get('id', 'imported_scene')),
               'navigation': raw.get('navigation', raw.get('nav', 'straight')), 'entities': []}
    guide = raw.get('reference_line', raw.get('guide'))
    if guide is not None:
        project['entities'].append({'id': 'guide', 'kind': 'guide', 'mode': 'polyline', 'points': points(guide)})
    for kind, field in FIELDS.items():
        for i, obj in enumerate(raw.get(field, [])):
            project['entities'].append({'id': f'{kind}_{i+1}', 'kind': kind,
                'mode': 'polygon' if kind in ('stop', 'cross') else 'polyline',
                'points': points(obj.get('points'), 3 if kind in ('stop', 'cross') else 2),
                'confidence': obj.get('confidence', 1.), 'color': obj.get('color', 'unknown'),
                'line_style': obj.get('line_style', 'unknown')})
    if raw.get('gt_path') is not None: project['gt_path'] = points(raw['gt_path'])
    return import_project(project)


def render_scene_png(project, row=None):
    """Snapshot complete edited geometry, independent of the browser camera."""
    import io
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import font_manager
    from matplotlib.patches import Polygon
    colors = {'guide':'#e69a17','gt':'#16975d','line':'#407fe8','curb':'#475569','stop':'#d75f61','cross':'#2baca9'}
    with _PREVIEW_LOCK:
        font = font_manager.FontProperties(family=['Noto Sans CJK JP','DejaVu Sans'])
        fig = Figure(figsize=(10,8), dpi=150, facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.subplots(); ax.set_facecolor('#fbfcff')
        for entity in project['entities']:
            p = np.asarray(curve_points(entity)); kind=entity['kind']
            color = '#b9970c' if kind=='line' and entity.get('color')=='yellow' else colors[kind]
            if kind in ('stop','cross'):
                ax.add_patch(Polygon(np.c_[-p[:,1],p[:,0]],facecolor=color+'22',edgecolor=color,lw=1.3))
                ax.update_datalim(np.c_[-p[:,1],p[:,0]])
            else:
                dashed = kind=='guide' or kind=='line' and entity.get('line_style')=='dashed'
                ax.plot(-p[:,1],p[:,0],color=color,ls='--' if dashed else '-',lw=1.6,
                        alpha=.5 if kind=='guide' and row is not None else 1)
            middle = p[len(p)//2]
            ax.annotate(entity['id'],(-middle[1],middle[0]),xytext=(4,4),textcoords='offset points',
                        fontsize=7,color=color,fontproperties=font)
        if row is not None:
            p=np.asarray(row['reference_line'])
            ax.plot(-p[:,1],p[:,0],color=colors['guide'],ls='--',lw=2.1,label='Guide: first 80 m')
            if row.get('gt_path') is not None:
                p=np.asarray(row['gt_path'])
                ax.plot(-p[:,1],p[:,0],color=colors['gt'],lw=2.1,label='GT: first 80 m (label only)')
        if project.get('gt_path'):
            p=np.asarray(project['gt_path']);ax.plot(-p[:,1],p[:,0],color='#16975d',lw=1.5,label='Reference target')
        ax.scatter([0],[0],marker='^',s=50,color='#253951',label='Ego (0, 0)',zorder=6)
        ax.axhline(0,color='#d8e1ee',lw=.7);ax.axvline(0,color='#d8e1ee',lw=.7)
        ax.autoscale_view();ax.margins(.1)
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.22)
        ax.set_xlabel('Right (-ego y), m');ax.set_ylabel('Forward (ego x), m')
        ax.set_title(f"{project.get('name','scene')} | nav={project.get('navigation','unknown')} | {len(project['entities'])} elements",fontproperties=font)
        ax.legend(loc='best',fontsize=8);fig.tight_layout()
        data=io.BytesIO();fig.savefig(data,format='png',dpi=150);fig.clear()
        return data.getvalue()


def update_gallery(output_dir):
    """Offline contact sheet for browsing, importing and manually reviewing saves."""
    import html
    output_dir=Path(output_dir)
    with _PREVIEW_LOCK:
        cards=[]
        for preview in sorted(output_dir.glob('*/scene.png'),reverse=True):
            folder=preview.parent
            try:project=json.loads((folder/'scene.json').read_text())
            except (OSError,ValueError):continue
            name=html.escape(str(project.get('name',folder.name)))
            nav=html.escape(str(project.get('navigation','unknown')))
            from urllib.parse import quote
            rel=quote(folder.name)
            cards.append(f'<article><a href="{rel}/scene.png"><img loading="lazy" src="{rel}/scene.png" alt="{name}"></a><b>{name}</b> · {nav}<p>{html.escape(folder.name)}</p><a href="{rel}/scene.json">编辑工程</a> · <a href="{rel}/input.json">模型输入</a></article>')
        content='<!doctype html><meta charset="utf-8"><title>BEV场景存档</title><style>body{font:14px system-ui;margin:24px;color:#26394c;background:#f5f7fb}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:18px}article{background:white;border:1px solid #dbe2ed;border-radius:8px;padding:12px}img{width:100%;height:240px;object-fit:contain}p{color:#6f8098}</style><h1>BEV 场景存档</h1><p>点击图片查看大图；可下载编辑工程重新导入。相似场景请人工核对后处理，不自动删除。</p><div class="grid">'+''.join(cards)+'</div>'
        temporary=output_dir/'.gallery.tmp'
        temporary.write_text(content);temporary.replace(output_dir/'gallery.html')
        return len(cards)


def backfill_previews(output_dir):
    output_dir=Path(output_dir);created=0;errors=[]
    for source in sorted(output_dir.glob('*/scene.json')):
        preview=source.parent/'scene.png'
        if preview.exists():continue
        try:
            project=import_project(json.loads(source.read_text()))
            input_path=source.parent/'input.json'
            row=json.loads(input_path.read_text()) if input_path.exists() else None
            data=render_scene_png(project,row)
            # Exclusive create: never overwrite a concurrent save/backfill.
            with preview.open('xb') as stream:stream.write(data)
            created+=1
        except FileExistsError:continue
        except Exception as error:errors.append({'scene':str(source),'error':str(error)})
    count=update_gallery(output_dir) if output_dir.exists() else 0
    return {'created':created,'gallery_scenes':count,'errors':errors}


def save_project(project, output_dir):
    import numpy as np
    row, encoded, summary = export_project(project)
    preview = render_scene_png(project, row)
    folder = output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'input.json').write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (folder/'scene.json').write_text(json.dumps(project, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(folder/'model_input.npz', **{k: v.numpy() for k, v in encoded.items()})
    (folder/'scene.png').write_bytes(preview)
    update_gallery(output_dir)
    path = folder/'input.json'
    import shlex
    command = ('python infer.py --checkpoint outputs/training_road_mixed_v2/best.pt '
               f'--input {shlex.quote(str(path.resolve()))} --device cuda')
    return {'input_path': str(path.resolve()), 'scene_path': str((folder/'scene.json').resolve()),
            'encoded_path': str((folder/'model_input.npz').resolve()),
            'preview_path': str((folder/'scene.png').resolve()),
            'preview_url': f'/api/scene-image/{folder.name}.png',
            'gallery_path': str((output_dir/'gallery.html').resolve()),
            'command': command, 'summary': summary, 'input': row}


class Handler(BaseHTTPRequestHandler):
    output_dir = ROOT/'outputs/scene_editor'

    def send(self, status, value, mime='application/json; charset=utf-8'):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path == '/api/health':
            return self.send(200, {'ok': True, 'editor_version': EDITOR_VERSION})
        match = re.fullmatch(r'/api/scene-image/(\d{8}_\d{6}_\d{6})\.png', self.path)
        if match:
            path = (self.output_dir/match[1]/'scene.png').resolve()
            if not path.is_relative_to(self.output_dir.resolve()) or not path.is_file():
                return self.send(404, {'error': '场景图不存在'})
            return self.send(200, path.read_bytes(), 'image/png')
        if self.path == '/api/example':
            return self.send(200, import_project(json.loads((ROOT/'examples/custom_input_split.json').read_text())))
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/editor.js': ('editor.js', 'text/javascript; charset=utf-8'),
                  '/measurement.js': ('measurement.js', 'text/javascript; charset=utf-8'),
                  '/geometry.js': ('geometry.js', 'text/javascript; charset=utf-8'),
                  '/editor.css': ('editor.css', 'text/css; charset=utf-8')}
        if self.path not in assets: return self.send(404, {'error': 'Not found'})
        name, mime = assets[self.path]
        self.send(200, (ASSETS/name).read_bytes(), mime)

    def do_POST(self):
        try:
            # Only same-origin browser requests may modify this local editor.
            if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
                return self.send(403, {'error': '请求来源与编辑器不一致'})
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4*1024*1024: raise ValueError('请求为空或大于4MB')
            raw = json.loads(self.rfile.read(size))
            if self.path == '/api/import': return self.send(200, import_project(raw))
            if self.path == '/api/save': return self.send(200, save_project(raw, self.output_dir))
            if self.path == '/api/preview':
                row, encoded, summary = export_project(raw)
                lines = []
                for i, valid in enumerate(encoded['polyline_mask']):
                    if valid:
                        mask = encoded['point_mask'][i]
                        lines.append({'kind': int(encoded['polyline_attr'][i, :4].argmax()),
                                      'points': encoded['bev_points'][i, mask, :2].tolist()})
                return self.send(200, {'guide': row['reference_line'], 'gt': row.get('gt_path'), 'lines': lines, 'summary': summary})
            return self.send(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError, OverflowError) as e:
            self.send(400, {'error': str(e)})
        except Exception as e:
            self.send(500, {'error': f'保存/编码失败：{e}'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/scene_editor')
    parser.add_argument('--open', action='store_true', help='Open browser automatically')
    parser.add_argument('--background', action='store_true', help='Run independently of the terminal, with a log file')
    parser.add_argument('--backfill-previews', action='store_true', help='Add missing scene.png to existing saves and rebuild offline gallery, then exit')
    args = parser.parse_args()
    if args.backfill_previews:
        print(json.dumps(backfill_previews(args.output_dir), ensure_ascii=False, indent=2))
        return
    if args.background:
        import subprocess, sys, time, urllib.request
        url = f'http://127.0.0.1:{args.port}'
        # Check the port before spawning so an existing editor is not replaced.
        import socket
        with socket.socket() as probe:
            occupied = probe.connect_ex(('127.0.0.1', args.port)) == 0
        if occupied:
            print(f'端口 {args.port} 已有服务：{url}；如需加载新版后端，请先退出旧服务。')
            return
        folder = args.output_dir.resolve()/('server_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        folder.mkdir(parents=True)
        with (folder/'server.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--port', str(args.port), '--output-dir', str(args.output_dir.resolve())], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        (folder/'server.pid').write_text(str(process.pid)+'\n')
        for _ in range(50):
            if process.poll() is not None:
                raise RuntimeError(f'服务启动失败，请查看 {folder}/server.log')
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url+'/api/health', timeout=.3) as response:
                    if json.load(response).get('ok'): break
            except (OSError, ValueError): time.sleep(.1)
        else: raise RuntimeError(f'服务尚未就绪，请查看 {folder}/server.log')
        print(f'编辑器后台运行：{url}\nPID：{process.pid}\n日志：{folder}/server.log')
        if args.open: webbrowser.open(url)
        return
    Handler.output_dir = args.output_dir.resolve()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'BEV 场景编辑器：{url}\n保存位置：{Handler.output_dir}\n按 Ctrl+C 退出。', flush=True)
    if args.open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
