import json
from pathlib import Path
import numpy as np
import pytest
from scene_editor import export_project, import_project, save_project, curve_points
from infer import prepare_sample, encode_input


def project():
    return {'version':1,'name':'test_scene','navigation':'right','entities':[
        {'id':'guide','kind':'guide','mode':'bezier','points':[[0,1.2],[28,1.2],[57,1.2],[85,1.2]]},
        {'id':'line','kind':'line','mode':'polyline','points':[[0,3],[20,3],[85,3]],
         'confidence':.42,'color':'yellow','line_style':'solid'},
        {'id':'curb','kind':'curb','mode':'bezier','points':[[0,-2],[25,-2],[55,-5],[85,-5]],'confidence':.8}]}


def test_export_preserves_guide_offset_and_matches_actual_encoder():
    row,encoded,summary=export_project(project())
    guide=np.asarray(row['reference_line'])
    np.testing.assert_allclose(guide[:,0],np.arange(81),atol=1e-10)
    np.testing.assert_allclose(guide[:,1],1.2)
    assert row['navigation']=='right' and summary['retained']==2
    reencoded=encode_input(prepare_sample(row),32,32)
    for k in encoded:np.testing.assert_array_equal(encoded[k].numpy(),reencoded[k].numpy())
    attr=encoded['polyline_attr'].numpy()
    line=attr[attr[:,0]>0][0]
    assert line[4]==pytest.approx(.42)
    assert line[5]==1 and line[6]==0 and line[7]==1 and line[8]==0


def test_save_produces_runnable_input_editable_source_and_exact_npz(tmp_path):
    data=save_project(project(),tmp_path)
    row=json.loads(open(data['input_path']).read());source=json.loads(open(data['scene_path']).read())
    assert source==project()
    encoded=encode_input(prepare_sample(row),32,32)
    with np.load(data['encoded_path']) as saved:
        assert set(saved.files)==set(encoded)
        for k in saved.files:np.testing.assert_array_equal(saved[k],encoded[k].numpy())
    from PIL import Image
    with Image.open(data['preview_path']) as image:
        assert image.format=='PNG' and image.width>=1000 and image.height>=800
        assert np.asarray(image.convert('RGB')).std()>10
    assert Path(data['gallery_path']).exists()
    assert Path(data['preview_path']).parent.name in Path(data['gallery_path']).read_text()
    assert '--input ' in data['command'] and data['input_path'] in data['command']


def test_short_guide_fails_before_writing_any_file(tmp_path):
    p=project();p['entities'][0]={'id':'guide','kind':'guide','mode':'polyline','points':[[0,0],[50,0]]}
    with pytest.raises(ValueError,match='至少 80'):save_project(p,tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_guide_is_sampled_by_arc_not_forward_coordinate():
    p=project();p['entities'][0]={'id':'guide','kind':'guide','mode':'polyline','points':[[0,0],[30,40],[90,40]]}
    row,_,_=export_project(p);g=np.asarray(row['reference_line'])
    np.testing.assert_allclose(g[50],[30,40],atol=1e-9)
    np.testing.assert_allclose(g[80],[60,40],atol=1e-9)


def test_smooth_curve_interpolates_endpoints_without_mutating_controls():
    e=project()['entities'][2];saved=json.loads(json.dumps(e));p=curve_points(e)
    assert p[0]==e['points'][0] and p[-1]==e['points'][-1] and e==saved
    assert len(p)>256


def test_import_existing_custom_scene_and_reexport():
    from pathlib import Path
    raw=json.loads((Path(__file__).resolve().parents[1]/'examples/custom_input_split.json').read_text())
    p=import_project(raw);row,_,_=export_project(p)
    assert row['navigation']==raw['navigation']
    assert len(row['reference_line'])==81
    assert len(row['lane_markings'])==len(raw['lane_markings'])
    assert len(row['road_boundaries'])==len(raw['road_boundaries'])
    assert import_project(p)==p


@pytest.mark.parametrize('bad', [float('nan'),-0.1,1.1,True])
def test_invalid_confidence_is_rejected(bad):
    p=project();p['entities'][1]['confidence']=bad
    with pytest.raises(ValueError,match='置信度'):export_project(p)


def test_closed_polygons_are_encoded_as_closed_shapes():
    p=project();p['entities'].append({'id':'stop','kind':'stop','mode':'polygon',
        'points':[[20,-2],[20,2],[20.4,2],[20.4,-2]],'confidence':.9})
    _,encoded,_=export_project(p)
    attr=encoded['polyline_attr'].numpy()
    assert attr[attr[:,2]>0][0,9]==1


def test_duplicate_guides_and_nonfinite_points_are_rejected():
    p=project();p['entities'].append({**p['entities'][0],'id':'second'})
    with pytest.raises(ValueError,match='只能有一条'):export_project(p)
    p=project();p['entities'][0]['points'][0][0]=float('inf')
    with pytest.raises(ValueError,match='有限'):export_project(p)


def test_backfill_adds_only_missing_previews_and_preserves_originals(tmp_path):
    from scene_editor import backfill_previews
    folder=tmp_path/'20260101_120000_000001';folder.mkdir()
    scene=folder/'scene.json';scene.write_text(json.dumps(project()))
    original=scene.read_bytes()
    first=backfill_previews(tmp_path)
    assert first['created']==1 and first['errors']==[]
    assert scene.read_bytes()==original
    png=(folder/'scene.png').read_bytes()
    second=backfill_previews(tmp_path)
    assert second['created']==0 and (folder/'scene.png').read_bytes()==png
    assert 'test_scene' in (tmp_path/'gallery.html').read_text()
