"""Frozen 75-stimulus reference-isolation contract (no model inference)."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'stage1/data/results'

def _read(filename):
    return json.loads((RESULTS/filename).read_text(encoding='utf8'))

def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_reference_excludes_all_study_images_at_recorded_population_boundary():
    manifest=_read('study_isolated_reference_manifest.json')
    combined=_read('feature_norms.json')
    meta=combined['meta']
    assert manifest['status']=='CANDIDATE__PENDING_STAGE1_RELEASE_REGRESSION'
    assert manifest['original_train_total']==1404
    assert manifest['reference_total']==1380
    assert manifest['study_train_excluded_count']==24
    assert manifest['study_practice_train_excluded']==2
    assert manifest['heldout_test_total']==81
    assert manifest['reference_counts']=={'desktop':460,'mobile':460,'web':460}
    assert meta['reference_partition']=='Train'
    assert meta['reference_selection_policy']=='official_Train_minus_final_study_pool_2026-09-30'
    assert meta['development_population_sha256']==manifest['development_population_sha256']
    assert meta['original_train_population_sha256']==manifest['original_train_population_sha256']
    assert meta['development_population_sha256']!=meta['original_train_population_sha256']
    assert meta['num_images']==1380
    assert len(combined['features'])==13
    assert all(f['n']==1380 for f in combined['features'].values())

def test_exact_study_manifest_and_split_exclusion_provenance():
    with (RESULTS/'study_stimulus_split_75.csv').open(newline='',encoding='utf8') as src:
        rows=list(csv.DictReader(src))
    assert len(rows)==75
    assert len({r['stimulus_id'] for r in rows})==75
    assert len({r['image_sha256'] for r in rows})==75
    assert collections.Counter(r['original_split'] for r in rows)=={'Train':24,'Test':51}
    assert collections.Counter(r['category'] for r in rows if r['original_split']=='Train')=={
        'desktop':8,'mobile':8,'web':8}
    assert sum(r['role']=='practice' and r['original_split']=='Train' for r in rows)==2
    assert all(len(r['image_sha256'])==64 for r in rows)
    assert all(r['original_ueyes_filename'].endswith(('.jpg','.jpeg','.png')) for r in rows)
    manifest=_read('study_isolated_reference_manifest.json')
    assert _sha(RESULTS/'study_stimulus_train_exclusions.json')==manifest['input_hashes']['exclusions_json']

def test_study_isolation_manifest_pins_every_new_reference_artifact():
    manifest=_read('study_isolated_reference_manifest.json')
    for name,spec in manifest['outputs'].items():
        artifact=RESULTS/name
        assert artifact.stat().st_size==spec['bytes']
        assert _sha(artifact)==spec['sha256']

def test_reference_domains_and_source_hashes_are_preserved():
    combined=_read('feature_norms.json')
    visual=_read('canonical_visual_feature_norms.json')
    saliency=_read('development_saliency_feature_norms.json')
    assert visual['analysis_domain']=='canonical'
    assert visual['canonical_long_side']==1280
    assert saliency['analysis_domain']=='native_input'
    assert saliency['resolution_policy']=='NATIVE_INPUT'
    assert combined['meta']['visual_norms_provenance']['source_csv_sha256']=='b9769935527a58cf7c9e158a6d3ed0bcd33edaa37aa3dcc40b3ed273c453804a'
    assert combined['meta']['saliency_norms_provenance']['source_csv_sha256']=='1f576987fa080bbc7c34062ce1dde86c41c16eaa1fd17be9e00adac4e6a01638'
    assert visual['num_images']==1380 and saliency['num_images']==1380
    for feature,stats in visual['features'].items():
        assert stats==combined['features'][feature]
    for feature,stats in saliency['features'].items():
        assert stats==combined['features'][feature]

def test_reference_drift_reports_actual_change():
    drift=_read('reference_norm_drift_1404_to_1380.json')
    assert drift['old_n']==1404 and drift['new_n']==1380
    assert len(drift['features'])==13
    assert any(abs(v['mean']['absolute_delta'])>0 for v in drift['features'].values())
    check=_read('study_isolated_reference_verification.json')
    assert check['new_reference']['overlap_with_final_stimuli']==0
    assert check['baseline_reproduction']['max_absolute_error']<=1e-12
