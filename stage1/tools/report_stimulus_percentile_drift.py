#!/usr/bin/env python3
"""Supplemental 75-stimulus numerical normalization drift (NOT final x19/J).

Reuses the exact piecewise-linear percentile calculation in stage1/app.py.
Computed from already-existing per-image v8/s5 features, not eye-tracking.
"""
import argparse,csv,json,zipfile,io,collections,hashlib
from pathlib import Path

FEATURES=('shannon_entropy','edge_density','feature_congestion','subband_entropy',
 'layout_symmetry','chromatic_coherence','visual_hierarchy','interactive_element_density',
 'saliency_dispersion','saliency_peak_count','saliency_center_bias','saliency_entropy','saliency_coverage')

def empirical_percentile(stats,value):
    anchors=[(stats['min'],0.0),(stats['p5'],5.0),(stats['p25'],25.0),
        (stats['p50'],50.0),(stats['p75'],75.0),(stats['p95'],95.0),(stats['max'],100.0)]
    if value<=anchors[0][0]:return 0.0
    if value>=anchors[-1][0]:return 100.0
    for (x0,p0),(x1,p1) in zip(anchors,anchors[1:]):
        if x0<=value<=x1:
            if x1==x0:return p1
            return p0+(p1-p0)*(value-x0)/(x1-x0)
    return 50.0

def run(srczip,splitcsv,studyzip,newdir,outdir):
    with zipfile.ZipFile(srczip) as z:
        v=list(csv.DictReader(io.StringIO(z.read('canonical_visual_feature_rows.csv').decode())))
        s=list(csv.DictReader(io.StringIO(z.read('saliency_feature_rows.csv').decode())))
    visual={r['filename']:r for r in v};saliency={r['relative_path']:r for r in s}
    bysha={r['image_sha256']:r['filename'] for r in v}
    train={r['Image Name'] for r in csv.DictReader(splitcsv.open(),delimiter=';') if r.get('Train/Test')=='Train' and r.get('Category','').lower() in ('desktop','mobile','web')}
    with zipfile.ZipFile(studyzip) as z:
        k=[x for x in z.namelist() if x.endswith('/04_documentation/final_pool_manifest.json') and not x.startswith('__MACOSX/')]
        study=json.loads(z.read(k[0]))['stimuli']
    norms=json.loads((newdir/'stage1/data/results/feature_norms.json').read_text())['features']
    names=sorted(train)
    import numpy as np
    old={}
    for f in FEATURES:
        dataset=visual if f in v[0] else saliency
        arr=np.asarray([float(dataset[n][f]) for n in names],dtype=np.float64)
        q=np.percentile(arr,[5,25,50,75,95],method='linear')
        old[f]=dict(zip(('min','p5','p25','p50','p75','p95','max'),(float(arr.min()),*map(float,q),float(arr.max()))))
    rows=[]
    for item in study:
        name=bysha[item['packaged_sha256']]
        for f in FEATURES:
            dataset=visual if f in visual[name] else saliency
            value=float(dataset[name][f])
            before=empirical_percentile(old[f],value)
            after=empirical_percentile(norms[f],value)
            rows.append({'stimulus_id':item['stimulus_id'],'role':item['role'],
                'feature':f,'raw_value':value,'old_percentile':before,'new_percentile':after,
                'delta_percentile_points':after-before})
    assert len(rows)==75*13
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'stimulus_normalization_drift_75x13.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    byfeat={}
    for feature in FEATURES:
        d=[r['delta_percentile_points'] for r in rows if r['feature']==feature]
        byfeat[feature]={'max_abs_percentile_point_change':max(abs(x) for x in d),
            'mean_abs_percentile_point_change':sum(abs(x) for x in d)/len(d),
            'mean_signed_percentile_point_change':sum(d)/len(d)}
    report={'schema_version':'vicop-study-stimulus-reference-percentile-drift-v1',
        'caveat':'Changes in 13 corpus-relative percentiles for 75 study stimuli; NOT recomputed HCEye h6 or final layout index J; does not establish cognitive-load validity.',
        'num_stimuli':75,'comparisons':len(rows),'summary_by_feature':byfeat,
        'overall_max_abs_percentile_point_change':max(abs(r['delta_percentile_points']) for r in rows),
        'overall_mean_abs_percentile_point_change':sum(abs(r['delta_percentile_points']) for r in rows)/len(rows)}
    (outdir/'stimulus_normalization_drift_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS 75 x 13; max absolute normalized-feature percentile shift:',report['overall_max_abs_percentile_point_change'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-zip',type=Path,required=True);p.add_argument('--split-csv',type=Path,required=True);p.add_argument('--stimuli-zip',type=Path,required=True);p.add_argument('--candidate-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.source_zip,a.split_csv,a.stimuli_zip,a.candidate_dir,a.output_dir)
