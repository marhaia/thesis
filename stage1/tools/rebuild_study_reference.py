#!/usr/bin/env python3
"""Rebuild the ViCoP GUI reference for the final 75-stimulus study pool.

Produces candidate artifacts in a separate output directory. Fails closed on
missing inputs, unexpected hashes, changed per-image features, ambiguous
original identities, or any study-image overlap with reference rows.

Uses previously calculated v8/s5 rows. No model training, no new feature
extraction, no score recalculation, and no writes to the production repository.
Compatible with Python 3.9+ / NumPy 1.26+. The official release replay must be
performed separately in the frozen project environment.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
from typing import Mapping

import numpy as np

VISUAL = ("shannon_entropy", "edge_density", "feature_congestion", "subband_entropy",
          "layout_symmetry", "chromatic_coherence", "visual_hierarchy", "interactive_element_density")
SALIENCY = ("saliency_dispersion", "saliency_peak_count", "saliency_center_bias", "saliency_entropy", "saliency_coverage")
CATEGORIES = ("desktop", "mobile", "web")
SHA = {
    "official_split": "fc22d679849862c0f0476b1d9c270a50383c62304c7635b87cb18a2559a25c68",
    "visual_rows": "b9769935527a58cf7c9e158a6d3ed0bcd33edaa37aa3dcc40b3ed273c453804a",
    "saliency_rows": "1f576987fa080bbc7c34062ce1dde86c41c16eaa1fd17be9e00adac4e6a01638",
    "official_train_population": "4c90612e1d228d8ea8d0e2d6ad1dccb6db8a67ff9180569545e11584071b1801",
    "official_test_population": "1d98dd2072f99e2bfbdcc5baa46517d5d1bb80fdd113956b30139dc86607f6a8",
    "final_stimulus_zip": "6bbf0fcfdb32e00181a8a7c05a3c84a7fb95e018a1344e2e27ff6d48b8e6c154",
    "final_stimulus_manifest": "e77faface786f339d28bf9683cf4b6b0981ce4abe3c7787f0bf48d40c21628e8",
    "UMSI_weights": "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe",
}
ROOT_PREFIX = "Stimulus_Pool_Final_75_2026-09-30/"
OUT_SCHEMA = "ueyes-study-isolated-reference-v1"

class IntegrityError(RuntimeError):
    pass

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def require(actual, expected, detail):
    if actual != expected:
        raise IntegrityError("%s: expected %r, found %r" % (detail, expected, actual))

def parse_csv_bytes(data: bytes, *, delimiter=","):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"),newline=""), delimiter=delimiter))

def json_bytes(p):
    return (json.dumps(p,indent=2,ensure_ascii=False)+"\n").encode("utf-8")

def write(path: Path, data: bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise IntegrityError("Output already exists (refuse overwrite): %s"%path)
    path.write_bytes(data)

def population_hash(names, visual):
    lines=["%s|%s|%s"%(n,visual[n]["category"],visual[n]["image_sha256"]) for n in names]
    return digest(("\n".join(sorted(lines))+"\n").encode())

def stats(rows, names, features):
    output={}
    for feature in features:
        try:
            x=np.asarray([float(rows[n][feature]) for n in names],dtype=np.float64)
        except (ValueError,TypeError,KeyError) as exc:
            raise IntegrityError("Malformed feature %s: %s"%(feature,exc)) from exc
        if x.size!=len(names) or x.size<2 or not np.isfinite(x).all():
            raise IntegrityError("Nonfinite/missing %s"%feature)
        p5,p25,p50,p75,p95=(float(v) for v in np.percentile(x,[5,25,50,75,95],method="linear"))
        output[feature]={"n":len(names),"mean":float(np.mean(x)),"std":float(np.std(x,ddof=1)),
            "min":float(np.min(x)),"max":float(np.max(x)),
            "p5":p5,"p25":p25,"p50":p50,"p75":p75,"p95":p95}
    return output

def canonical_study_manifest(z: zipfile.ZipFile):
    keys=[n for n in z.namelist() if n.endswith('/04_documentation/final_pool_manifest.json') and not n.startswith('__MACOSX/')]
    require(len(keys),1,"final study manifest count")
    data=z.read(keys[0]); require(digest(data),SHA['final_stimulus_manifest'],"final study manifest sha")
    manifest=json.loads(data)
    require(manifest.get("counts",{}).get("total"),75,"final stimuli count")
    require(manifest.get("counts",{}).get("main"),72,"main stimuli count")
    require(manifest.get("counts",{}).get("practice"),3,"practice stimuli count")
    return manifest

def run(*,source_zip:Path,stimuli_zip:Path,split_csv:Path,exclusions_json:Path,
        baseline_json:Path,output_dir:Path):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise IntegrityError("Output directory must be empty (no overwrite): %s"%output_dir)
    require(digest(split_csv.read_bytes()),SHA['official_split'],"official image_types CSV")
    require(digest(stimuli_zip.read_bytes()),SHA['final_stimulus_zip'],"study ZIP")
    official_rows=parse_csv_bytes(split_csv.read_bytes(),delimiter=';')
    split={x['Image Name']:x for x in official_rows if x.get('Category','').lower() in CATEGORIES}
    require(len(split),1485,"official authorized GUI count")
    require(len(official_rows),1980,"official total GUI/poster count")
    require(len(set(x['Image Name'] for x in official_rows)),1980,"unique official original names")
    for cat in CATEGORIES:
        require(sum(x['Category'].lower()==cat and x['Train/Test']=='Train' for x in split.values()),468,cat+' train')
        require(sum(x['Category'].lower()==cat and x['Train/Test']=='Test' for x in split.values()),27,cat+' test')

    with zipfile.ZipFile(source_zip) as z:
        required=['canonical_visual_feature_rows.csv','saliency_feature_rows.csv',
                  'canonical_visual_feature_rows.csv.provenance.json','saliency_feature_rows.csv.provenance.json']
        require(set(z.namelist()),set(required),"source files included")
        visual_raw=z.read(required[0]);saliency_raw=z.read(required[1])
        require(digest(visual_raw),SHA['visual_rows'],"source v8 CSV")
        require(digest(saliency_raw),SHA['saliency_rows'],"source s5 CSV")
        vprov_raw=z.read(required[2]); sprov_raw=z.read(required[3])
        vp=json.loads(vprov_raw); sp=json.loads(sprov_raw)
    require(vp['status'],'completed',"visual provenance status")
    require(sp['status'],'completed',"saliency provenance status")
    require(vp['run_fingerprint']['image_types_csv_sha256'],SHA['official_split'],"visual official split identity")
    require(sp['run_fingerprint']['weights_sha256'],SHA['UMSI_weights'],"saliency UMSI checkpoint identity")
    visual_rows=parse_csv_bytes(visual_raw); saliency_rows=parse_csv_bytes(saliency_raw)
    require(len(visual_rows),1485,"v8 row count");require(len(saliency_rows),1485,"s5 row count")
    vr={r['filename']:r for r in visual_rows}; sr={r['relative_path']:r for r in saliency_rows}
    require(len(vr),1485,"v8 uniqueness");require(len(sr),1485,"s5 uniqueness")
    require(set(vr),set(split),"v8 original names")
    require(set(sr),set(split),"s5 original names")
    for filename in sorted(split):
        cat=split[filename]['Category'].lower()
        require(vr[filename]['category'],cat,"v8 category "+filename)
        require(sr[filename]['category'],cat,"s5 category "+filename)
        require(vr[filename]['image_sha256'],sr[filename]['image_sha256'],"v8 s5 SHA "+filename)
        require(len(vr[filename]['image_sha256']),64,"source image hash length "+filename)
        require(vr[filename]['analysis_domain'],'canonical',"v8 domain "+filename)
        require(vr[filename]['canonical_long_side'],'1280',"v8 resolution "+filename)
        require(vr[filename]['run_fingerprint_sha256'],vp['run_fingerprint_sha256'],"v8 fingerprint "+filename)
        require(sr[filename]['run_fingerprint_sha256'],sp['run_fingerprint_sha256'],"s5 fingerprint "+filename)
    official_train=sorted(n for n in split if split[n]['Train/Test']=='Train')
    official_test=sorted(n for n in split if split[n]['Train/Test']=='Test')
    require(population_hash(official_train,vr),SHA['official_train_population'],"official Train population")
    require(population_hash(official_test,vr),SHA['official_test_population'],"official Test population")
    hashes_to_names=collections.defaultdict(list)
    for filename in sorted(vr):
        hashes_to_names[vr[filename]['image_sha256']].append(filename)

    with zipfile.ZipFile(stimuli_zip) as z:
        study_manifest=canonical_study_manifest(z)
        study_stimuli=study_manifest['stimuli']
        require(len(study_stimuli),75,"study manifest number of stimuli")
        resolved=[]
        for stim in study_stimuli:
            sid=stim['stimulus_id']; role=stim['role']; category=stim['platform']
            require(role in ('main','practice'),True,"role "+sid)
            require(category in CATEGORIES,True,"category "+sid)
            sha=stim['source_sha256']
            require(sha,stim['packaged_sha256'],"unchanged study package image "+sid)
            zpath=next((n for n in z.namelist() if n.endswith('/'+stim['package_path']) and not n.startswith('__MACOSX/')),None)
            if zpath is None:
                raise IntegrityError('Missing study stimulus bytes: '+sid)
            require(digest(z.read(zpath)),sha,'packaged file bytes '+sid)
            original=hashes_to_names.get(sha,[])
            require(len(original),1,"unique original UEyes identity "+sid)
            original=original[0]
            require(split[original]['Category'].lower(),category,"study platform original platform "+sid)
            resolved.append({'stimulus_id':sid,'role':role,'category':category,'original_ueyes_filename':original,
                'original_split':split[original]['Train/Test'],
                'original_block':split[original]['Block'],
                'image_sha256':sha,'packaged_path':stim['package_path']})
    require(len({x['stimulus_id'] for x in resolved}),75,"unique Study IDs")
    require(len({x['image_sha256'] for x in resolved}),75,"no byte-duplicate study screenshots")
    counts=collections.Counter((x['category'],x['original_split']) for x in resolved)
    for cat in CATEGORIES:
        require(counts[(cat,'Train')],8,"Study exclusions per category "+cat)
        require(counts[(cat,'Test')],17,"Study Test per category "+cat)
    require(sum(x['original_split']=='Train' for x in resolved),24,"Train overlap")
    require(sum(x['original_split']=='Test' for x in resolved),51,"Test split count")
    require(sum(x['role']=='practice' and x['original_split']=='Train' for x in resolved),2,"practice exclusions")
    exc_payload=exclusions_json.read_bytes();exc=json.loads(exc_payload)
    require(exc['status'],'VERIFIED_EXCLUSION_SET__NOT_APPLIED_TO_PRODUCTION_NORMS',"exclusion input status")
    require(exc['study_pool_zip_sha256'],SHA['final_stimulus_zip'],"exclusion study zip identity")
    require(exc['study_pool_manifest_sha256'],SHA['final_stimulus_manifest'],"exclusion manifest identity")
    require(exc['official_split_file_sha256'],SHA['official_split'],"exclusion official split identity")
    expected={x['stimulus_id']:x for x in exc['exclusions']}
    overlap=[x for x in resolved if x['original_split']=='Train']
    require(len(expected),24,"exclusion source count")
    require(set(expected),{x['stimulus_id'] for x in overlap},"full study exclusion set")
    for r in overlap:
        src=expected[r['stimulus_id']]
        for key in ('stimulus_id','role','category','original_ueyes_filename','image_sha256','packaged_path'):
            require(src[key],r[key],"original exclusion manifest "+r['stimulus_id']+" "+key)

    heldout_names={x['original_ueyes_filename'] for x in resolved}
    new_train=sorted(set(official_train)-heldout_names)
    require(len(new_train),1380,"final study-isolated ref count")
    require(set(new_train).isdisjoint(heldout_names),True,"no study stimuli in reference")
    for cat in CATEGORIES:
        require(sum(split[n]['Category'].lower()==cat for n in new_train),460,"post-exclusion ref "+cat)

    base=json.loads(baseline_json.read_text(encoding='utf8'))
    require(set(base['features']),set(VISUAL+SALIENCY),"reference baseline feature coverage")
    old_v=stats(vr,official_train,VISUAL); old_s=stats(sr,official_train,SALIENCY)
    old_stats={**old_v,**old_s}
    keys=['n','mean','std','min','max','p5','p25','p50','p75','p95']
    max_baseline_absolute_error=0.0
    for key,stat in old_stats.items():
        for k,expected in zip(keys,base['features'][key]):
            error=abs(stat[k]-float(expected))
            max_baseline_absolute_error=max(max_baseline_absolute_error,error)
            if error>1e-12:
                raise IntegrityError("Baseline reproduction discrepancy %s/%s: %.12g"%(key,k,error))
    new_v=stats(vr,new_train,VISUAL);new_s=stats(sr,new_train,SALIENCY)
    new_stats={**new_v,**new_s}
    new_pop_sha=population_hash(new_train,vr)
    exclusion_identity=digest(exc_payload)
    refcounts={cat:460 for cat in CATEGORIES}
    testcounts={cat:27 for cat in CATEGORIES}
    originalcounts={cat:468 for cat in CATEGORIES}
    norms_common={
        'official_split_csv_sha256':SHA['official_split'],
        'reference_partition':'Train', 'reference_selection_policy':'official_Train_minus_final_study_pool_2026-09-30',
        'evaluation_partition':'Test','reference_category_counts':refcounts,'reference_total':1380,
        'original_train_total':1404,'original_train_category_counts':originalcounts,
        'evaluation_category_counts':testcounts,'evaluation_total':81,
        'study_stimuli_total':75,'study_stimuli_train_excluded':24,
        'study_stimuli_train_excluded_per_category':{cat:8 for cat in CATEGORIES},
        'study_stimuli_practice_train_excluded':2,
        'final_study_manifest_sha256':SHA['final_stimulus_manifest'],
        'study_exclusion_manifest_sha256':exclusion_identity,
        'excluded_categories':['poster'],
        'development_population_sha256':new_pop_sha,
        'original_train_population_sha256':SHA['official_train_population'],
        'heldout_test_population_sha256':SHA['official_test_population'],
        'standard_deviation':{'convention':'sample standard deviation','ddof':1},
        'stored_quantiles':{'implementation':'numpy.percentile','numpy_version_contract':'1.26.4',
                            'actual_reaggregation_numpy_version':np.__version__,
                            'method':'linear','probabilities_percent':[5,25,50,75,95]},
    }
    visual={
        'schema_version':'canonical-visual-feature-norms-v2', 'analysis_domain':'canonical',
        'canonical_long_side':1280,
        'canonical_analysis_version':'canonical-analysis-v1.1:long1280:both-dims-min16:area-down/linear-up',
        'feature_list':list(VISUAL),'includes_saliency':False,'num_images':1380,
        **norms_common,
        'source_rows':{'filename':'canonical_visual_feature_rows.csv','sha256':SHA['visual_rows'],'row_count':1485},
        'features':new_v,
    }
    saliency={
        'schema_version':'canonical-saliency-feature-norms-v2',
        'analysis_domain':'native_input','resolution_policy':'NATIVE_INPUT',
        'feature_order':list(SALIENCY),'num_images':1380,
        **norms_common,
        'source_rows':{'filename':'saliency_feature_rows.csv','sha256':SHA['saliency_rows'],'row_count':1485},
        'features':new_s,
    }
    out=output_dir
    result=out/'stage1/data/results'
    vbytes=json_bytes(visual); sbytes=json_bytes(saliency)
    combined={
        'features':new_stats,
        'meta':{
            'schema_version':OUT_SCHEMA,
            'source':'UEyes dataset (Jiang et al., CHI 2023; Zenodo 8010312)',
            'categories_included':list(CATEGORIES),'categories_excluded':['poster'],
            'reference_partition':'Train','reference_selection_policy':'official_Train_minus_final_study_pool_2026-09-30',
            'evaluation_partition_excluded':'Test', 'category_counts':refcounts,'num_images':1380,
            'original_train_category_counts':originalcounts,'original_train_total':1404,
            'includes_saliency':True, 'is_subsample':True,
            'reference_population_note':('Corpus-relative references computed exclusively from the 1,404 official UEyes Train GUI images '
                'after excluding the 24 final Main/Practice study images in that partition. The entire official Test split '
                'of 81 GUI images is held out. No study image contributes to reference statistics. This is a study-specific '
                'nonrandom exclusion, not external or human cognitive-load norms.'),
            'official_split_csv_sha256':SHA['official_split'],
            'development_population_sha256':new_pop_sha,
            'original_train_population_sha256':SHA['official_train_population'],
            'heldout_test_population_sha256':SHA['official_test_population'],
            'study_pool_manifest_sha256':SHA['final_stimulus_manifest'],
            'study_exclusion_manifest_sha256':exclusion_identity,
            'study_pool_count':75,'study_train_excluded_count':24,
            'study_excluded_categories':{cat:8 for cat in CATEGORIES},
            'visual_norms_provenance':{
                'source_artifact_filename':'canonical_visual_feature_norms.json',
                'source_artifact_sha256':digest(vbytes),
                'source_csv_filename':'canonical_visual_feature_rows.csv','source_csv_sha256':SHA['visual_rows'],
                'analysis_domain':'canonical','canonical_long_side':1280,'processed_count':1380,
                'processed_category_counts':refcounts},
            'saliency_norms_provenance':{
                'source_artifact_filename':'development_saliency_feature_norms.json',
                'source_artifact_sha256':digest(sbytes),
                'source_csv_filename':'saliency_feature_rows.csv','source_csv_sha256':SHA['saliency_rows'],
                'analysis_domain':'native_input','resolution_policy':'NATIVE_INPUT',
                'processed_count':1380,'processed_category_counts':refcounts},
            'stimulus_selection_provenance':{
                'schema_version':'final-stimulus-pool-v2', 'study_pool_count':75,
                'main_count':72,'practice_count':3,
                'selection_manifest_sha256':SHA['final_stimulus_manifest'],
                'exclusion_manifest_sha256':exclusion_identity,
                'eligibility_review':'separate from this reference-isolation calculation'},
        }}
    cbytes=json_bytes(combined)
    drift={'schema_version':'vicop-reference-drift-train1404-to-study1380-v1',
           'comparison':'official_Train_1404 minus final_Study_Train_24',
           'old_n':1404,'new_n':1380,'features':{}}
    for feature in VISUAL+SALIENCY:
        f={}
        for k in keys[1:]:
            old=float(old_stats[feature][k]);new=float(new_stats[feature][k]);delta=new-old
            f[k]={'old':old,'new':new,'absolute_delta':delta,
                  'relative_delta_percent':(100.0*delta/old if old else None)}
        drift['features'][feature]=f
    driftbytes=json_bytes(drift)
    resolved.sort(key=lambda x:(CATEGORIES.index(x['category']),x['stimulus_id']))
    csv_buffer=io.StringIO(newline='')
    fields=list(resolved[0])
    w=csv.DictWriter(csv_buffer,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(resolved)
    overlapbytes=csv_buffer.getvalue().encode()
    report={
       'schema_version':'vicop-study-isolation-verification-v1',
       'production_state':'CANDIDATE_ONLY__NO_PRODUCTION_MUTATION',
       'official_split_sha256':SHA['official_split'],
       'source_zip_sha256':digest(source_zip.read_bytes()),
       'source_visual_csv_sha256':SHA['visual_rows'],
       'source_saliency_csv_sha256':SHA['saliency_rows'],
       'source_visual_provenance_sha256':digest(vprov_raw),
       'source_saliency_provenance_sha256':digest(sprov_raw),
       'source_visual_fingerprint_sha256':vp['run_fingerprint_sha256'],
       'source_saliency_fingerprint_sha256':sp['run_fingerprint_sha256'],
       'final_stimulus_zip_sha256':SHA['final_stimulus_zip'],
       'final_stimulus_manifest_sha256':SHA['final_stimulus_manifest'],
       'study_exclusion_manifest_sha256':exclusion_identity,
       'original_train_population_sha256':SHA['official_train_population'],
       'new_development_population_sha256':new_pop_sha,
       'official_test_population_sha256':SHA['official_test_population'],
       'baseline_reproduction':{'features_verified':13,'statistics_per_feature':10,
                                'max_absolute_error':max_baseline_absolute_error,'tolerance':1e-12},
       'official_corpus':{'GUI_authorized':1485,'Train':1404,'Test':81},
       'final_stimuli':{'total':75,'main':72,'practice':3,'official_Test':51,
                        'official_Train_excluded':24,'Train_main':22,'Train_practice':2},
       'new_reference':{'total':1380,'counts':refcounts,'overlap_with_final_stimuli':0},
       'model_checkpoint_sha256_unchanged':SHA['UMSI_weights'],
       'analysis_limitation':('Only reference statistics were recalculated. HCEye intermediate heuristics, '
                'Stage-1 x19 and final layout-index regression on real screenshots not yet run; '
                'separate frozen runtime release verification required.'),
       'numpy_reaggregate_version':np.__version__,
    }
    files_to_write={
        result/'canonical_visual_feature_norms.json':vbytes,
        result/'development_saliency_feature_norms.json':sbytes,
        result/'feature_norms.json':cbytes,
        result/'reference_norm_drift_1404_to_1380.json':driftbytes,
        result/'study_stimulus_split_75.csv':overlapbytes,
        out/'checks/reference_isolation_verification.json':json_bytes(report),
    }
    manifest={
        'schema_version':OUT_SCHEMA,'status':'CANDIDATE__PENDING_STAGE1_RELEASE_REGRESSION',
        'claim_boundary':'Study-specific UEyes GUI feature distribution, not a cognitive-load norm or independent UMSI++ model test.',
        'reference_partition':'Train','reference_exclusion_policy':'final_study_Main_plus_Practice',
        'original_train_total':1404,'reference_total':1380,'reference_counts':refcounts,
        'heldout_test_total':81,'heldout_test_population_sha256':SHA['official_test_population'],
        'study_stimuli_total':75,'study_train_excluded_count':24,'study_practice_train_excluded':2,
        'excluded_stimuli_per_category':{cat:8 for cat in CATEGORIES},
        'official_split_csv_sha256':SHA['official_split'],
        'original_train_population_sha256':SHA['official_train_population'],
        'reaggregation_script_sha256':digest(Path(__file__).read_bytes()),
        'development_population_sha256':new_pop_sha,
        'input_hashes':{
            'visual_csv':SHA['visual_rows'],'saliency_csv':SHA['saliency_rows'],
            'source_zip':digest(source_zip.read_bytes()),
            'stimulus_zip':SHA['final_stimulus_zip'],
            'stimulus_manifest':SHA['final_stimulus_manifest'],
            'exclusions_json':exclusion_identity,
        },
        'frozen_conventions':{'numpy_target':'1.26.4','reaggregation_numpy_actual':np.__version__,
            'standard_deviation_ddof':1,'quantile_method':'linear',
            'visual_domain':'canonical_long_side_1280','saliency_domain':'native_input'},
        'baseline_reproduction_max_absolute_error':max_baseline_absolute_error,
        'outputs':{str(p.relative_to(result)):{'sha256':digest(b),'bytes':len(b)}
            for p,b in files_to_write.items() if p.is_relative_to(result)},
    }
    mpath=result/'study_isolated_reference_manifest.json';files_to_write[mpath]=json_bytes(manifest)
    output_dir.mkdir(parents=True,exist_ok=True)
    for filename,content in files_to_write.items():write(filename,content)
    print('PASS: baseline 13 features reproduce original GitHub norms, tolerance 1e-12')
    print('PASS: 75 final stimuli verified by bytes; 51 Test and 24 Train, no ambiguity')
    print('PASS: final reference 1,380 rows (460 desktop, 460 mobile, 460 web) with no study overlap')
    print('CANDIDATE artifacts:',str(output_dir))
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-zip',type=Path,required=True)
    ap.add_argument('--stimuli-zip',type=Path,required=True)
    ap.add_argument('--split-csv',type=Path,required=True)
    ap.add_argument('--exclusions-json',type=Path,required=True)
    ap.add_argument('--baseline-json',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    p=ap.parse_args()
    try:
        run(source_zip=p.source_zip,stimuli_zip=p.stimuli_zip,split_csv=p.split_csv,
            exclusions_json=p.exclusions_json,baseline_json=p.baseline_json,output_dir=p.output_dir)
    except (IntegrityError,KeyError,ValueError,AssertionError,zipfile.BadZipFile) as e:
        raise SystemExit('FAIL CLOSED: '+str(e))
if __name__=='__main__':main()
