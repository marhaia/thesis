"""Explore HCEye OSF - get detailed file listings and download key files."""
import requests
import json
import os

headers = {'Accept': 'application/json'}
BASE = '/Users/Q682780/Thesis_G/hceye'
os.makedirs(BASE, exist_ok=True)

def list_folder(url, prefix=""):
    """Recursively list folder contents."""
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        items = r.json()['data']
        for item in items:
            name = item['attributes']['name']
            kind = item['attributes']['kind']
            size = item['attributes'].get('size', '?')
            if kind == 'folder':
                print(f"{prefix}[DIR] {name}/")
                folder_url = item['relationships']['files']['links']['related']['href']
                list_folder(folder_url, prefix + "  ")
            else:
                print(f"{prefix}[FILE] {name} ({size} bytes)")

def download_file(url, filepath):
    """Download a file from OSF."""
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    if r.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f"  Downloaded: {filepath} ({len(r.content)} bytes)")
    else:
        print(f"  Failed to download: {r.status_code}")

# Explore Saliency Prediction component
print("=" * 60)
print("SALIENCY PREDICTION COMPONENT (67j5a)")
print("=" * 60)
r = requests.get('https://api.osf.io/v2/nodes/67j5a/files/osfstorage/', headers=headers, timeout=15)
if r.status_code == 200:
    items = r.json()['data']
    for item in items:
        name = item['attributes']['name']
        kind = item['attributes']['kind']
        size = item['attributes'].get('size', '?')
        if kind == 'folder':
            print(f"\n[DIR] {name}/")
            folder_url = item['relationships']['files']['links']['related']['href']
            list_folder(folder_url, "  ")
        else:
            print(f"\n[FILE] {name} ({size} bytes)")
            # Download Python files
            if name.endswith('.py'):
                download_url = item['links']['download']
                download_file(download_url, os.path.join(BASE, 'saliency_pred', name))

print("\n")
print("=" * 60)
print("GAZE BEHAVIOUR COMPONENT (z8bgm)")
print("=" * 60)
r = requests.get('https://api.osf.io/v2/nodes/z8bgm/files/osfstorage/', headers=headers, timeout=15)
if r.status_code == 200:
    items = r.json()['data']
    for item in items:
        name = item['attributes']['name']
        kind = item['attributes']['kind']
        size = item['attributes'].get('size', '?')
        print(f"\n[{'DIR' if kind == 'folder' else 'FILE'}] {name} ({size} bytes)")
        
        # Download the CSV (12MB - manageable)
        if name == 'fixation_AOI_metrics_final.csv':
            download_url = item['links']['download']
            os.makedirs(os.path.join(BASE, 'gaze'), exist_ok=True)
            download_file(download_url, os.path.join(BASE, 'gaze', name))
        
        # Download README
        if name == 'README.md':
            download_url = item['links']['download']
            os.makedirs(os.path.join(BASE, 'gaze'), exist_ok=True)
            download_file(download_url, os.path.join(BASE, 'gaze', name))
        
        # Download data_processing_pipeline.py
        if name == 'data_processing_pipeline.py':
            download_url = item['links']['download']
            os.makedirs(os.path.join(BASE, 'gaze'), exist_ok=True)
            download_file(download_url, os.path.join(BASE, 'gaze', name))
