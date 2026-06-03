"""
HCEye Data Analysis - Explore cognitive load sensitivity patterns.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('hceye/gaze/fixation_AOI_metrics_final.csv')

# Key gaze metrics
metrics = ['TotalNumFixations', 'MeanFixationDuration', 'FixationFrequency']

print('=== Per CognitiveLoad condition (mean across all) ===')
grouped = df.groupby('CognitiveLoad')[metrics].mean()
print(grouped)
print()

# Per image + condition
img_cog = df.groupby(['Image_Name', 'CognitiveLoad'])[metrics].mean().reset_index()

# Compute sensitivity: ratio of High/Absent for each image
print('=== Cognitive Load Sensitivity (TotalNumFixations) ===')
pivot = img_cog.pivot_table(index='Image_Name', columns='CognitiveLoad', values='TotalNumFixations')
if 'Absent' in pivot.columns and 'High' in pivot.columns:
    pivot['sensitivity_high'] = pivot['High'] / pivot['Absent']
    print(pivot['sensitivity_high'].describe())
    print()

# Duration sensitivity
print('=== Cognitive Load Sensitivity (MeanFixationDuration) ===')
pivot_dur = img_cog.pivot_table(index='Image_Name', columns='CognitiveLoad', values='MeanFixationDuration')
if 'Absent' in pivot_dur.columns and 'High' in pivot_dur.columns:
    pivot_dur['sensitivity_high'] = pivot_dur['High'] / pivot_dur['Absent']
    print(pivot_dur['sensitivity_high'].describe())
    print()

# Highlight effect under cognitive load
print('=== Highlight x CognitiveLoad interaction ===')
hl_cog = df.groupby(['Highlight', 'CognitiveLoad'])[metrics].mean()
print(hl_cog)
print()

# AOI hit rate per condition
print('=== AOI Hit Rate per condition ===')
aoi_rate = df.groupby(['CognitiveLoad', 'Highlight'])['AOI_Hit'].mean()
print(aoi_rate)
print()

# Image-level aggregation (what we'll use as ground truth)
print('=== Image-level summary (Absent cognitive load, no highlight) ===')
baseline = df[(df['CognitiveLoad'] == 'Absent') & (df['Highlight'] == 'Absent')]
img_baseline = baseline.groupby('Image_Name')[metrics].mean()
print(f'Images with baseline data: {len(img_baseline)}')
print(img_baseline.describe())
