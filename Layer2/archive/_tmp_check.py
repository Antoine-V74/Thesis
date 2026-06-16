import pandas as pd, numpy as np

df = pd.read_csv('Results/final_mitbih_validation/beat_sync_v2_final/per_beat.csv', low_memory=False)
df = df[(df['mode']=='layer1_adaptive_gated') & (df['feature_set']=='all')]

total_h = (df['label']=='healthy').sum()
total_ab = (df['label']=='abnormal_v').sum()

print('=== Reason breakdown on healthy false inhibits ===')
h_inh = df[(df['label']=='healthy') & (df['permit']==0)]
print(h_inh['reason'].value_counts().to_string())

print()
print('=== Reason breakdown on abnormal TRUE inhibits ===')
ab_inh = df[(df['label']=='abnormal_v') & (df['permit']==0)]
print(ab_inh['reason'].value_counts().to_string())

print()
print(f'HP = {df[df["label"]=="healthy"]["permit"].mean():.4f}')
print(f'AI = {(~df[df["label"]=="abnormal_v"]["permit"]).mean():.4f}')
