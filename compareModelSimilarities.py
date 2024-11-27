import math
import pandas as pd
import numpy as np
from itertools import combinations

# Read the data
sbert_sim = pd.read_csv('./SBERT_similarities/all_course_job_similarity.csv')
roberta_sim = pd.read_csv('./Roberta_similarities/all_course_job_similarity_roberta.csv')
use_sim = pd.read_csv('./USE_similarities/use_all_course_job_similarity.csv')
gte_sim = pd.read_csv('./GTE_similarities/all_course_job_similarity.csv')
e5_sim = pd.read_csv('./e5_similarities/e5_all_course_job_similarity.csv')

# Calculate average similarities
models = {
    'SBERT': sbert_sim.groupby('Course Name')['Similarity'].mean().sort_values(ascending=False),
    'ROBERTA': roberta_sim.groupby('Course Name')['Similarity'].mean().sort_values(ascending=False),
    'USE': use_sim.groupby('Course Name')['Similarity'].mean().sort_values(ascending=False),
    'GTE': gte_sim.groupby('Course Name')['Similarity'].mean().sort_values(ascending=False),
    'E5': e5_sim.groupby('Course Name')['Similarity'].mean().sort_values(ascending=False)
}

# Create rankings dataframes
rankings = {}
for name, model in models.items():
    rankings[name] = model.reset_index().rename(columns={'Similarity': f'{name}_Similarity'})
    rankings[name][f'{name}_Rank'] = rankings[name][f'{name}_Similarity'].rank(ascending=False)

# Merge all rankings
merged_rankings = rankings['SBERT']
for name in ['ROBERTA', 'USE', 'GTE', 'E5']:
    merged_rankings = pd.merge(merged_rankings, rankings[name], on='Course Name')

# Calculate pairwise rank differences and similarities
model_pairs = list(combinations(['SBERT', 'ROBERTA', 'USE', 'GTE', 'E5'], 2))
for model1, model2 in model_pairs:
    # Rank difference
    diff_col = f'{model1}_vs_{model2}_Rank_Diff'
    merged_rankings[diff_col] = abs(merged_rankings[f'{model1}_Rank'] - merged_rankings[f'{model2}_Rank'])

    # Similarity difference
    sim_diff_col = f'{model1}_vs_{model2}_Sim_Diff'
    merged_rankings[sim_diff_col] = abs(
        merged_rankings[f'{model1}_Similarity'] - merged_rankings[f'{model2}_Similarity'])

# Calculate statistics for each pair
comparison_stats = []
for model1, model2 in model_pairs:
    rank_diff_col = f'{model1}_vs_{model2}_Rank_Diff'
    sim_diff_col = f'{model1}_vs_{model2}_Sim_Diff'

    stats = {
        'Model Pair': f'{model1} vs {model2}',
        'Avg Rank Difference': merged_rankings[rank_diff_col].mean(),
        'Max Rank Difference': merged_rankings[rank_diff_col].max(),
        'Avg Similarity Difference': merged_rankings[sim_diff_col].mean(),
        'Max Similarity Difference': merged_rankings[sim_diff_col].max(),
    }
    comparison_stats.append(stats)

comparison_df = pd.DataFrame(comparison_stats)

# Print statistics
print("Model Comparison Statistics:")
print(comparison_df.to_string(index=False))

# Find courses with highest disagreement across all models
merged_rankings['Max_Rank_Difference'] = merged_rankings[[f'{m1}_vs_{m2}_Rank_Diff'
                                                          for m1, m2 in model_pairs]].max(axis=1)

print("\nTop 10 Courses with Highest Rank Disagreement Across Models:")
disagreement_courses = merged_rankings.nlargest(10, 'Max_Rank_Difference')
print(disagreement_courses[['Course Name', 'Max_Rank_Difference'] +
                           [f'{model}_Rank' for model in ['SBERT', 'ROBERTA', 'USE', 'GTE', 'E5']]])

# Save detailed results
output_cols = ['Course Name'] + \
              [f'{model}_Similarity' for model in ['SBERT', 'ROBERTA', 'USE', 'GTE', 'E5']] + \
              [f'{model}_Rank' for model in ['SBERT', 'ROBERTA', 'USE', 'GTE', 'E5']] + \
              [f'{m1}_vs_{m2}_Rank_Diff' for m1, m2 in model_pairs]

merged_rankings[output_cols].to_excel('./Model_Comparison_Analysis.xlsx')
comparison_df.to_excel('./Model_Comparison_Statistics.xlsx')