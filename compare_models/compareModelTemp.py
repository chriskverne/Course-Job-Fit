import pandas as pd
import itertools

# Load similarity data
mpnet_sim = pd.read_csv('./all_course_job_similarity_mpnet.csv')
e5_sim = pd.read_csv('./e5_all_course_job_similarity.csv')
gte_sim = pd.read_csv('./gte_all_course_job_similarity.csv')
sbert_sim = pd.read_csv('./sbert_all_course_job_similarity.csv')
use_sim = pd.read_csv('./use_all_course_job_similarity.csv')

# Function to compute average similarity rank
def rank_courses_avg_similarity(df):
    avg_sims = df.groupby('Course Name')['Similarity'].mean().sort_values()
    return avg_sims

# Compute average similarity ranks for each model
models = {
    "MPNet": rank_courses_avg_similarity(mpnet_sim),
    "E5": rank_courses_avg_similarity(e5_sim),
    "GTE": rank_courses_avg_similarity(gte_sim),
    "SBERT": rank_courses_avg_similarity(sbert_sim),
    "USE": rank_courses_avg_similarity(use_sim)
}

# Helper function to rank courses for average similarity
def get_avg_sim_ranks(avg_sims, model_name):
    avg_sims_rank = avg_sims.rank(ascending=False).astype(int)
    return pd.DataFrame({
        'Course Name': avg_sims.index,
        f'{model_name}_Avg_Sim_Rank': avg_sims_rank
    }).set_index('Course Name')

# Create a DataFrame for all ranks
all_ranks = None
for model_name, avg_sims in models.items():
    ranks_df = get_avg_sim_ranks(avg_sims, model_name)
    if all_ranks is None:
        all_ranks = ranks_df
    else:
        all_ranks = all_ranks.join(ranks_df, how='outer')

# Calculate pairwise rank differences
def calculate_rank_differences(all_ranks):
    model_columns = all_ranks.columns
    model_pairs = list(itertools.combinations(model_columns, 2))

    rank_differences = {}
    for model1, model2 in model_pairs:
        diff = (all_ranks[model1] - all_ranks[model2]).abs()
        avg_diff = diff.mean()
        rank_differences[(model1, model2)] = avg_diff

    return rank_differences

rank_differences = calculate_rank_differences(all_ranks)

# Convert rank differences to a DataFrame
rank_diff_df = pd.DataFrame([
    {"Model Pair": f"{pair[0]} vs {pair[1]}", "Average Rank Difference": diff}
    for pair, diff in rank_differences.items()
])

# Calculate average similarity of each model to others
def calculate_model_avg_similarity(rank_differences):
    model_sums = {}
    model_counts = {}

    for (model1, model2), avg_diff in rank_differences.items():
        model_sums[model1] = model_sums.get(model1, 0) + avg_diff
        model_sums[model2] = model_sums.get(model2, 0) + avg_diff
        model_counts[model1] = model_counts.get(model1, 0) + 1
        model_counts[model2] = model_counts.get(model2, 0) + 1

    model_avg_similarity = {model: model_sums[model] / model_counts[model] for model in model_sums}
    return model_avg_similarity

model_avg_similarity = calculate_model_avg_similarity(rank_differences)

# Convert model average similarity to DataFrame
model_avg_similarity_df = pd.DataFrame({
    'Model': model_avg_similarity.keys(),
    'Average Similarity to Others': model_avg_similarity.values()
}).sort_values(by='Average Similarity to Others')

# Calculate Spearman correlations between models
spearman_corr = all_ranks.corr(method='spearman')

# Rank courses based on average similarity rank across models
all_ranks['Average_Course_Rank'] = all_ranks.mean(axis=1)
course_rankings = all_ranks.sort_values(by='Average_Course_Rank')

# Save all results to Excel
rank_diff_df.to_excel('model_rank_differences.xlsx', index=False)
model_avg_similarity_df.to_excel('model_similarity_rank.xlsx', index=False)
course_rankings.to_excel('aggregated_course_rankings.xlsx')
spearman_corr.to_excel('model_spearman_correlations.xlsx')

# Print summary of results
print("Most Similar Models:")
print(rank_diff_df.loc[rank_diff_df["Average Rank Difference"].idxmin()])

print("\nLeast Similar Models:")
print(rank_diff_df.loc[rank_diff_df["Average Rank Difference"].idxmax()])

print("\nModel Average Similarity to Others:")
print(model_avg_similarity_df)

print("\nTop Ranked Courses Based on Average Rank:")
print(course_rankings.head())
