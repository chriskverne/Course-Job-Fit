import pandas as pd
import matplotlib.pyplot as plt


def rank_courses(similarity_path, threshold):
    df = pd.read_csv(similarity_path)
    df.reset_index()
    # Average similarity for each course
    avg_sims = df.groupby('Course Name')['Similarity'].mean().sort_values()
    print(avg_sims[200:210])

    # Strong match count
    strong_matches = df[df['Similarity'] > threshold]
    smc = strong_matches.groupby('Course Name').size().sort_values()
    #print(smc)

    # Weighted salary
    df_salary = df.dropna(subset=['Job Salary'])  # Filter only rows with salary
    avg_sum_sim = df_salary.groupby('Course Name')['Similarity'].sum().mean()  # Based on salary data only

    def weighted_salary(course):
        w_salary_sum = (course['Similarity'] * course['Job Salary']).sum()
        sum_sim = course['Similarity'].sum()
        return w_salary_sum
        #return w_salary_sum / avg_sum_sim
        #return w_salary_sum / sum_sim

    weighted_mean_salary = df_salary.groupby('Course Name').apply(weighted_salary).sort_values()
    #print("\nWeighted Mean Salary by Course:\n", weighted_mean_salary)

#print('--------------- new data --------------')
#rank_courses('./BGE_all_course_ds_jobs.csv', 0.7)
#print('--------- old data------------')
#rank_courses('./computed_similarities/BGE/BGE_all_course_ds_jobs.csv', 0.7)

print('--------------- new data --------------')
rank_courses('./GTE_all_course_ds_jobs.csv', 0.7)
print('--------- old data------------')
rank_courses('./computed_similarities/GTE/GTE_all_course_ds_jobs.csv', 0.7)