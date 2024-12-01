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

path_sbert = './computed_similarities/SBERT/SBERT_all_course_it_jobs.csv'
path_sbert2 = './computed_similarities/SBERT/SBERT_all_course_cs_jobs.csv'

path_sbert_trunc = './SBERT_similarities/all_course_job_similarity_with_trunc.csv'
path_use = './USE_similarities/all_course_job_similarity.csv'
#rank_courses(path_roberta, 0.70)
#rank_courses(path_sbert, 0.70)
rank_courses(path_sbert2, 0.70)
#rank_courses(path_sbert_trunc, 0.70)
#rank_courses(path_use, 0.70)


#avg_sims = df.groupby('Course Name')['Similarity'].mean()#.sort_values()
#print(len(avg_sims))

#high_sims_jobs = df[df['Similarity'] > 0.75].groupby('Course Name').size().sort_values()
#print(high_sims_jobs)

#plt.figure()
#avg_sims.plot(kind='barh')
#plt.show()

#plt.figure()
#high_sim_count.plot(kind='barh')
#plt.show()