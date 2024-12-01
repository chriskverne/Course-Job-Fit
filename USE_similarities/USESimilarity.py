import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import time
import os
os.environ['TFHUB_CACHE_DIR'] = './tf_cache'

# Initialize the USE model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[4], 'GPU')  # Use GPU 4
visible_gpus = tf.config.get_visible_devices('GPU')
for gpu in visible_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth

def calculate_similarity(course_path, output_path, job_path):
    # Load cleaned course and job descriptions
    courses_df = pd.read_excel(course_path)
    jobs_df = pd.read_excel(job_path)

    # Extract the course descriptions and job descriptions
    course_names = courses_df['Course Name'].tolist()
    course_descriptions = courses_df['Course Description'].tolist()

    job_titles = jobs_df['title'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()
    job_salaries = jobs_df['mean_salary'].tolist()

    # Compute embeddings for course descriptions and job descriptions
    start_time = time.time()
    
    # Create embeddings
    course_embeddings = [model(desc) for desc in course_descriptions]
    job_embeddings = [model(desc) for desc in job_descriptions]
    
    # Stack into 2D tensors and normalize
    course_embeddings = tf.nn.l2_normalize(tf.stack(course_embeddings), axis=1)
    job_embeddings = tf.nn.l2_normalize(tf.stack(job_embeddings), axis=1)
    
    # Compute cosine similarities
    similarity_matrix = tf.linalg.matmul(course_embeddings, job_embeddings, transpose_b=True)
    
    end_time = time.time()
    print("Total time USE:", end_time - start_time)

    # Create results list
    results = [
        [course_name, job_title, similarity_matrix[i][j].numpy(), job_salaries[j]]
        for i, course_name in enumerate(course_names)
        for j, job_title in enumerate(job_titles)
    ]

    # Convert results to a DataFrame and save
    results_df = pd.DataFrame(results, columns=['Course Name', 'Job Title', 'Similarity', 'Job Salary'])
    results_df.to_csv(output_path, index=False)
    print(f"USE-based similarity between courses and jobs calculated and saved to '{output_path}'.")


course_path = '../Datasets/cleaned_all_courses.xlsx'
calculate_similarity(course_path, './USE_all_course_cs_jobs.csv', '../Datasets/cs_jobs.xlsx')
calculate_similarity(course_path, './USE_all_course_ds_jobs.csv', '../Datasets/ds_jobs.xlsx')
calculate_similarity(course_path, './USE_all_course_it_jobs.csv', '../Datasets/it_jobs.xlsx')
calculate_similarity(course_path, './USE_all_course_pm_jobs.csv', '../Datasets/pm_jobs.xlsx')
calculate_similarity(course_path, './USE_all_course_swe_jobs.csv', '../Datasets/swe_jobs.xlsx')