import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import time
import os
os.environ['TFHUB_CACHE_DIR'] = './tf_cache'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU(s) memory growth set successfully.")
    except RuntimeError as e:
        print(e)

# Function to compute embeddings in batches
def compute_embeddings_in_batches(model, texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_embeddings = model(texts[i:i+batch_size])
        embeddings.append(batch_embeddings)
    return tf.concat(embeddings, axis=0)

# Calculate similarity using the USE model
def calculate_use_similarity(course_path, output_path, job_path, batch_size=64):
    # Load cleaned course and job descriptions
    courses_df = pd.read_excel(course_path)
    jobs_df = pd.read_excel(job_path)

    # Initialize the USE model
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Extract the course descriptions and job descriptions
    course_names = courses_df['Course Name'].tolist()
    course_descriptions = courses_df['Course Description'].tolist()

    job_titles = jobs_df['title'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()
    job_salaries = jobs_df['mean_salary'].tolist()

    # Compute embeddings for course descriptions and job descriptions in batches
    start_time = time.time()
    course_embeddings = compute_embeddings_in_batches(use_model, course_descriptions, batch_size)
    job_embeddings = compute_embeddings_in_batches(use_model, job_descriptions, batch_size)
    end_time = time.time()
    print("total time USE, ", end_time - start_time)

    # Compute cosine similarities between each course and job description
    similarity_matrix = tf.linalg.matmul(course_embeddings, job_embeddings, transpose_b=True)

    # Create a list to store the results in the desired format
    results = []

    # Loop through each course and job to create a flat structure
    for i, course_name in enumerate(course_names):
        for j, job_title in enumerate(job_titles):
            similarity_score = similarity_matrix[i][j].numpy()  # Get similarity score
            job_salary = job_salaries[j]  # Get the salary for the job
            results.append([course_name, job_title, similarity_score, job_salary])

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Course Name', 'Job Title', 'Similarity', 'Job Salary'])

    # Save the similarity results to a CSV file
    results_df.to_csv(output_path, index=False)

    print(f"USE-based similarity between courses and jobs calculated and saved to '{output_path}'.")

# Calculate similarity for the all courses dataset
calculate_use_similarity('../Datasets/cleaned_all_courses.xlsx', './USE_all_course.csv', '../Datasets/final_jobs.xlsx')
