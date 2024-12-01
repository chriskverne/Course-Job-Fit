import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import time
import os
import numpy as np
from tqdm import tqdm
import gc

# Force TensorFlow to use CPU for model loading
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TFHUB_CACHE_DIR'] = './tf_cache'

# Load model on CPU
print("Loading model on CPU...")
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Now enable GPU with strict memory limits
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Enable first GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Set a very conservative memory limit (512MB)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=512)]
            )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def clear_memory():
    """Clear memory between operations"""
    gc.collect()
    tf.keras.backend.clear_session()

def batch_encode(texts, batch_size=8):
    """Encode texts in very small batches"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Move computation to CPU if needed
        with tf.device('/CPU:0'):
            batch_embeddings = model(batch)
            embeddings.append(batch_embeddings.numpy())
        
        if i % (batch_size * 2) == 0:
            clear_memory()
    
    return np.vstack(embeddings)

def calculate_similarity_batched(course_path, output_path, job_path, batch_size=8):
    try:
        # Load data
        courses_df = pd.read_excel(course_path)
        jobs_df = pd.read_excel(job_path)
        
        course_names = courses_df['Course Name'].tolist()
        course_descriptions = courses_df['Course Description'].tolist()
        job_titles = jobs_df['title'].tolist()
        job_descriptions = jobs_df['cleaned_description'].tolist()
        job_salaries = jobs_df['mean_salary'].tolist()

        start_time = time.time()
        
        print("Encoding course descriptions...")
        course_embeddings = batch_encode(course_descriptions, batch_size)
        course_embeddings = tf.nn.l2_normalize(tf.convert_to_tensor(course_embeddings), axis=1)
        clear_memory()
        
        print("Encoding job descriptions...")
        job_embeddings = batch_encode(job_descriptions, batch_size)
        job_embeddings = tf.nn.l2_normalize(tf.convert_to_tensor(job_embeddings), axis=1)
        clear_memory()
        
        print("Computing similarities...")
        # Process similarities in very small chunks
        similarity_matrix = []
        mini_batch = 4  # Very small batch size for matrix multiplication
        
        for i in tqdm(range(0, len(course_embeddings), mini_batch)):
            course_batch = course_embeddings[i:i + mini_batch]
            # Try GPU first, fall back to CPU if needed
            try:
                batch_similarities = tf.linalg.matmul(course_batch, job_embeddings, transpose_b=True)
            except:
                with tf.device('/CPU:0'):
                    batch_similarities = tf.linalg.matmul(course_batch, job_embeddings, transpose_b=True)
            
            similarity_matrix.append(batch_similarities.numpy())
            
            if i % (mini_batch * 2) == 0:
                clear_memory()
        
        similarity_matrix = np.vstack(similarity_matrix)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

        # Process results in small batches
        print("Creating results dataframe...")
        results = []
        batch_size = 50  # Process results in batches
        for i in range(0, len(course_names), batch_size):
            batch_end = min(i + batch_size, len(course_names))
            for course_idx in range(i, batch_end):
                for j, job_title in enumerate(job_titles):
                    results.append([
                        course_names[course_idx],
                        job_title,
                        float(similarity_matrix[course_idx][j]),
                        job_salaries[j]
                    ])
            clear_memory()

        # Save results
        results_df = pd.DataFrame(results, columns=['Course Name', 'Job Title', 'Similarity', 'Job Salary'])
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to '{output_path}'")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        clear_memory()
        raise e

# Process each job category
course_path = '../Datasets/cleaned_all_courses.xlsx'
job_categories = {
    'cs': '../Datasets/cs_jobs.xlsx',
    'ds': '../Datasets/ds_jobs.xlsx',
    'it': '../Datasets/it_jobs.xlsx',
    'pm': '../Datasets/pm_jobs.xlsx',
    'swe': '../Datasets/swe_jobs.xlsx'
}

for category, job_path in job_categories.items():
    print(f"\nProcessing {category} jobs...")
    try:
        clear_memory()
        output_path = f'./USE_all_course_{category}_jobs.csv'
        calculate_similarity_batched(course_path, output_path, job_path)
    except Exception as e:
        print(f"Error processing {category}: {str(e)}")
        clear_memory()
        continue