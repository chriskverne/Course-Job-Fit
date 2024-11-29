import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time

# Model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("thenlper/gte-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base") # "thenlper/gte-large" for better performance
max_token_len = model.config.max_position_embeddings 

# How many descriptions are above token limit
truncation_count = 0
total_count = 0

def encode_text(text):
    global truncation_count, total_count
    with torch.no_grad():
        tokens = tokenizer(text, truncation=False)['input_ids'][0]
        if len(tokens) > max_token_len:
            truncation_count+=1
            embedding = get_mean_pooled_embedding(tokens)
        else:
            embedding = model.encode(text)
        total_count+=1
        return embedding

def get_mean_pooled_embedding(tokens):
    chunks = [tokens[i: i + max_token_len] for i in range(0, len(tokens), max_token_len)]
    embeddings = []
    token_counts = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunk_embedding = model.encode(chunk_text, convert_to_tensor=True)
        embeddings.append(chunk_embedding)
        token_counts.append(len(chunk))

    total_tokens = sum(token_counts)
    weighted_embeddings = [embedding * (count / total_tokens) for embedding, count in zip(embeddings, token_counts)]
    final_embedding = torch.stack(weighted_embeddings).sum(dim=0)
    return final_embedding

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

    # Compute embeddings for course/job descriptions
    start_time = time.time()
    course_embeddings = [encode_text(desc) for desc in course_descriptions]
    job_embeddings = [encode_text(desc) for desc in job_descriptions]

    print(f"Elapsed time: {time.time() - start_time} seconds")  # Fixed
    print(f"Total descriptions truncated: {truncation_count}, tt_description : {total_count}")

    # Convert lists to tensors for similarity calculation
    course_embeddings = torch.stack(course_embeddings)
    job_embeddings = torch.stack(job_embeddings)

    # Compute cosine similarities
    similarity_matrix = util.cos_sim(course_embeddings, job_embeddings)

    # Create results list
    results = [
        [course_name, job_title, similarity_matrix[i][j].item(), job_salaries[j]]
        for i, course_name in enumerate(course_names)
        for j, job_title in enumerate(job_titles)
    ]

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results, columns=['Course Name', 'Job Title', 'Similarity', 'Job Salary'])
    results_df.to_csv(output_path, index=False)

    print(f"Similarity between courses and jobs calculated and saved to '{output_path}'.")


# Paths for datasets and outputs
core_path = '../../Datasets/cleaned_core_courses.xlsx'
core_output = './core_course_job_similarity.csv'
elective_path = '../../Datasets/cleaned_elective_courses.xlsx'
elective_output = './elective_course_job_similarity.csv'
all_path = '../Datasets/cleaned_all_courses.xlsx'
all_output = './all_course_job_similarity.csv'

# Calculate similarity for the all courses dataset
calculate_similarity(all_path, all_output, '../Datasets/final_jobs.xlsx')