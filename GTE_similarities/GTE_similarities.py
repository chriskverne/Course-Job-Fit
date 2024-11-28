import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("thenlper/gte-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base") # "thenlper/gte-large" for better performance
max_token_len = model.config.max_position_embeddings 

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    # Expand mask to embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Zero out padding tokens and get mean over real tokens only
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_mean_pooled_embedding_gte(text, model, tokenizer, max_token_len):
    # Single tokenization
    tokens = tokenizer(text, truncation=False, return_tensors='pt', padding=True)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    
    embeddings = []
    for i in range(0, tokens['input_ids'].size(1), max_token_len):
        # Directly use slices of the original tokenization
        chunk_tokens = {
            k: v[:, i:i + max_token_len] for k, v in tokens.items()
        }
        
        with torch.no_grad():
            model_output = model(**chunk_tokens)
            
        embedding = mean_pooling(model_output, chunk_tokens['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        embeddings.append(embedding[0])

    mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return mean_embedding

def calculate_similarity(course_path, output_path, job_path):
    # Load cleaned course and job descriptions
    courses_df = pd.read_excel(course_path)
    jobs_df = pd.read_excel(job_path)

    # Initialize the GTE model and tokenizer
    model_name = "thenlper/gte-base"  # or "thenlper/gte-large" for better performance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    max_token_len = model.config.max_position_embeddings 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Extract the course descriptions and job descriptions
    course_names = courses_df['Course Name'].tolist()
    course_descriptions = courses_df['Course Description'].tolist()
    job_titles = jobs_df['title'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()
    job_salaries = jobs_df['mean_salary'].tolist()

    # Counter for truncations
    truncation_count = 0
    tt_count = 0

    # Compute embeddings for course descriptions
    course_embeddings = []
    for description in course_descriptions:
        tokens = tokenizer(description, truncation=False)['input_ids']
        if len(tokens) > max_token_len:
            truncation_count += 1
            embedding = get_mean_pooled_embedding_gte(description, model, tokenizer, max_token_len)
        else:
            # For shorter texts, compute embedding directly
            encoded = tokenizer(description,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                model_output = model(**encoded)
            embedding = mean_pooling(model_output, encoded['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)[0]
        course_embeddings.append(embedding.cpu())  # Move embedding back to CPU
        tt_count += 1

    # Compute embeddings for job descriptions
    job_embeddings = []
    for description in job_descriptions:
        tokens = tokenizer(description, truncation=False)['input_ids']
        if len(tokens) > max_token_len:
            truncation_count += 1
            embedding = get_mean_pooled_embedding_gte(description, model, tokenizer, max_token_len)
        else:
            encoded = tokenizer(description,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                model_output = model(**encoded)
            embedding = mean_pooling(model_output, encoded['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)[0]
        job_embeddings.append(embedding.cpu())  # Move embedding back to CPU
        tt_count += 1

    print(f"Total descriptions truncated: {truncation_count}, tt_description : {tt_count}")

    # Convert lists to tensors for similarity calculation
    course_embeddings = torch.stack(course_embeddings)
    job_embeddings = torch.stack(job_embeddings)

    # Compute cosine similarities
    similarity_matrix = F.cosine_similarity(course_embeddings.unsqueeze(1),
                                            job_embeddings.unsqueeze(0),
                                            dim=2)

    # Create results list
    results = []
    for i, course_name in enumerate(course_names):
        for j, job_title in enumerate(job_titles):
            similarity_score = similarity_matrix[i][j].item()
            job_salary = job_salaries[j]
            results.append([course_name, job_title, similarity_score, job_salary])

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