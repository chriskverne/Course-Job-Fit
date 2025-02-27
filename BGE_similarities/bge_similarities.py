import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time

"""
# Model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
max_token_len = tokenizer.model_max_length

# How many descriptions are above token limit
truncation_count = 0
total_count = 0

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text):
    global truncation_count, total_count
    # BGE models expect a query prefix for better performance
    text = f"Represent this text for semantic similarity: {text}"
    
    with torch.no_grad():
        tokens = tokenizer(text, truncation=False)['input_ids']
        if len(tokens) > max_token_len:
            truncation_count += 1
            embedding = get_mean_pooled_embedding(tokens)
        else:
            encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = model(**encoded)
            embedding = mean_pooling(model_output, encoded['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)[0]
        total_count += 1
        return embedding.cpu()

def get_mean_pooled_embedding(tokens):
    # Simple chunking like SBERT
    chunks = [tokens[i:i + max_token_len] for i in range(0, len(tokens), max_token_len)]
    embeddings = []
    token_counts = []
    
    for chunk in chunks:
        # Add the prefix to each chunk for consistency
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunk_text = f"Represent this text for semantic similarity: {chunk_text}"
        
        encoded = tokenizer(chunk_text, padding=True, truncation=True, return_tensors='pt').to(device)
        model_output = model(**encoded)
        chunk_embedding = mean_pooling(model_output, encoded['attention_mask'])
        chunk_embedding = F.normalize(chunk_embedding, p=2, dim=1)[0]
        
        embeddings.append(chunk_embedding)
        token_counts.append(len(chunk))

    total_tokens = sum(token_counts)
    weighted_embeddings = [emb * (count / total_tokens) for emb, count in zip(embeddings, token_counts)]
    final_embedding = torch.stack(weighted_embeddings).sum(dim=0)
    final_embedding = F.normalize(final_embedding.unsqueeze(0), p=2, dim=1)[0]
    
    return final_embedding
"""

# ------------------------------- Code 2 ----------------------------------
# Model parameters
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
max_token_len = tokenizer.model_max_length

# How many descriptions are above token limit
truncation_count = 0
total_count = 0

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode_text(text):
    global truncation_count, total_count
    text = f"Represent this text for semantic similarity: {text}"
    with torch.no_grad():
        tokens = tokenizer(text, truncation=False, return_tensors="pt", padding=True).to(device)
        input_ids = tokens['input_ids'][0]
        attention_mask = tokens['attention_mask'][0]

        if len(input_ids) > max_token_len:
            truncation_count+=1
            total_count+=1
            chunk_size = max_token_len - 2
            
            input_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
            mask_chunks = [attention_mask[i:i + chunk_size] for i in range(0, len(attention_mask), chunk_size)]
            
            embeddings = []
            token_counts = []
                        
            for input_chunk, mask_chunk in zip(input_chunks, mask_chunks):
                chunk_text = tokenizer.decode(input_chunk, skip_special_tokens=True)
                chunk_text = f"Represent this text for semantic similarity: {chunk_text}"
                chunk_input = tokenizer(chunk_text, padding=True, truncation=True, return_tensors='pt').to(device)
                model_output = model(**chunk_input)
                
                # Use average pooling instead of CLS token
                embedding = average_pool(model_output.last_hidden_state, chunk_input['attention_mask'])
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)[0]
                
                valid_tokens = mask_chunk.sum().item()
                embeddings.append(embedding)
                token_counts.append(valid_tokens)
            
            total_tokens = sum(token_counts)
            weighted_embeddings = [emb * (count / total_tokens) for emb, count in zip(embeddings, token_counts)]
            final_embedding = torch.stack(weighted_embeddings).sum(dim=0)
            return F.normalize(final_embedding, p=2, dim=0).cpu()
        else:
            total_count+=1
            model_output = model(**tokens)
            # Use average pooling instead of CLS token
            embeddings = average_pool(model_output.last_hidden_state, tokens['attention_mask'])
            return F.normalize(embeddings, p=2, dim=1)[0].cpu()

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

    # Compute embeddings
    start_time = time.time()
    course_embeddings = [encode_text(desc) for desc in course_descriptions]
    job_embeddings = [encode_text(desc) for desc in job_descriptions]

    print(f"Elapsed time: {time.time() - start_time} seconds")
    print(f"Total descriptions truncated: {truncation_count}, total_descriptions: {total_count}")

    # Convert lists to tensors for similarity calculation
    course_embeddings = torch.stack(course_embeddings)
    job_embeddings = torch.stack(job_embeddings)

    # Compute cosine similarities
    similarity_matrix = F.cosine_similarity(
        course_embeddings.unsqueeze(1), 
        job_embeddings.unsqueeze(0), 
        dim=2
    )

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
course_path = '../Datasets/cleaned_all_courses.xlsx'
calculate_similarity(course_path, './BGE_all_course_cs_jobs.csv', '../Datasets/cs_jobs.xlsx')
calculate_similarity(course_path, './BGE_all_course_ds_jobs.csv', '../Datasets/ds_jobs.xlsx')
calculate_similarity(course_path, './BGE_all_course_it_jobs.csv', '../Datasets/it_jobs.xlsx')
calculate_similarity(course_path, './BGE_all_course_pm_jobs.csv', '../Datasets/pm_jobs.xlsx')
calculate_similarity(course_path, './BGE_all_course_swe_jobs.csv', '../Datasets/swe_jobs.xlsx')