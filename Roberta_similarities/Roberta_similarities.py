import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the RoBERTa model and tokenizer
model = RobertaModel.from_pretrained('roberta-base').to(device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def get_mean_pooled_roberta_embedding(text, model, tokenizer):
    # Tokenize and get both tokens and attention mask
    tokenized = tokenizer(text, truncation=False, return_tensors='pt', padding=True)
    tokens = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    # Split into chunks of max 512 tokens
    chunks_tokens = [tokens[:, i:i + 512] for i in range(0, tokens.shape[1], 512)]
    chunks_mask = [attention_mask[:, i:i + 512] for i in range(0, attention_mask.shape[1], 512)]

    embeddings = []
    for chunk_tokens, chunk_mask in zip(chunks_tokens, chunks_mask):
        with torch.no_grad():
            # Get model outputs
            outputs = model(chunk_tokens)
            last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
            
            # Apply attention mask to zero out padding tokens
            # Expand attention_mask to same dims as hidden state
            mask = chunk_mask.unsqueeze(-1).expand(last_hidden_state.size())
            
            # Mask out padding tokens
            masked_embeddings = last_hidden_state * mask
            
            # Sum up non-padding tokens and divide by attention mask sum
            sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over sequence length
            sum_mask = chunk_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)  # Avoid division by zero
            
            # Mean pool over non-padding tokens
            chunk_embedding = sum_embeddings / sum_mask
            embeddings.append(chunk_embedding)

    mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return mean_embedding

def calculate_roberta_similarity(course_path, output_path, job_path):
    # Load cleaned course and job descriptions
    courses_df = pd.read_excel(course_path)
    jobs_df = pd.read_excel(job_path)

    # Extract the course descriptions and job descriptions
    course_names = courses_df['Course Name'].tolist()
    course_descriptions = courses_df['Course Description'].tolist()

    job_titles = jobs_df['title'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()
    job_salaries = jobs_df['mean_salary'].tolist()

    # Compute embeddings for course descriptions
    course_embeddings = []
    for description in course_descriptions:
        # Check token length first
        tokens = tokenizer(description, truncation=False)['input_ids']
        if len(tokens) > 512:
            # Use mean pooling function for long texts
            embedding = get_mean_pooled_roberta_embedding(description, model, tokenizer)
        else:
            # For short texts, simple forward pass
            inputs = tokenizer(description, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
        course_embeddings.append(embedding)

    # Compute embeddings for job descriptions
    job_embeddings = []
    for description in job_descriptions:
        tokens = tokenizer(description, truncation=False, return_tensors='pt')['input_ids'].to(device)
        if tokens.shape[1] > 512:
            embedding = get_mean_pooled_roberta_embedding(description, model, tokenizer)
        else:
            with torch.no_grad():
                embedding = model(tokens).last_hidden_state.mean(dim=1)
        job_embeddings.append(embedding)

    # Move embeddings back to CPU for similarity calculation
    course_embeddings = torch.cat(course_embeddings).cpu()
    job_embeddings = torch.cat(job_embeddings).cpu()

    # Compute cosine similarities between each course and job description
    similarity_matrix = F.cosine_similarity(course_embeddings.unsqueeze(1), job_embeddings.unsqueeze(0), dim=-1)

    # Store and save the results
    results = []
    for i, course_name in enumerate(course_names):
        for j, job_title in enumerate(job_titles):
            similarity_score = similarity_matrix[i][j].item()
            job_salary = job_salaries[j]
            results.append([course_name, job_title, similarity_score, job_salary])

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['Course Name', 'Job Title', 'Similarity', 'Job Salary'])
    results_df.to_csv(output_path, index=False)

    print(f"Similarity between courses and jobs calculated using RoBERTa and saved to '{output_path}'.")


# Paths for datasets and outputs
all_path = '../Datasets/cleaned_all_courses.xlsx'
all_output_roberta = './all_course_job_similarity_roberta.csv'
job_path = '../Datasets/final_jobs.xlsx'

calculate_roberta_similarity(all_path, all_output_roberta, job_path)
