import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def mean_pooling(token_embeddings, attention_mask):
    # Mean pooling - take attention mask into account for correct averaging
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_chunked_embedding(text, model, tokenizer, max_length):
    # Split text into chunks and process each chunk
    tokens = tokenizer(text, truncation=False, return_tensors='pt', padding=True)

    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    # Calculate how many tokens we have
    total_tokens = tokens['input_ids'].size(1)

    # If we're under max_length, just process normally
    if total_tokens <= max_length:
        with torch.no_grad():
            outputs = model(**{k: v for k, v in tokens.items()})
        embeddings = mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

    # Otherwise, split into chunks
    chunks = []
    for i in range(0, total_tokens, max_length):
        chunk_tokens = {
            k: v[:, i:i + max_length] for k, v in tokens.items()
        }
        with torch.no_grad():
            outputs = model(**chunk_tokens)
        chunk_embedding = mean_pooling(outputs.last_hidden_state, chunk_tokens['attention_mask'])
        chunks.append(chunk_embedding)

    # Average all chunk embeddings
    final_embedding = torch.mean(torch.stack(chunks), dim=0)
    return F.normalize(final_embedding, p=2, dim=1)

def calculate_similarity_e5(course_path, output_path, job_path):
    # Load cleaned course and job descriptions
    courses_df = pd.read_excel(course_path)
    jobs_df = pd.read_excel(job_path)

    # Initialize the E5 model and tokenizer
    model_name = "intfloat/e5-large-v2"  # Correct
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    max_token_len = model.config.max_position_embeddings

    # Move model to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)
    model.eval()

    # Extract the course descriptions and job descriptions
    course_names = courses_df['Course Name'].tolist()
    course_descriptions = courses_df['Course Description'].tolist()

    job_titles = jobs_df['title'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()
    job_salaries = jobs_df['mean_salary'].tolist()

    # Counter for tracking
    processed_count = 0
    total_items = len(course_descriptions) + len(job_descriptions)

    # Compute embeddings for course descriptions
    print("Processing course descriptions...")
    course_embeddings = []
    for description in course_descriptions:
        # Prefix for task-specific prompt
        prefixed_text = f"passage: {description}"
        embedding = get_chunked_embedding(prefixed_text, model, tokenizer, max_token_len)
        course_embeddings.append(embedding)
        
    # Compute embeddings for job descriptions
    print("\nProcessing job descriptions...")
    job_embeddings = []
    for description in job_descriptions:
        # Prefix for task-specific prompt
        prefixed_text = f"passage: {description}"
        embedding = get_chunked_embedding(prefixed_text, model, tokenizer, max_token_len)
        job_embeddings.append(embedding)
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count}/{total_items} items...")

    # Convert lists to tensors and compute similarity
    course_embeddings = torch.cat(course_embeddings)
    job_embeddings = torch.cat(job_embeddings)

    # Compute cosine similarities
    similarity_matrix = torch.mm(course_embeddings, job_embeddings.transpose(0, 1))

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

    print(f"\nSimilarity between courses and jobs calculated and saved to '{output_path}'.")


if __name__ == "__main__":
    # Paths for datasets and outputs
    core_path = '../../Datasets/cleaned_core_courses.xlsx'
    core_output = './e5_core_course_job_similarity.csv'

    elective_path = '../../Datasets/cleaned_elective_courses.xlsx'
    elective_output = './e5_elective_course_job_similarity.csv'

    all_path = '../Datasets/cleaned_all_courses.xlsx'
    all_output = './e5_all_course_job_similarity.csv'

    # Calculate similarity for the all courses dataset
    calculate_similarity_e5(all_path, all_output, '../Datasets/final_jobs.xlsx')