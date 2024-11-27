import pandas as pd
from sentence_transformers import SentenceTransformer, util

def avg_tokens(courses_df, jobs_df, model):
    tokenizer = model.tokenizer
    course_descriptions = courses_df['Course Description'].tolist()
    job_descriptions = jobs_df['cleaned_description'].tolist()

    tt_course_tokens = 0
    tt_job_tokens = 0

    for description in course_descriptions:
        tokens  = tokenizer(description, truncation=False)['input_ids']
        tt_course_tokens += len(tokens)

    for description in job_descriptions:
        tokens = tokenizer(description, truncation=False)['input_ids']
        tt_job_tokens += len(tokens)

    avg_course_tokens = tt_course_tokens / len(course_descriptions)
    avg_job_tokens = tt_job_tokens / len(job_descriptions)

    print('Avg course tokens: ', avg_course_tokens, 'Avg job tokens: ', avg_job_tokens)

    tt_course_words = 0
    tt_job_words = 0
    for course in course_descriptions:
        tt_course_words += len(course.split(' '))
    for job in job_descriptions:
        tt_job_words += len(job.split(' '))

    avg_course_words = tt_course_words / len(course_descriptions)
    avg_job_words = tt_job_words / len(job_descriptions)

    print('Avg words course: ', avg_course_words, 'Avg job words', avg_job_words)


courses_df = pd.read_excel('../Datasets/cleaned_all_courses.xlsx')
jobs_df = pd.read_excel('../Datasets/final_jobs.xlsx')
model = SentenceTransformer('all-MiniLM-L6-v2')

avg_tokens(courses_df, jobs_df, model)