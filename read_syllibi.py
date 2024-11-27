import PyPDF2
import re
import os
from nltk.corpus import stopwords
import pandas as pd

# Load stopwords for English
stop_words = set(stopwords.words('english'))

def clean_course(course):
    # Convert to lowercase
    course = course.lower()
    
    # Remove non-alphabetical characters
    course = re.sub('[^a-zA-Z]', ' ', course)
    
    # Remove extra whitespace
    course = re.sub(r'\s+', ' ', course)
    
    # Tokenize and remove stopwords
    words = course.split()
    filtered_course = ' '.join([word for word in words if word not in stop_words])
    
    return filtered_course

cleaned_courses = {}

def process_files(folder_path, output_path):
    print(f'Number of files: {len(os.listdir(folder_path))}')
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        reader = PyPDF2.PdfReader(file_path)
        
        course_text = ''
        for page in reader.pages:
            course_text += page.extract_text()
            
        # Clean the extracted course text
        cleaned_course = clean_course(course_text)
        
        course_name = os.path.splitext(filename)[0]  # Remove the .pdf extension
        cleaned_courses[course_name] = cleaned_course

    df = pd.DataFrame(list(cleaned_courses.items()), columns=['Course Name', 'Course Description'])

    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False)

    print('Courses cleaned and stored in xlsx file')

core_folder_path = './core_courses'
core_output_file = './Datasets/cleaned_core_courses.xlsx'

elective_folder_path = './elective_courses'
elective_output_file = './Datasets/cleaned_elective_courses.xlsx'

all_folder_path = './all_courses'
all_output_file = './Datasets/cleaned_all_courses.xlsx'

#process_files(core_folder_path, core_output_file)
#process_files(elective_folder_path, elective_output_file)
process_files(all_folder_path, all_output_file)