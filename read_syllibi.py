import PyPDF2
import re
import os
import pandas as pd

def clean_course(course):
    if pd.isnull(course):
        return ""
        
    # Convert to lowercase
    course = course.lower()
    
    # Remove extra whitespace (including newlines)
    course = re.sub(r'\s+', ' ', course)
    
    # Remove any non-printable characters that might cause Excel issues
    course = ''.join(char for char in course if char.isprintable())
    
    return course.strip()

cleaned_courses = {}

def process_files(folder_path, output_path):
    print(f'Number of files: {len(os.listdir(folder_path))}')
    for filename in os.listdir(folder_path):
        try:
            file_path = os.path.join(folder_path, filename)
            reader = PyPDF2.PdfReader(file_path)
            
            course_text = ''
            for page in reader.pages:
                course_text += page.extract_text()
                
            # Clean the extracted course text
            cleaned_course = clean_course(course_text)
            
            course_name = os.path.splitext(filename)[0]  # Remove the .pdf extension
            cleaned_courses[course_name] = cleaned_course

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue

    df = pd.DataFrame(list(cleaned_courses.items()), columns=['Course Name', 'Course Description'])

    try:
        # Try saving to Excel
        df.to_excel(output_path, index=False)
    except Exception as e:
        print(f"Error saving to Excel: {str(e)}")
        # Fallback to CSV if Excel fails
        csv_path = output_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved as CSV instead: {csv_path}")

    print('Courses cleaned and stored in file')

core_folder_path = './core_courses'
core_output_file = './Datasets/cleaned_core_courses.xlsx'

elective_folder_path = './elective_courses'
elective_output_file = './Datasets/cleaned_elective_courses.xlsx'

all_folder_path = './all_courses'
all_output_file = './Datasets/cleaned_all_courses.xlsx'

process_files(core_folder_path, core_output_file)
process_files(elective_folder_path, elective_output_file)
process_files(all_folder_path, all_output_file)