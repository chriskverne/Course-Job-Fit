import pandas as pd
core_courses = ['CDA_3102', 'CEN_4010', 'CGS_1920', 'CGS_3095', 'CIS_3950', 'CIS_4951', 'CNT_4713', 'COP_2210', 'COP_3337', 'COP_3530', 'COP_4338', 'COP_4555', 'COP_4610', 'COT_3100', 'ENC_3249', 'MAD_2104']
elective_courses = ['CAP_4052', 'CAP_4104', 'CAP_4453', 'CAP_4506', 'CAP_4612', 'CAP_4630', 'CAP_4641', 'CAP_4710', 'CAP_4770', 'CAP_4830', 'CDA_4625', 'CEN_4021', 'CEN_4072', 'CEN_4083', 'CIS_4203', 'CIS_4731', 'COP_4226', 'COP_4520', 'COP_4534', 'COP_4604', 'COP_4655', 'COP_4710', 'COP_4751', 'COT_3510', 'COT_3541', 'COT_4431', 'COT_4521', 'COT_4601', 'CTS_4408', 'MAD_3301', 'MAD_3401', 'MAD_3512', 'MAD_4203', 'MHF_4302']

programs = {
    'CS': '../compare_models/CS/course_rankings.xlsx',
    'DS': '../compare_models/DS/course_rankings.xlsx',
    'IT': '../compare_models/IT/course_rankings.xlsx',
    'PM': '../compare_models/PM/course_rankings.xlsx',
    'SWE': '../compare_models/SWE/course_rankings.xlsx'
}

def read_program_rankings(file_path, course_list):
    df = pd.read_excel(file_path)
    # Filter only the courses we're interested in
    mask = df['Course Name'].isin(course_list)
    return df[mask][['Course Name', 'Average_Course_Rank']]

def analyze_program_courses(programs, core_courses, elective_courses):
    # Initialize dictionaries to store results
    core_rankings = {}
    elective_rankings = {}
    
    for program, file_path in programs.items():
        # Read core courses
        core_df = read_program_rankings(file_path, core_courses)
        core_rankings[program] = core_df.set_index('Course Name')['Average_Course_Rank']
        
        # Read elective courses
        elective_df = read_program_rankings(file_path, elective_courses)
        elective_rankings[program] = elective_df.set_index('Course Name')['Average_Course_Rank']
    
    # Convert to DataFrames for easier comparison
    core_comparison = pd.DataFrame(core_rankings)
    elective_comparison = pd.DataFrame(elective_rankings)
    
    return core_comparison, elective_comparison

def generate_statistics(df):
    stats = {
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max()
    }
    return pd.DataFrame(stats)

core_df, elective_df = analyze_program_courses(programs, core_courses, elective_courses)

print("--------- core ----------")
print(f'Num core-courses: {len(core_courses)}')
print(core_df['CS'].mean())
print(core_df['DS'].mean())
print(core_df['SWE'].mean())
print(core_df['PM'].mean())
print(core_df['IT'].mean())

print('------------ elective -----------')
print(f'Num elective-courses: {len(elective_courses)}')
print(elective_df['CS'].mean())
print(elective_df['DS'].mean())
print(elective_df['SWE'].mean())
print(elective_df['PM'].mean())
print(elective_df['IT'].mean())