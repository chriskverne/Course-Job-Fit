
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def analyze_top_courses(comparison_df: pd.DataFrame, n: int = 20) -> Dict:
    """
    Analyzes the top N courses in both general and salary rankings
    
    Args:
        comparison_df: DataFrame with both rankings
        n: Number of top courses to analyze (default 20)
    """
    # Get top N courses from both rankings
    top_salary = comparison_df.nsmallest(n, 'Salary_Rank')[['Course Name', 'Salary_Rank', 'General_Rank']]
    top_general = comparison_df.nsmallest(n, 'General_Rank')[['Course Name', 'General_Rank', 'Salary_Rank']]
    
    # Find overlapping courses
    top_salary_courses = set(top_salary['Course Name'])
    top_general_courses = set(top_general['Course Name'])
    common_courses = top_salary_courses.intersection(top_general_courses)
    
    # Calculate overlap statistics
    overlap_stats = {
        'total_common': len(common_courses),
        'overlap_percentage': (len(common_courses) / n) * 100,
        'unique_to_salary': len(top_salary_courses - common_courses),
        'unique_to_general': len(top_general_courses - common_courses)
    }
    
    # Analyze common courses
    common_course_details = []
    for course in common_courses:
        salary_rank = float(top_salary[top_salary['Course Name'] == course]['Salary_Rank'].iloc[0])
        general_rank = float(top_general[top_general['Course Name'] == course]['General_Rank'].iloc[0])
        rank_diff = abs(salary_rank - general_rank)
        common_course_details.append({
            'course': course,
            'salary_rank': salary_rank,
            'general_rank': general_rank,
            'rank_difference': rank_diff
        })
    
    # Sort common courses by rank difference
    common_course_details = sorted(common_course_details, key=lambda x: x['rank_difference'])
    
    # Analyze courses unique to each list
    unique_salary_courses = []
    for course in (top_salary_courses - common_courses):
        course_data = comparison_df[comparison_df['Course Name'] == course].iloc[0]
        unique_salary_courses.append({
            'course': course,
            'salary_rank': course_data['Salary_Rank'],
            'general_rank': course_data['General_Rank'],
            'rank_improvement': course_data['General_Rank'] - course_data['Salary_Rank']
        })
    
    unique_general_courses = []
    for course in (top_general_courses - common_courses):
        course_data = comparison_df[comparison_df['Course Name'] == course].iloc[0]
        unique_general_courses.append({
            'course': course,
            'general_rank': course_data['General_Rank'],
            'salary_rank': course_data['Salary_Rank'],
            'rank_decline': course_data['Salary_Rank'] - course_data['General_Rank']
        })
    
    return {
        'overlap_stats': overlap_stats,
        'common_courses': common_course_details,
        'unique_salary_courses': sorted(unique_salary_courses, key=lambda x: x['salary_rank']),
        'unique_general_courses': sorted(unique_general_courses, key=lambda x: x['general_rank'])
    }

def print_analysis_report(analysis_results: Dict, n: int = 20):
    """
    Prints a formatted report of the top courses analysis
    """
    print(f"\nAnalysis of Top {n} Courses")
    print("=" * 50)
    
    # Print overlap statistics
    stats = analysis_results['overlap_stats']
    print(f"\nOverlap Statistics:")
    print(f"- Common courses: {stats['total_common']} ({stats['overlap_percentage']:.1f}%)")
    print(f"- Unique to salary rankings: {stats['unique_to_salary']}")
    print(f"- Unique to general rankings: {stats['unique_to_general']}")
    
    # Print common courses
    print(f"\nCourses in Both Top {n} Lists:")
    print("-" * 50)
    for course in analysis_results['common_courses']:
        print(f"\n{course['course']}:")
        print(f"  Salary Rank: {course['salary_rank']:.1f}")
        print(f"  General Rank: {course['general_rank']:.1f}")
        print(f"  Rank Difference: {course['rank_difference']:.1f}")
    
    # Print unique salary courses
    print(f"\nCourses Unique to Salary Top {n}:")
    print("-" * 50)
    for course in analysis_results['unique_salary_courses']:
        print(f"\n{course['course']}:")
        print(f"  Salary Rank: {course['salary_rank']:.1f}")
        print(f"  General Rank: {course['general_rank']:.1f}")
        print(f"  Rank Improvement: {course['rank_improvement']:.1f}")
    
    # Print unique general courses
    print(f"\nCourses Unique to General Top {n}:")
    print("-" * 50)
    for course in analysis_results['unique_general_courses']:
        print(f"\n{course['course']}:")
        print(f"  General Rank: {course['general_rank']:.1f}")
        print(f"  Salary Rank: {course['salary_rank']:.1f}")
        print(f"  Rank Decline: {course['rank_decline']:.1f}")

def analyze_program_rankings(program_key: str):
    """
    Analyzes rankings for a specific program
    """
    print(f"\nAnalyzing {program_key} Program Rankings")
    print("=" * 50)
    
    # Load data
    general_df = pd.read_excel(programs[program_key])
    salary_df = pd.read_excel(high_paying_programs[program_key])
    
    # Merge the dataframes
    comparison_df = pd.merge(
        general_df[['Course Name', 'Average_Course_Rank']],
        salary_df[['Course Name', 'Average Rank']],
        on='Course Name',
        how='inner'
    )
    
    # Rename columns for clarity
    comparison_df.columns = ['Course Name', 'General_Rank', 'Salary_Rank']
    
    # Run analysis
    analysis_results = analyze_top_courses(comparison_df, n=20)
    print_analysis_report(analysis_results, n=20)
    
    return comparison_df, analysis_results

# Define data paths
programs = {
    'CS': '../compare_models/CS/course_rankings.xlsx',
    'DS': '../compare_models/DS/course_rankings.xlsx',
    'IT': '../compare_models/IT/course_rankings.xlsx',
    'PM': '../compare_models/PM/course_rankings.xlsx',
    'SWE': '../compare_models/SWE/course_rankings.xlsx'
}

high_paying_programs = {
    'CS': '../compare_models/CS/highest_paying_courses.xlsx',
    'DS': '../compare_models/DS/highest_paying_courses.xlsx',
    'IT': '../compare_models/IT/highest_paying_courses.xlsx',
    'PM': '../compare_models/PM/highest_paying_courses.xlsx',
    'SWE': '../compare_models/SWE/highest_paying_courses.xlsx'
}

# Execute analysis
cs_comparison_df, cs_analysis_results = analyze_program_rankings('CS')
# ds_comparison_df, ds_analysis_results = analyze_program_rankings('DS')
# it_comparison_df, it_analysis_results = analyze_program_rankings('IT')
# pm_comparison_df, pm_analysis_results = analyze_program_rankings('PM')
# swe_comparison_df, swe_analysis_results = analyze_program_rankings('SWE')

"""
import pandas as pd
import numpy as np
from typing import Tuple

def analyze_ranking_changes(general_df: pd.DataFrame, salary_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    Analyzes changes in course rankings between general job alignment and high-paying job alignment.
    
    Args:
        general_df: DataFrame with general job alignment rankings
        salary_df: DataFrame with high-paying job alignment rankings
    
    Returns:
        Tuple containing:
        - DataFrame with ranking comparisons
        - Dictionary with summary statistics
    # Merge the dataframes on Course Name
    comparison_df = pd.merge(
        general_df[['Course Name', 'Average_Course_Rank']],
        salary_df[['Course Name', 'Average Rank']],
        on='Course Name',
        how='inner'
    )
    
    # Rename columns for clarity
    comparison_df.columns = ['Course Name', 'General_Rank', 'Salary_Rank']
    
    # Calculate rank changes
    comparison_df['Rank_Change'] = comparison_df['General_Rank'] - comparison_df['Salary_Rank']
    comparison_df['Absolute_Change'] = abs(comparison_df['Rank_Change'])
    
    # Sort by absolute change to identify biggest movers
    comparison_df = comparison_df.sort_values('Absolute_Change', ascending=False)
    
    # Calculate summary statistics
    stats = {
        'total_courses': len(comparison_df),
        'improved_for_salary': len(comparison_df[comparison_df['Rank_Change'] > 0]),
        'worsened_for_salary': len(comparison_df[comparison_df['Rank_Change'] < 0]),
        'unchanged': len(comparison_df[comparison_df['Rank_Change'] == 0]),
        'avg_absolute_change': comparison_df['Absolute_Change'].mean(),
        'median_absolute_change': comparison_df['Absolute_Change'].median(),
        'max_improvement': comparison_df['Rank_Change'].max(),
        'max_decline': comparison_df['Rank_Change'].min(),
        'top_improvers': comparison_df[comparison_df['Rank_Change'] > 0].head(5)[['Course Name', 'Rank_Change']].to_dict('records'),
        'top_decliners': comparison_df[comparison_df['Rank_Change'] < 0].head(5)[['Course Name', 'Rank_Change']].to_dict('records')
    }
    
    return comparison_df, stats

def analyze_top_course_differences(comparison_df, top_n=30):
    # Get top courses from both rankings
    top_salary = comparison_df.nsmallest(top_n, 'Salary_Rank')
    top_general = comparison_df.nsmallest(top_n, 'General_Rank')
    
    # Find overlapping courses
    top_salary_courses = set(top_salary['Course Name'])
    top_general_courses = set(top_general['Course Name'])
    common_courses = top_salary_courses.intersection(top_general_courses)
    
    print(f"\nAnalysis of Top {top_n} Courses:")
    print(f"Courses in both top lists: {len(common_courses)}")
    print(f"Unique to salary top list: {len(top_salary_courses - common_courses)}")
    print(f"Unique to general top list: {len(top_general_courses - common_courses)}")
    
    print("\nCourses appearing in both top lists:")
    for course in common_courses:
        course_data = comparison_df[comparison_df['Course Name'] == course].iloc[0]
        print(f"\n{course}:")
        print(f"  Salary Rank: {course_data['Salary_Rank']:.1f}")
        print(f"  General Rank: {course_data['General_Rank']:.1f}")
        print(f"  Rank Change: {course_data['Rank_Change']:.1f}")
    
    print("\nCourses unique to high-paying jobs top list:")
    salary_only = top_salary_courses - common_courses
    for course in salary_only:
        course_data = comparison_df[comparison_df['Course Name'] == course].iloc[0]
        print(f"\n{course}:")
        print(f"  Salary Rank: {course_data['Salary_Rank']:.1f}")
        print(f"  General Rank: {course_data['General_Rank']:.1f}")
        print(f"  Improvement: {course_data['Rank_Change']:.1f}")

# Run the analysis
# Analyze the CS program data
programs = {
    'CS': '../compare_models/CS/course_rankings.xlsx',
    'DS': '../compare_models/DS/course_rankings.xlsx',
    'IT': '../compare_models/IT/course_rankings.xlsx',
    'PM': '../compare_models/PM/course_rankings.xlsx',
    'SWE': '../compare_models/SWE/course_rankings.xlsx'
}
high_paying_programs = {
    'CS': '../compare_models/CS/highest_paying_courses.xlsx',
    'DS': '../compare_models/DS/highest_paying_courses.xlsx',
    'IT': '../compare_models/IT/highest_paying_courses.xlsx',
    'PM': '../compare_models/PM/highest_paying_courses.xlsx',
    'SWE': '../compare_models/SWE/highest_paying_courses.xlsx'
}

cs_df = pd.read_excel(programs['CS'])
cs_salary_df = pd.read_excel(high_paying_programs['CS'])
comparison_df, stats = analyze_ranking_changes(cs_df, cs_salary_df)

analyze_top_course_differences(comparison_df)
"""