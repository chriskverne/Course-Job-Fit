import pandas as pd
import numpy as np
from typing import Tuple

def analyze_ranking_changes(general_df: pd.DataFrame, salary_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Analyzes changes in course rankings between general job alignment and high-paying job alignment.
    
    Args:
        general_df: DataFrame with general job alignment rankings
        salary_df: DataFrame with high-paying job alignment rankings
    
    Returns:
        Tuple containing:
        - DataFrame with ranking comparisons
        - Dictionary with summary statistics
    """
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

"""
# Print summary findings
print("\nSummary Statistics:")
print(f"Total courses analyzed: {stats['total_courses']}")
print(f"Courses that improved for high-paying jobs: {stats['improved_for_salary']}")
print(f"Courses that declined for high-paying jobs: {stats['worsened_for_salary']}")
print(f"Courses with unchanged rankings: {stats['unchanged']}")
print(f"\nAverage absolute rank change: {stats['avg_absolute_change']:.2f}")
print(f"Median absolute rank change: {stats['median_absolute_change']:.2f}")
print(f"Largest improvement: {stats['max_improvement']:.2f} positions")
print(f"Largest decline: {stats['max_decline']:.2f} positions")

print("\nTop 5 Courses that Improved Most for High-Paying Jobs:")
for course in stats['top_improvers']:
    print(f"{course['Course Name']}: Improved by {course['Rank_Change']:.2f} positions")

print("\nTop 5 Courses that Declined Most for High-Paying Jobs:")
for course in stats['top_decliners']:
    print(f"{course['Course Name']}: Declined by {abs(course['Rank_Change']):.2f} positions")

# Display full comparison dataframe
print("\nFull Ranking Comparison (sorted by absolute change):")
print(comparison_df)
"""

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
analyze_top_course_differences(comparison_df)