#!/usr/bin/env python3
"""
MOCA Score Classification Script

This script reads MOCA scores from SeniorProfile.csv and classifies them into:
- Healthy: >=26
- Mild: 18-25  
- Moderate: 10-17
- Severe: <10

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def classify_moca_score(score):
    """
    Classify MOCA score into cognitive categories.
    
    Parameters:
    score (int/float): MOCA score
    
    Returns:
    str: Classification category
    """
    if score >= 26:
        return "Healthy"
    elif score >= 18:
        return "Mild"
    elif score >= 10:
        return "Moderate"
    else:
        return "Severe"

def analyze_moca_classifications(csv_file_path):
    """
    Analyze and visualize MOCA score classifications.
    
    Parameters:
    csv_file_path (str): Path to the SeniorProfile.csv file
    """
    print("=== MOCA SCORE CLASSIFICATION ANALYSIS ===\n")
    
    # Load the data
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✓ Successfully loaded data from {csv_file_path}")
        print(f"  Dataset shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: File {csv_file_path} not found")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Classify MOCA scores
    df['Classification'] = df['MoCA'].apply(classify_moca_score)
    
    # Display basic statistics
    print(f"\n=== MOCA SCORE STATISTICS ===")
    print(f"Total subjects: {len(df)}")
    print(f"MOCA score range: {df['MoCA'].min()} - {df['MoCA'].max()}")
    print(f"Mean MOCA score: {df['MoCA'].mean():.2f}")
    print(f"Median MOCA score: {df['MoCA'].median():.1f}")
    print(f"Standard deviation: {df['MoCA'].std():.2f}")
    
    # Classification summary
    print(f"\n=== CLASSIFICATION SUMMARY ===")
    classification_counts = df['Classification'].value_counts()
    classification_percentages = df['Classification'].value_counts(normalize=True) * 100
    
    for category in ['Healthy', 'Mild', 'Moderate', 'Severe']:
        count = classification_counts.get(category, 0)
        percentage = classification_percentages.get(category, 0)
        print(f"{category:>10}: {count:>3} subjects ({percentage:>5.1f}%)")
    
    # Detailed breakdown
    print(f"\n=== DETAILED BREAKDOWN ===")
    print("Category definitions:")
    print("  Healthy:  MOCA ≥ 26")
    print("  Mild:     MOCA 18-25")
    print("  Moderate: MOCA 10-17")
    print("  Severe:   MOCA < 10")
    
    print(f"\nSubjects by category:")
    for category in ['Healthy', 'Mild', 'Moderate', 'Severe']:
        subjects_in_category = df[df['Classification'] == category]
        if len(subjects_in_category) > 0:
            print(f"\n{category} ({len(subjects_in_category)} subjects):")
            for _, subject in subjects_in_category.iterrows():
                print(f"  {subject['SubjectID']}: MOCA = {subject['MoCA']}")
        else:
            print(f"\n{category}: No subjects in this category")
    
    # Create visualizations
    create_visualizations(df)
    
    # Save results
    save_results(df)
    
    return df

def create_visualizations(df):
    """
    Create visualizations for MOCA score classifications.
    
    Parameters:
    df (DataFrame): DataFrame with MOCA scores and classifications
    """
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MOCA Score Classification Analysis', fontsize=16, fontweight='bold')
    
    # 1. Distribution of MOCA scores
    axes[0, 0].hist(df['MoCA'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(df['MoCA'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["MoCA"].mean():.1f}')
    axes[0, 0].axvline(df['MoCA'].median(), color='orange', linestyle='--', 
                       label=f'Median: {df["MoCA"].median():.1f}')
    axes[0, 0].set_title('Distribution of MOCA Scores')
    axes[0, 0].set_xlabel('MOCA Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Classification pie chart
    classification_counts = df['Classification'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    axes[0, 1].pie(classification_counts.values, labels=classification_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0, 1].set_title('Distribution by Classification')
    
    # 3. Bar chart of classifications
    classification_counts.plot(kind='bar', ax=axes[1, 0], color=colors)
    axes[1, 0].set_title('Number of Subjects by Classification')
    axes[1, 0].set_xlabel('Classification')
    axes[1, 0].set_ylabel('Number of Subjects')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot by classification
    df.boxplot(column='MoCA', by='Classification', ax=axes[1, 1])
    axes[1, 1].set_title('MOCA Scores by Classification')
    axes[1, 1].set_xlabel('Classification')
    axes[1, 1].set_ylabel('MOCA Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('moca_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizations created and saved as 'moca_classification_analysis.png'")

def save_results(df):
    """
    Save classification results to files.
    
    Parameters:
    df (DataFrame): DataFrame with MOCA scores and classifications
    """
    print(f"\n=== SAVING RESULTS ===")
    
    # Save classified data
    output_file = 'moca_classifications.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Classified data saved to '{output_file}'")
    
    # Save summary statistics
    summary_file = 'moca_classification_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MOCA SCORE CLASSIFICATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total subjects: {len(df)}\n")
        f.write(f"MOCA score range: {df['MoCA'].min()} - {df['MoCA'].max()}\n")
        f.write(f"Mean MOCA score: {df['MoCA'].mean():.2f}\n")
        f.write(f"Median MOCA score: {df['MoCA'].median():.1f}\n")
        f.write(f"Standard deviation: {df['MoCA'].std():.2f}\n\n")
        
        f.write("CLASSIFICATION BREAKDOWN:\n")
        f.write("-" * 25 + "\n")
        classification_counts = df['Classification'].value_counts()
        classification_percentages = df['Classification'].value_counts(normalize=True) * 100
        
        for category in ['Healthy', 'Mild', 'Moderate', 'Severe']:
            count = classification_counts.get(category, 0)
            percentage = classification_percentages.get(category, 0)
            f.write(f"{category:>10}: {count:>3} subjects ({percentage:>5.1f}%)\n")
        
        f.write(f"\nCATEGORY DEFINITIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("Healthy:  MOCA ≥ 26\n")
        f.write("Mild:     MOCA 18-25\n")
        f.write("Moderate: MOCA 10-17\n")
        f.write("Severe:   MOCA < 10\n")
    
    print(f"✓ Summary statistics saved to '{summary_file}'")

def main():
    """Main function to run the MOCA classification analysis."""
    csv_file = 'SeniorProfile.csv'
    
    print("MOCA Score Classification Script")
    print("=" * 50)
    
    # Run the analysis
    results_df = analyze_moca_classifications(csv_file)
    
    if results_df is not None:
        print(f"\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        print(f"Check the generated files:")
        print(f"  - moca_classifications.csv (classified data)")
        print(f"  - moca_classification_summary.txt (summary statistics)")
        print(f"  - moca_classification_analysis.png (visualizations)")
    else:
        print(f"\n❌ Analysis failed. Please check the input file.")

if __name__ == "__main__":
    main()
