#!/usr/bin/env python3
"""
Script to consolidate data from individual subject files (E01.xls to E81.xls)
Each file contains balance test data for a single subject.
"""

import os
import pandas as pd
import re
from pathlib import Path

def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., 'E01.xls' -> 'E01')"""
    return filename.split('.')[0]

def parse_file(filepath):
    """
    Parse a single subject file and extract both header info and performance data
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Extract header information
    header_info = {}
    data_section_start = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('Test Name:'):
            header_info['test_name'] = line.split(':', 1)[1].strip()
        elif line.startswith('Person Name:'):
            header_info['person_name'] = line.split(':', 1)[1].strip()
        elif line.startswith('Person Height:'):
            header_info['person_height'] = line.split(':', 1)[1].strip()
        elif line.startswith('Person Weight[lbs]:'):
            header_info['person_weight_lbs'] = line.split(':', 1)[1].strip()
        elif line.startswith('Created:'):
            header_info['created'] = line.split(':', 1)[1].strip()
        elif line.startswith('Protocol:'):
            header_info['protocol'] = line.split(':', 1)[1].strip()
        elif line.startswith('Perf. time[s]:'):
            header_info['performance_time_s'] = line.split(':', 1)[1].strip()
        elif line.startswith('Romberg Quotient:'):
            header_info['romberg_quotient'] = line.split(':', 1)[1].strip()
        elif 'Performance Name' in line and 'Trace Length' in line:
            # Found the header of the data table
            data_section_start = i
            break
    
    if data_section_start is None:
        print(f"Warning: No data section found in {filepath}")
        return None, None
    
    # Read the data section
    try:
        # Read from the header line onwards
        data_lines = lines[data_section_start:]
        
        # Clean up the lines - remove trailing tabs and whitespace
        cleaned_lines = []
        for line in data_lines:
            # Remove trailing whitespace and tabs, but keep the line structure
            cleaned_line = line.rstrip()
            # Remove trailing tab if present
            if cleaned_line.endswith('\t'):
                cleaned_line = cleaned_line.rstrip('\t')
            cleaned_lines.append(cleaned_line)
        
        # Create a temporary file-like object from the cleaned lines
        from io import StringIO
        data_text = '\n'.join(cleaned_lines)
        
        # Read as tab-separated values, ensuring proper handling of the data
        df = pd.read_csv(StringIO(data_text), sep='\t', encoding='utf-8')
        
        # Clean up any extra whitespace in column names
        df.columns = df.columns.str.strip()
        
        return header_info, df
    
    except Exception as e:
        print(f"Error parsing data section in {filepath}: {e}")
        return header_info, None

def consolidate_all_files(directory_path):
    """
    Consolidate data from all E*.xls files in the directory
    """
    directory = Path(directory_path)
    
    # Find all E*.xls files (including duplicated ones like E78-duplicated.xls)
    pattern = re.compile(r'^E\d+.*\.xls$')
    files = [f for f in directory.iterdir() if f.is_file() and pattern.match(f.name)]
    files.sort()  # Sort to ensure consistent order
    
    print(f"Found {len(files)} subject files to process")
    
    # Lists to store consolidated data
    all_header_data = []
    all_performance_data = []
    
    for filepath in files:
        subject_id = extract_subject_id(filepath.name)
        print(f"Processing {filepath.name} (Subject: {subject_id})")
        
        header_info, performance_df = parse_file(filepath)
        
        if header_info:
            # Add subject ID to header info
            header_info['subject_id'] = subject_id
            header_info['filename'] = filepath.name
            all_header_data.append(header_info)
        
        if performance_df is not None and not performance_df.empty:
            # Add subject ID to each row of performance data
            performance_df['subject_id'] = subject_id
            performance_df['filename'] = filepath.name
            all_performance_data.append(performance_df)
    
    # Combine all data
    if all_header_data:
        header_df = pd.DataFrame(all_header_data)
        print(f"\nConsolidated header data: {len(header_df)} subjects")
    else:
        header_df = pd.DataFrame()
        print("\nNo header data found")
    
    if all_performance_data:
        performance_df_combined = pd.concat(all_performance_data, ignore_index=True)
        print(f"Consolidated performance data: {len(performance_df_combined)} total records")
    else:
        performance_df_combined = pd.DataFrame()
        print("No performance data found")
    
    return header_df, performance_df_combined

def main():
    # Get the current directory (where the script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '1. Raw Data From HUR Platform')
    
    print(f"Processing files in: {data_dir}")
    
    # Consolidate all data
    header_df, performance_df = consolidate_all_files(data_dir)
    
    # Save the consolidated data
    if not header_df.empty:
        header_output = os.path.join(current_dir, 'consolidated_header_data.csv')
        header_df.to_csv(header_output, index=False)
        print(f"\nHeader data saved to: {header_output}")
        print(f"Header data columns: {list(header_df.columns)}")
        print(f"Sample header data:")
        print(header_df.head())
    
    if not performance_df.empty:
        performance_output = os.path.join(current_dir, 'consolidated_performance_data.csv')
        performance_df.to_csv(performance_output, index=False)
        print(f"\nPerformance data saved to: {performance_output}")
        print(f"Performance data columns: {list(performance_df.columns)}")
        print(f"Sample performance data:")
        print(performance_df.head())
        
        # Create a summary by condition
        if 'Performance Name' in performance_df.columns:
            print(f"\nSummary by test condition:")
            summary = performance_df.groupby('Performance Name').agg({
                'subject_id': 'count',
                'Trace Length["]': ['mean', 'std'],
                'C90 Area["^2]': ['mean', 'std'],
                'STD Velocity["/s]': ['mean', 'std']
            }).round(3)
            print(summary)
    
    print(f"\nConsolidation complete!")
    print(f"Total subjects processed: {len(header_df) if not header_df.empty else 0}")
    print(f"Total performance records: {len(performance_df) if not performance_df.empty else 0}")

if __name__ == "__main__":
    main()
