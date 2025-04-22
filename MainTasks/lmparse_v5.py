##############################################################################
##############################################################################
#MIT License

#Copyright (c) 2024 Christophe Khalil
#Email: christophe.khalil@outlook.com / cak29@mail.aub.edu

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
##############################################################################
##############################################################################

import re
import os
import json
import argparse
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def get_all_txt_files(directory: str) -> List[str]:
    """
    Recursively get all .txt files from directory and its subdirectories.
    
    Args:
        directory (str): Root directory path
        
    Returns:
        List[str]: List of full paths to all .txt files
        
    Time complexity: O(n) where n is the total number of files and directories
    """
    txt_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        return sorted(txt_files)  # Sort for consistent processing order
    except Exception as e:
        print(f"Error accessing directory {directory}: {str(e)}")
        return []


class DuplicateTracker:
    """
    Class to track and detect duplicates based on evidence and claim pairs.
    Uses hash tables for O(1) lookup performance.
    """
    def __init__(self):
        self.seen_pairs: Set[tuple] = set()
        self.duplicate_count: int = 0
    
    def is_duplicate(self, evidence: str, claim: str) -> bool:
        """
        Check if a given evidence-claim pair is a duplicate.
        Time complexity: O(1) average case
        """
        pair = (evidence.strip(), claim.strip())
        if pair in self.seen_pairs:
            self.duplicate_count += 1
            return True
        self.seen_pairs.add(pair)
        return False

class ArabicTextParser:
    def __init__(self, verbose: bool = False, remove_duplicates: bool = False):
        """
        Initialize parser with settings and enhanced summary tracking.
        """
        self.verbose = verbose
        self.remove_duplicates = remove_duplicates
        self.summary = {
            'total_files': 0,
            'total_initial_tuples': 0,  # Total before duplicate removal
            'total_final_tuples': 0,    # Total after duplicate removal
            'total_malformed': 0,
            'file_stats': [],
            'total_duplicates': 0
        }

    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def clean_text(self, content: str) -> str:
        """Remove header information and clean the text."""
        lines = content.split('\n')
        content = '\n'.join(lines)
        content = re.sub(r'-+', '', content)
        return content.strip()

    def extract_number_and_text(self, line: str) -> Optional[Tuple[int, str]]:
        """
        Extract number and text from a line with flexible separator formats and spacing.
        Time complexity: O(n) where n is the length of the line
        """
        patterns = [
            r'^\s*(\d+)\s*[.:_\-]\s*"(.*?)"\s*$',  # Quoted text
            r'^\s*(\d+)\s*[.:_\-]\s*(.+?)\s*$'     # Unquoted text
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                try:
                    number = int(match.group(1))
                    text = match.group(2).strip()
                    if text:
                        return (number, text)
                except ValueError:
                    continue
        
        return None

    def parse_content(self, content: str, source: str) -> Tuple[List[Dict], int]:
        """
        Parse content into structured data and track malformed entries.
        Time complexity: O(n) where n is the number of lines
        """
        cleaned_content = self.clean_text(content)
        lines = [line.strip() for line in cleaned_content.split('\n') if line.strip()]
        
        valid_entries = {}
        for line in lines:
            extracted = self.extract_number_and_text(line)
            if extracted:
                number, text = extracted
                valid_entries[number] = text
        
        parsed_data = []
        malformed_count = 0
        max_number = max(valid_entries.keys()) if valid_entries else 0
        
        for start_num in range(1, max_number + 1, 5):
            if all(i in valid_entries for i in range(start_num, start_num + 5)):
                parsed_section = {
                    'source': source,
                    'evidence': valid_entries[start_num],
                    'claim': valid_entries[start_num + 1],
                    'Entity_in_Claim': valid_entries[start_num + 2],
                    'Co_referenced_in_Evidence': valid_entries[start_num + 3],
                    'Co_referenced_in_Text': valid_entries[start_num + 4]
                }
                parsed_data.append(parsed_section)
            else:
                malformed_count += 1
        
        return parsed_data, malformed_count

    def parse_file(self, file_path: str, duplicate_tracker: DuplicateTracker) -> Tuple[List[Dict], Dict]:
        """
        Parse a single text file and track duplicates.
        Returns parsed data and file statistics.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            filename = os.path.basename(file_path)
            source = os.path.splitext(filename)[0].split('_')[0]
            
            self.log(f"Processing file: {file_path}")
            parsed_data, malformed_count = self.parse_content(content, source)
            
            # Track initial count before duplicate removal
            initial_count = len(parsed_data)
            
            # Track duplicates per file
            if self.remove_duplicates:
                unique_data = []
                for item in parsed_data:
                    if not duplicate_tracker.is_duplicate(item['evidence'], item['claim']):
                        unique_data.append(item)
                file_duplicates = len(parsed_data) - len(unique_data)
                parsed_data = unique_data
            else:
                file_duplicates = 0
                
            file_stats = {
                'filename': filename,
                'initial_tuples': initial_count,
                'duplicates': file_duplicates,
                'net_tuples': len(parsed_data),
                'malformed': malformed_count,
                'status': 'DONE'
            }
            
            return parsed_data, file_stats
            
        except Exception as e:
            self.log(f"Error processing file {file_path}: {str(e)}")
            return [], {}

    def assign_ids(self, data: List[Dict]) -> List[Dict]:
        """
        Assign sequential IDs to the final filtered data.
        """
        for i, item in enumerate(data, start=1):
            item['id'] = i
        return data

    def print_summary(self) -> None:
        """Print detailed summary statistics."""
        # Print individual file statistics
        print("\nFile Statistics:")
        print("-" * 120)
        print(f"{'Filename':<30} {'Initial Tuples':<15} {'Duplicates':<12} {'Net Tuples':<12} "
              f"{'Malformed':<10} {'Status':<10}")
        print("-" * 120)
        
        for stat in self.summary['file_stats']:
            print(f"{stat['filename']:<30} {stat['initial_tuples']:<15} {stat['duplicates']:<12} "
                  f"{stat['net_tuples']:<12} {stat['malformed']:<10} {stat['status']:<10}")

        # Print overall statistics
        print("\nOverall Statistics:")
        print("-" * 50)
        print(f"Total Files Processed: {self.summary['total_files']}")
        print(f"Total Initial Tuples: {self.summary['total_initial_tuples']}")
        print(f"Total Malformed Entries: {self.summary['total_malformed']}")
        print(f"Total Duplicates Found: {self.summary['total_duplicates']}")
        print(f"Final Unique Tuples: {self.summary['total_final_tuples']}")
        
        if self.remove_duplicates:
            duplicate_percentage = (self.summary['total_duplicates'] / 
                                 self.summary['total_initial_tuples'] * 100 
                                 if self.summary['total_initial_tuples'] > 0 else 0)
            print(f"Duplicate Percentage: {duplicate_percentage:.2f}%")

    def process_directory(self, input_path: str, output_path: str, output_format: str = 'json', recursive: bool = False) -> None:
        """
        Process text files and generate output. Can process recursively through subdirectories.
        
        Args:
            input_path (str): Input file or directory path
            output_path (str): Output file path
            output_format (str): Output format ('json' or 'txt')
            recursive (bool): Whether to process subdirectories recursively
            
        Time complexity: O(n * m) where n is number of files and m is average lines per file
        """
        all_parsed_data = []
        duplicate_tracker = DuplicateTracker()
        
        try:
            # Determine files to process
            if os.path.isfile(input_path):
                files_to_process = [input_path]
            else:
                if recursive:
                    self.log(f"Recursively processing directory: {input_path}")
                    files_to_process = get_all_txt_files(input_path)
                    if self.verbose:
                        print(f"Found {len(files_to_process)} .txt files in {input_path} and its subdirectories")
                else:
                    files_to_process = [os.path.join(input_path, f) 
                                      for f in os.listdir(input_path) 
                                      if f.endswith('.txt')]
            
            # Process each file
            for file_path in files_to_process:
                relative_path = os.path.relpath(file_path, input_path)
                self.log(f"Processing file: {relative_path}")
                
                parsed_data, file_stats = self.parse_file(file_path, duplicate_tracker)
                all_parsed_data.extend(parsed_data)
                
                # Update summary
                self.summary['total_files'] += 1
                self.summary['total_initial_tuples'] += file_stats['initial_tuples']
                self.summary['total_malformed'] += file_stats['malformed']
                self.summary['total_duplicates'] += file_stats['duplicates']
                self.summary['file_stats'].append(file_stats)
            
            # Assign IDs to the final filtered data
            all_parsed_data = self.assign_ids(all_parsed_data)
            self.summary['total_final_tuples'] = len(all_parsed_data)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save merged data
            if output_format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_parsed_data, f, ensure_ascii=False, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in all_parsed_data:
                        f.write("---\n")
                        for key, value in item.items():
                            f.write(f"{key}: {value}\n")
                        f.write("---\n")

            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"Error processing directory: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description='Parse Arabic text files containing numbered groups of related text.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single file
    python script.py -i input.txt -o output.json

    # Process all txt files in a directory
    python script.py -i ./input_dir -o output.json

    # Process directory recursively
    python script.py -i ./input_dir -o output.json -r

    # Output in text format with duplicate removal and recursive processing
    python script.py -i input.txt -o output.txt -f txt -s -r
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input file or directory path')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('-f', '--format', choices=['json', 'txt'], default='json',
                      help='Output format (default: json)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-s', '--remove-duplicates', action='store_true',
                      help='Remove duplicate entries based on evidence and claim fields')
    parser.add_argument('-r', '--recursive', action='store_true',
                      help='Recursively process subdirectories')
    
    args = parser.parse_args()
    
    try:
        text_parser = ArabicTextParser(verbose=args.verbose, remove_duplicates=args.remove_duplicates)
        text_parser.process_directory(args.input, args.output, args.format, args.recursive)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
