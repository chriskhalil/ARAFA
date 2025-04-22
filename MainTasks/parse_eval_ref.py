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
from typing import List, Tuple
import json

def parse_document(text: str) -> List[Tuple[str, str]]:
    """
    Parse a document to extract all ID and Judgement pairs in order of appearance.
    Supports multiple formats:
    1. Original regex-based extraction
    2. JSON-like formats with or without quotes
    Returns a list of (id, judgement) tuples.
    """
    results = []
    # Split into lines while preserving order
    lines = text.split('\n')
    
    current_id = None
    current_block = []
    parsing_block = False
    
    for line in lines:
        # Check for start of JSON-like block
        if line.strip() == '{':
            parsing_block = True
            current_block = []
            continue
        
        # Check for end of JSON-like block
        if line.strip() == '}':
            if current_block:
                # Try to parse block as JSON or extract values
                block_text = '\n'.join(current_block)
                
                # Try direct JSON parsing first
                try:
                    # Remove quotes around keys and values if present
                    block_text = block_text.replace('"', '')
                    block_dict = json.loads('{' + block_text + '}')
                    
                    # Extract ID and Judgement
                    if 'ID' in block_dict and 'Judgement' in block_dict:
                        results.append((str(block_dict['ID']), block_dict['Judgement']))
                except (json.JSONDecodeError, TypeError):
                    # Fallback to regex parsing for JSON-like formats
                    id_match = re.search(r'ID:\s*["\']?(\d+)["\']?', block_text)
                    judgement_match = re.search(r'Judgement:\s*["\']?(NEI|Supported)["\']?', block_text)
                    
                    if id_match and judgement_match:
                        results.append((id_match.group(1), judgement_match.group(1)))
            
            parsing_block = False
            current_block = []
            current_id = None
            continue
        
        # If parsing a block, collect its lines
        if parsing_block:
            current_block.append(line.strip())
        
        # Original regex-based extraction
        # Check for ID first
        id_match = re.search(r'ID:\s*(\d+)', line)
        if id_match:
            current_id = id_match.group(1)
            continue
            
        if current_id:
            # Look for judgment in current line
            judgement_match = re.search(r'\*?\*?Judgement:\*?\*?\s*(NEI|Supported)[^\n]*', line)
            if not judgement_match:
                judgement_match = re.search(r'\d+\.\s*\*?\*?Judgement:\*?\*?\s*(NEI|Supported)[^\n]*', line)
            
            if judgement_match:
                judgement_type = re.search(r'(NEI|Supported)', judgement_match.group(0))
                if judgement_type:
                    # Append in order of appearance
                    results.append((current_id, judgement_type.group(1)))
                    current_id = None
    
    return results

# The rest of the code remains the same as in the original script
def natural_sort_key(s: str) -> List:
    """
    Return a key for natural sorting of filenames.
    Converts 'file2.txt' and 'file10.txt' into a format that sorts correctly.
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def process_directory(directory_path: str) -> List[Tuple[str, str]]:
    """
    Process all .txt files in natural sort order and preserve ID-Judgement pair order.
    Returns a list of (id, judgement) tuples.
    """
    all_results = []
    txt_files = []
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return all_results
    
    # Collect all .txt files from directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                txt_files.append((file, file_path))
    
    # Sort files using natural sort
    txt_files.sort(key=lambda x: natural_sort_key(x[0]))
    
    # Process each file in order
    for filename, file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_results = parse_document(content)
                if file_results:
                    # Extend preserves order of pairs within the file
                    all_results.extend(file_results)
                    print(f"Successfully parsed: {filename} - Found {len(file_results)} ID-Judgement pairs")
                else:
                    print(f"No ID-Judgement pairs found in: {filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
    
    return all_results

def save_results(results: List[Tuple[str, str]], output_file: str = 'output.csv') -> None:
    """
    Save the results in CSV format, preserving the original order.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Id,Judgement\n')
        # Write pairs in the order they were found
        for id_num, judgement in results:
            f.write(f'{id_num},{judgement}\n')

def main():
    directory_path = ''
    
    print(f"\nProcessing files in: {directory_path}")
    results = process_directory(directory_path)
    
    if results:
        output_file = 'output.csv'
        save_results(results)
        print(f"\nProcessed {len(results)} ID-Judgement pairs successfully.")
        print(f"Results saved to: {output_file}")
    else:
        print("\nNo valid ID-Judgement pairs were found or processed.")

if __name__ == '__main__':
    main()
