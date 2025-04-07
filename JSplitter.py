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

import json
import math
import os

def split_json_file(input_file, output_directory, objects_per_file=100):
    """
    Split a JSON file into multiple files with specified number of objects per file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_directory (str): Directory where split files will be saved
        objects_per_file (int): Number of objects per output file (default: 100)
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Read the input JSON file with proper encoding for Arabic
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure we're working with a list
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of objects")
    
    # Calculate the number of files needed
    total_objects = len(data)
    num_files = math.ceil(total_objects / objects_per_file)
    
    # Split and write the data
    for i in range(num_files):
        start_idx = i * objects_per_file
        end_idx = min((i + 1) * objects_per_file, total_objects)
        
        # Create subset of data
        subset = data[start_idx:end_idx]
        
        # Generate output filename
        output_file = os.path.join(
            output_directory,
            f'split_{i + 1:03d}.json'  # Creates files like split_001.json
        )
        
        # Write the subset to a new file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        
        print(f"Created {output_file} with {len(subset)} objects")

# Example usage
if __name__ == "__main__":
    split_json_file(
        input_file="",
        output_directory="",
        objects_per_file=
    )
