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
import os
from typing import List, Dict, Any

def parse_json_file(file_path: str) -> None:
    """
    Parse JSON file and create individual files for each object.
    
    Args:
        file_path (str): Path to the JSON file to parse
    """
    try:
        # Read JSON file with UTF-8 encoding to handle Arabic text
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Create output directory if it doesn't exist
        output_dir = 'evaluation_files'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each object in the JSON array
        for index, item in enumerate(data, start=1):
            create_evaluation_file(item, index, output_dir)
            
        print(f"Successfully processed {len(data)} objects")
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def create_evaluation_file(item: Dict[str, Any], index: int, output_dir: str) -> None:
    """
    Create individual evaluation file for each JSON object.
    
    Args:
        item (Dict[str, Any]): JSON object to process
        index (int): File index number
        output_dir (str): Output directory path
    """
    output_file = os.path.join(output_dir, f'eval_{index}.txt')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write prompt
            f.write("prompt\n")
            f.write("## TEXT START HERE ##\n")
            f.write(item.get('prompt', '') + '\n')
            
            # Write batch content
            f.write("## TEXT START HERE ##\n")
            f.write(item.get('batch_content', '') + '\n')
            
    except Exception as e:
        print(f"Error creating file {output_file}: {str(e)}")

def main():
    """
    Main function to run the JSON parser.
    """
    json_file = ''  #
    parse_json_file(json_file)

if __name__ == "__main__":
    main()
