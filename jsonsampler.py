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
import random
import argparse
from pathlib import Path
from typing import List, Any
import codecs

def sample_json(input_file: str, output_file: str, k: int) -> None:
    """
    Randomly sample K objects from a UTF-8 encoded JSON file and write them to a new file.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        k (int): Number of objects to sample
    
    Raises:
        ValueError: If K is larger than the number of objects in the input file
        JSONDecodeError: If input file is not valid JSON
        FileNotFoundError: If input file doesn't exist
        UnicodeDecodeError: If file is not properly UTF-8 encoded
    """
    # Read the input JSON file with explicit UTF-8 encoding
    try:
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"The file {input_file} is not properly UTF-8 encoded")
    except json.JSONDecodeError:
        raise ValueError(f"The file {input_file} is not a valid JSON file")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {input_file} was not found")
    
    # Handle different JSON structures
    if isinstance(data, list):
        objects = data
    elif isinstance(data, dict):
        objects = [data]
    else:
        raise ValueError("JSON file must contain an object or array of objects")
    
    # Validate K
    if k > len(objects):
        raise ValueError(f"Cannot sample {k} items from a collection of {len(objects)} items")
    
    # Perform reservoir sampling if the JSON is very large
    sampled_objects = random.sample(objects, k)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write sampled objects to output file with UTF-8 encoding
    try:
        with codecs.open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_objects, f, indent=2, ensure_ascii=False)
    except UnicodeEncodeError:
        raise UnicodeEncodeError(f"Failed to encode some characters to UTF-8 when writing to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Sample K random objects from a UTF-8 encoded JSON file')
    parser.add_argument('input_file', help='Input JSON file path (UTF-8 encoded)')
    parser.add_argument('k', type=int, help='Number of objects to sample')
    parser.add_argument('--output', default='K_sample.json', 
                      help='Output JSON file path (default: K_sample.json)')
    
    args = parser.parse_args()
    
    try:
        sample_json(args.input_file, args.output, args.k)
        print(f"Successfully sampled {args.k} objects to {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
