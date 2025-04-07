import json
import sys

def merge_json_files(json1_path, json2_path, output_path):
    # Load JSON files
    try:
        with open(json1_path, 'r', encoding='utf-8') as file1:
            json1_data = json.load(file1)
    except Exception as e:
        print(f"Error reading first JSON file: {e}")
        return

    try:
        with open(json2_path, 'r', encoding='utf-8') as file2:
            json2_data = json.load(file2)
    except Exception as e:
        print(f"Error reading second JSON file: {e}")
        return

    # Check if the input is a list or a single object
    if not isinstance(json1_data, list):
        json1_data = [json1_data]
    if not isinstance(json2_data, list):
        json2_data = [json2_data]

    # Create dictionary for quick lookup of json2 items by id
    # Convert all ids to strings for consistent comparison
    json2_dict = {str(item["id"]): item for item in json2_data}
    
    # Process each item in json1
    merged_data = []
    for item in json1_data:
        # Convert json1 numeric id to string for comparison
        item_id = str(item.get("id"))
        
        # Check if this item has a corresponding entry in json2
        if item_id in json2_dict:
            # Remove specified keys
            if "judgement" in item:
                del item["judgement"]
            if "reasoning" in item:
                del item["reasoning"]
            
            # Replace claim with the one from json2
            item["claim"] = json2_dict[item_id]["claim"]
            
            # Add type from json2
            item["type"] = json2_dict[item_id]["type"]
            
            merged_data.append(item)
        else:
            print(f"Warning: Item with id {item_id} not found in the second JSON file. Skipping.")
    
    # Write the merged data to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, indent=4, ensure_ascii=False)
        print(f"Successfully merged JSON files. Output saved to {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py json1_path json2_path output_path")
    else:
        merge_json_files(sys.argv[1], sys.argv[2], sys.argv[3])

#python script.py input1.json input2.json output.json