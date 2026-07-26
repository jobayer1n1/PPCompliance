import os
import csv
import json
from pathlib import Path

def check_compliance():
    # Resolve paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "map.json"
    input_path = script_dir.parent / "privacy_conclude_result_last.csv"
    output_path = script_dir / "privacy_compliance_result.csv"

    print(f"Reading map from: {map_path}")
    if not map_path.exists():
        print(f"Error: Map file {map_path} does not exist.")
        return

    with open(map_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    gdpr_mapping = mapping_data.get("GDPR_MAPPING", {})

    print(f"Reading input from: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    compliance_rows = []
    
    # Read the conclusion file
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            
            ext_id = row[0]
            # Convert values (counts for labels 0 to 20) to integers
            try:
                counts = [int(val) for val in row[1:]]
            except ValueError as e:
                print(f"Skipping row with invalid integer value for extension {ext_id}: {e}")
                continue
            
            if len(counts) < 21:
                print(f"Warning: Row for extension {ext_id} has fewer than 21 labels. Padding with zeros.")
                counts += [0] * (21 - len(counts))
            
            # Evaluate compliance as 1 or 0
            dc = 1 if any(counts[l] > 0 for l in gdpr_mapping.get("DC", [])) else 0
            dr = 1 if any(counts[l] > 0 for l in gdpr_mapping.get("DR", [])) else 0
            du = 1 if any(counts[l] > 0 for l in gdpr_mapping.get("DU", [])) else 0
            ci = 1 if any(counts[l] > 0 for l in gdpr_mapping.get("CI", [])) else 0
            dsr = 1 if any(counts[l] > 0 for l in gdpr_mapping.get("DSR", [])) else 0
            
            compliance_rows.append([ext_id, dc, dr, du, ci, dsr])
            
    # Write the output file
    headers = ["ext", "DC", "DR", "DU", "CI", "DSR"]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(compliance_rows)
        
    print(f"Compliance results successfully written to: {output_path}")

if __name__ == "__main__":
    check_compliance()
