import os
import pandas as pd

# Define paths
input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "list_attr_cloth.txt")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)  
output_csv_path = os.path.join(output_folder, "attributes_clothes_fine.csv")

def parse_list_attr_cloth(file_path):
    with open(file_path, 'r') as file:
    
        num_attributes = int(file.readline().strip())

       
        headers = file.readline().strip().split()

      
        data = []
        for line in file:
            parts = line.strip().split()
            attribute_name = " ".join(parts[:-1])  
            attribute_type = int(parts[-1])      
            data.append([attribute_name, attribute_type])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=headers)
    return df

# Parse the file
df = parse_list_attr_cloth(input_file_path)

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())  