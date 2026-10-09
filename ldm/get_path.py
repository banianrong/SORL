import os

def list_files_to_txt(dir_path, output_txt_path):
    """
    Write the paths of all files in a directory to a text file, one path per line.

    Parameters:
    - dir_path: Path to the directory whose files will be listed.
    - output_txt_path: Path to the output text file.
    """
    with open(output_txt_path, 'w') as f:
        for root, _, files in os.walk(dir_path):
            for file in files:
                f.write(os.path.join(root, file) + '\n')
                print("Done.")

# Example usage
dir_path = "/path/to/celebahq/test/" # Update this path to the directory you want to list
output_txt_path = "/path/to/celebahq_test.txt"  # Update this path to where you want to save the txt file

# Call the function
list_files_to_txt(dir_path, output_txt_path)