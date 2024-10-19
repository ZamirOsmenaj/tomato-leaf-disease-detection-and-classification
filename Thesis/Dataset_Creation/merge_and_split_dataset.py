import os
import shutil
import random

def create_directory_structure(base_output_dir):
    """Creates the main directory and subdirectories for training, validation, and test sets."""
    os.makedirs(base_output_dir, exist_ok=True)
    for subset_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_output_dir, subset_name), exist_ok=True)

def list_files_in_directory(directory_path):
    """Lists all files in the given directory."""
    return [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file_name))]

def combine_and_shuffle_files(train_dir, val_dir):
    """Combines files from training and validation directories, then shuffles them."""
    all_file_paths = []
    if os.path.exists(train_dir):
        all_file_paths.extend(list_files_in_directory(train_dir))
    if os.path.exists(val_dir):
        all_file_paths.extend(list_files_in_directory(val_dir))
    random.shuffle(all_file_paths)
    return all_file_paths

def split_files(file_list, train_split_ratio, test_split_ratio):
    """Splits files into training, validation, and test sets based on provided ratios."""
    total_files_count = len(file_list)
    train_count = int(train_split_ratio * total_files_count)
    test_count = train_count + int(test_split_ratio * total_files_count)
    
    train_files = file_list[:train_count]
    test_files = file_list[train_count:test_count]
    val_files = file_list[test_count:]
    
    return train_files, val_files, test_files

def copy_files_to_subfolder(file_paths, target_base_dir, category_name):
    """Copies files to the specified target subdirectory."""
    target_category_dir = os.path.join(target_base_dir, category_name)
    os.makedirs(target_category_dir, exist_ok=True)
    for file_path in file_paths:
        shutil.copy(file_path, target_category_dir)

def main(input_base_dir, output_base_dir, train_ratio=0.8, test_ratio=0.1):
    """Main function to orchestrate the split and copying of files into train, validation, and test sets."""
    # Create the directory structure for the output
    create_directory_structure(output_base_dir)

    # Paths to the original subdirectories
    train_data_dir = os.path.join(input_base_dir, 'train')
    val_data_dir = os.path.join(input_base_dir, 'val')

    # List all subcategories (e.g., classes or types of leaves)
    subcategories = [category for category in os.listdir(val_data_dir) if os.path.isdir(os.path.join(val_data_dir, category))]

    for subcategory in subcategories:
        subcategory_train_dir = os.path.join(train_data_dir, subcategory)
        subcategory_val_dir = os.path.join(val_data_dir, subcategory)
        
        # Combine and shuffle files from both directories
        combined_file_list = combine_and_shuffle_files(subcategory_train_dir, subcategory_val_dir)
        
        if not combined_file_list:
            print(f"No files found in subdirectories for {subcategory}. Skipping...")
            continue

        # Split files into training, validation, and test sets
        train_files, val_files, test_files = split_files(combined_file_list, train_ratio, test_ratio)

        # Copy the files to the respective output directories
        copy_files_to_subfolder(train_files, os.path.join(output_base_dir, 'train'), subcategory)
        copy_files_to_subfolder(val_files, os.path.join(output_base_dir, 'val'), subcategory)
        copy_files_to_subfolder(test_files, os.path.join(output_base_dir, 'test'), subcategory)

    print("Files have been successfully copied and split into train, validation, and test sets.")

if __name__ == "__main__":
    input_base_dir = 'tomato'
    output_base_dir = '../Tomato_Leaves_Dataset'
    main(input_base_dir, output_base_dir)
