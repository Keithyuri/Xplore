import os

# Folder path
folder_path = r"C:\Users\Keith Chin\Radiography Project\backend\Sample images\Images to be demonstrated"

# Get all PNG files
png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

# Sort files to keep order consistent
png_files.sort()

# Rename files
for index, filename in enumerate(png_files, start=1):
    new_name = f"chest_test_original_{index}.png"
    
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_name}")

print("Renaming completed!")