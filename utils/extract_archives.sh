#!/bin/bash

# Define the base directory containing the 'raw_archives' and 'extracted' directories.
# You'll need to adjust this path according to where you place the script.
BASE_DIR_RAW="/gpfsstore/rech/ads/commun/datasets"
BASE_DIR_EXTRACT="/gpfsscratch/rech/ads/commun/datasets"

# Define the directory where the archives are stored and the extraction directory.
ARCHIVES_DIR="${BASE_DIR_RAW}/raw_archives"
EXTRACTED_DIR="${BASE_DIR_EXTRACT}/extracted"

# Create the 'extracted' directory if it doesn't exist.
mkdir -p "${EXTRACTED_DIR}"

# Iterate over the zip files in the 'raw_archives' directory.
for archive in "${ARCHIVES_DIR}"/*.zip; do
    # Extract each archive into the 'extracted' directory.
    # -o to overwrite existing files without prompting.
    unzip -o "$archive" -d "${EXTRACTED_DIR}"
done

echo "Extraction complete."

