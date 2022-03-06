# PoseCleaner

This code cleans the messy pose sequences (output of AlphaPose) of a single person pose estimation.

## Installation
```
pip install -r requirements.txt
```

## Usage
### Arguments
```
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        folder path of the input files. e.g. data/pose/
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        folder path of the output
```
Example
```
python clean_poses.py -i data/input_poses/ -o data/fixed_poses/
```