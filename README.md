# face_deid_ct
Function to de-identify the face of patients in head CTs


# Drown Volume Python Function

The `drown_volume` function is a Python function designed to process DICOM files from a specified directory. The function performs several operations including binarization, retrieving the largest connected component, dilation, and applying a mask. Following these operations, random values are applied to the dilated volume based on a unique values list which is obtained from the masked volume.

## Usage

```python
drown_volume(in_path, out_path=None, replacer='face')
```

Parameters:

in_path (str): The path to the directory containing the input DICOM files.
out_path (str, optional): The path to the directory where the output DICOM files will be saved. If not provided, the output files will be saved in the input directory appended by "_d".
replacer (str, optional): Indicates what kind of pixels are going to be replaced. Default is 'face'.
'face': replaces air and face with random values that are found in the skin and subcutaneous fat.
'air': replaces air and face with -1000 HU.
int: replaces air and face with int HU.
Returns:

The function does not return any value. Instead, it saves new DICOM files in the specified or default directory and prints the total elapsed time of the operation.

Examples
python
Copy code
drown_volume('/path/to/dicom/files')
This will process the DICOM files in the specified directory and save the output files in the same directory with "_d" appended to their names. The 'face' pixels will be replaced with random values found in the skin and subcutaneous fat.

python
Copy code
drown_volume('/path/to/dicom/files', out_path='/path/to/output/directory', replacer='air')
This will process the DICOM files in the specified directory and save the output files in a different directory. The 'air' and 'face' pixels will be replaced with -1000 HU.

Contribution
Feel free to fork the project, submit issues, or make pull requests.

vbnet
Copy code
This README file gives a brief overview of what the `drown_vol
