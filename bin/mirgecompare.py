#!/usr/bin/env python


from mirgecom.simutil import\
    compare_files_vtu, compare_files_xdmf, compare_files_hdf5

# run fidelity check
if __name__ == "__main__":
    import argparse
    import os

    # read in file and comparison info from command line
    parser = argparse.ArgumentParser(
        description="Process files to perform fidelity check")
    parser.add_argument("files", nargs=2, type=str)
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()

    first_file = args.files[0]
    second_file = args.files[1]

    # check for valid file path
    if not os.path.exists(first_file):
        raise ValueError(f"Fidelity test failed: {first_file} not found")
    if not os.path.exists(second_file):
        raise ValueError(f"Fidelity test failed: {second_file} not found")

    file_split = os.path.splitext(first_file)[1]
    file_type = file_split[1:]  # remove dot

    user_tolerance = 1e-12
    if args.tolerance:
        user_tolerance = args.tolerance

    # use appropriate comparison function for file type
    if file_type == "vtu" or file_type == "pvtu":
        compare_files_vtu(first_file, second_file, file_type, user_tolerance)
    elif file_type == "xmf":
        compare_files_xdmf(first_file, second_file, user_tolerance)
    elif file_type == "h5":
        compare_files_hdf5(first_file, second_file, user_tolerance)
    else:
        raise TypeError("File type not supported")
