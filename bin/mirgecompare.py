#!/usr/bin/env python


def compare_files_vtu(first_file, second_file, file_type, tolerance = 1e-12):
    import vtk

    # read files:
    if file_type == "vtu":
        reader1 = vtk.vtkXMLUnstructuredGridReader()
        reader2 = vtk.vtkXMLUnstructuredGridReader()
    else:
        reader1 = vtk.vtkXMLPUnstructuredGridReader()
        reader2 = vtk.vtkXMLPUnstructuredGridReader()

    reader1.SetFileName(first_file)
    reader1.Update()
    output1 = reader1.GetOutput()

    reader2.SetFileName(second_file)
    reader2.Update()
    output2 = reader2.GetOutput()

    # check fidelity
    point_data1 = output1.GetPointData()
    point_data2 = output2.GetPointData()

    # verify same number of PointData arrays in both files
    if point_data1.GetNumberOfArrays() != point_data2.GetNumberOfArrays():
        print("File 1:", point_data1.GetNumberOfArrays(), "\n", "File 2:", point_data2.GetNumberOfArrays())
        raise ValueError("Fidelity test failed: Mismatched data array count")

    for i in range(point_data1.GetNumberOfArrays()):
        arr1 = point_data1.GetArray(i)
        arr2 = point_data2.GetArray(i)

        # verify both files contain same arrays
        if point_data1.GetArrayName(i) != point_data2.GetArrayName(i):
            print("File 1:", point_data1.GetArrayName(i), "\n", "File 2:", point_data2.GetArrayName(i))
            raise ValueError("Fidelity test failed: Mismatched data array names")

        # verify arrays are same sizes in both files
        if arr1.GetSize() != arr2.GetSize():
            print("File 1, DataArray", i, ":", arr1.GetSize(), "\n", "File 2, DataArray", i, ":", arr2.GetSize())
            raise ValueError("Fidelity test failed: Mismatched data array sizes")

        # verify individual values w/in given tolerance
        for j in range(arr1.GetSize()):
            if abs(arr1.GetValue(j) - arr2.GetValue(j)) > tolerance: 
                print("Tolerance:", tolerance)
                raise ValueError("Fidelity test failed: Mismatched data array values with given tolerance")

    print("VTU Fidelity test completed successfully with tolerance", tolerance)

class Hdf5Reader():
    def __init__(self, filename):
        import h5py

        self.file_obj = h5py.File(filename, 'r')
    
    def read_specific_data(self, datapath):
        return self.file_obj[datapath]

class XdmfReader():
    # CURRENTLY DOES NOT SUPPORT MULTIPLE Grids

    def __init__(self, filename): 
        import xml.etree.ElementTree as ET

        tree = ET.parse(filename)
        root = tree.getroot()

        domains = tuple(root)
        self.domain = domains[0]
        self.grids = tuple(self.domain)
        self.uniform_grid = self.grids[0]

    def get_topology(self):
        connectivity = None

        for a in self.uniform_grid:
            if a.tag == "Topology":
                connectivity = a

        if connectivity == None:
            raise ValueError("File is missing grid connectivity data") 

        return connectivity

    def get_geometry(self):
        geometry = None

        for a in self.uniform_grid:
            if a.tag == "Geometry":
                geometry = a
        
        if geometry == None:
            raise ValueError("File is missing grid node location data") 

        return geometry
            
    def read_data_item(self, data_item):
        # CURRENTLY DOES NOT SUPPORT 'DataItem' THAT STORES VALUES DIRECTLY

        # check that data stored as separate hdf5 file
        if data_item.get("Format") != "HDF":
            raise TypeError("Data stored in unrecognized format")
        
        # get corresponding hdf5 file
        source_info = data_item.text
        split_source_info = source_info.partition(":")

        h5_filename = split_source_info[0]
        # TODO: change file name to match actual mirgecom output directory later
        h5_filename = "examples/" + h5_filename
        h5_datapath = split_source_info[2]

        # read data from corresponding hdf5 file
        h5_reader = Hdf5Reader(h5_filename)
        return h5_reader.read_specific_data(h5_datapath)

import numpy as np

def compare_files_xdmf(first_file, second_file, tolerance = 1e-12):
    # read files
    file_reader1 = XdmfReader(first_file)
    file_reader2 = XdmfReader(second_file)

    # check same number of grids
    if len(file_reader1.grids) != len(file_reader2.grids):
        print("File 1:", len(file_reader1.grids), "\n", "File 2:", len(file_reader2.grids))
        raise ValueError("Fidelity test failed: Mismatched grid count")
    
    # check same number of cells in grid
    if len(file_reader1.uniform_grid) != len(file_reader2.uniform_grid):
        print("File 1:", len(file_reader1.uniform_grid), "\n", "File 2:", len(file_reader2.uniform_grid))
        raise ValueError("Fidelity test failed: Mismatched cell count in uniform grid")

    # compare Topology: 
    top1 = file_reader1.get_topology()
    top2 = file_reader2.get_topology()

    # check TopologyType
    if top1.get("TopologyType") != top2.get("TopologyType"):
        print("File 1:", top1.get("TopologyType"), "\n", "File 2:", top2.get("TopologyType"))
        raise ValueError("Fidelity test failed: Mismatched topology type")
    
    # check number of connectivity values
    connectivities1 = file_reader1.read_data_item(tuple(top1)[0])
    connectivities2 = file_reader2.read_data_item(tuple(top2)[0])

    connectivities1 = np.array(connectivities1)
    connectivities2 = np.array(connectivities2)

    if connectivities1.shape != connectivities2.shape:
        print("File 1:", connectivities1.shape, "\n", "File 2:", connectivities2.shape)
        raise ValueError("Fidelity test failed: Mismatched connectivities count")
    
    if not np.allclose(connectivities1, connectivities2, atol = tolerance):
        print("Tolerance:", tolerance)
        raise ValueError("Fidelity test failed: Mismatched connectivity values with given tolerance")

    # compare Geometry:
    geo1 = file_reader1.get_geometry()
    geo2 = file_reader2.get_geometry()

    # check GeometryType
    if geo1.get("GeometryType") != geo2.get("GeometryType"):
        print("File 1:", geo1.get("GeometryType"), "\n", "File 2:", geo2.get("GeometryType"))
        raise ValueError("Fidelity test failed: Mismatched geometry type")

    # check number of node values
    nodes1 = file_reader1.read_data_item(tuple(geo1)[0])
    nodes2 = file_reader2.read_data_item(tuple(geo2)[0])

    nodes1 = np.array(nodes1)
    nodes2 = np.array(nodes2)

    if nodes1.shape != nodes2.shape:
        print("File 1:", nodes1.shape, "\n", "File 2:", nodes2.shape)
        raise ValueError("Fidelity test failed: Mismatched nodes count")
    
    if not np.allclose(nodes1, nodes2, atol = tolerance):
        print("Tolerance:", tolerance)
        raise ValueError("Fidelity test failed: Mismatched node values with given tolerance")

    # compare other Attributes:
    for i in range(len(file_reader1.uniform_grid)):
        curr_cell1 = file_reader1.uniform_grid[i]
        curr_cell2 = file_reader2.uniform_grid[i]

        # skip already checked cells
        if curr_cell1.tag == "Geometry" or curr_cell1.tag == "Topology":
            continue

        # check AttributeType
        if curr_cell1.get("AttributeType") != curr_cell2.get("AttributeType"):
            print("File 1:", curr_cell1.get("AttributeType"), "\n", "File 2:", curr_cell2.get("AttributeType"))
            raise ValueError("Fidelity test failed: Mismatched cell type")

        # check Attribtue name
        if curr_cell1.get("Name") != curr_cell2.get("Name"):
            print("File 1:", curr_cell1.get("Name"), "\n", "File 2:", curr_cell2.get("Name"))
            raise ValueError("Fidelity test failed: Mismatched cell name")

        # check number of Attribute values
        values1 = file_reader1.read_data_item(tuple(curr_cell1)[0])
        values2 = file_reader2.read_data_item(tuple(curr_cell2)[0])

        if len(values1) != len(values2):
            print("File 1,", curr_cell1.get("Name"), ":", len(values1), "\n", "File 2,", curr_cell2.get("Name"), ":", len(values2))
            raise ValueError("Fidelity test failed: Mismatched data values count")

        # check values w/in tolerance
        for i in range(len(values1)):
            if abs(values1[i] - values2[i]) > tolerance:
                print("Tolerance:", tolerance, "\n", "Cell:", curr_cell1.get("Name"))
                raise ValueError("Fidelity test failed: Mismatched data values with given tolerance")

    print("XDMF Fidelity test completed successfully with tolerance", tolerance)

def compare_files_hdf5(first_file, second_file, tolerance = 1e-12):
    file_reader1 = Hdf5Reader(first_file)
    file_reader2 = Hdf5Reader(second_file)

    f1 = file_reader1.file_obj
    f2 = file_reader2.file_obj

    objects1 = list(f1.keys()) 
    objects2 = list(f2.keys())

    # check number of Grids
    if len(objects1) != len(objects2):
        print("File 1:", len(objects1), "\n", "File 2:", len(objects2))
        raise ValueError("Fidelity test failed: Mismatched grid count")

    # loop through Grids
    for i in range(len(objects1)):
        obj_name1 = objects1[i] 
        obj_name2 = objects2[i]

        if obj_name1 != obj_name2:
            print("File 1:", obj_name1, "\n", "File 2:", obj_name2)
            raise ValueError("Fidelity test failed: Mismatched grid names")

        curr_o1 = list(f1[obj_name1]) 
        curr_o2 = list(f2[obj_name2])

        if len(curr_o1) != len(curr_o2):
            print("File 1,", obj_name1, ":", len(curr_o1), "\n", "File 2,", obj_name2, ":", len(curr_o2))
            raise ValueError("Fidelity test failed: Mismatched group count")

        # loop through Groups
        for j in range(len(curr_o1)):
            subobj_name1 = curr_o1[j] 
            subobj_name2 = curr_o2[j]

            if subobj_name1 != subobj_name2:
                print("File 1:", subobj_name1, "\n", "File 2:", subobj_name2)
                raise ValueError("Fidelity test failed: Mismatched group names")

            subpath1 = obj_name1 + "/" + subobj_name1
            subpath2 = obj_name2 + "/" + subobj_name2

            data_arrays_list1 = list(f1[subpath1])
            data_arrays_list2 = list(f2[subpath2])

            if len(data_arrays_list1) != len(data_arrays_list2):
                print("File 1,", subobj_name1, ":", len(data_arrays_list1), "\n", "File 2,", subobj_name2, ":", len(data_arrays_list2))
                raise ValueError("Fidelity test failed: Mismatched data list count")

            # loop through data arrays
            for k in range(len(data_arrays_list1)):
                curr_listname1 = data_arrays_list1[k] 
                curr_listname2 = data_arrays_list2[k]

                if curr_listname1 != curr_listname2:
                    print("File 1:", curr_listname1, "\n", "File 2:", curr_listname2)
                    raise ValueError("Fidelity test failed: Mismatched data list names")

                curr_listname1 = subpath1 + "/" + curr_listname1
                curr_listname2 = subpath2 + "/" + curr_listname2

                curr_datalist1 = np.array(list(f1[curr_listname1]))
                curr_datalist2 = np.array(list(f2[curr_listname2]))

                if curr_datalist1.shape != curr_datalist2.shape:
                    print("File 1,", curr_listname1, ":", curr_datalist1.shape, "\n", 
                          "File 2,", curr_listname2, ":", curr_datalist2.shape)
                    raise ValueError("Fidelity test failed: Mismatched data list size")
                
                if not np.allclose(curr_datalist1, curr_datalist2, atol = tolerance):
                    print("Tolerance:", tolerance, "\n", "Data List:", curr_listname1)
                    raise ValueError("Fidelity test failed: Mismatched data values with given tolerance")

    print("HDF5 Fidelity test completed successfully with tolerance", tolerance)

# run fidelity check
if __name__ == "__main__":
    import argparse
    import os

    # read in file and comparison info from command line
    parser = argparse.ArgumentParser(description = 'Process files to perform fidelity check')
    parser.add_argument('files', nargs = 2, type = str)
    parser.add_argument('--tolerance', type = float)
    args = parser.parse_args();

    first_file = args.files[0]  
    second_file = args.files[1] 

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
