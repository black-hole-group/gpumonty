/*
 * GPUmonty - hdf5_utils.h
 * Copyright (C) 2026 Pedro Naethe Motta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */
/*
 * hdf5_utils.h
 */
#ifndef HDF5_UTILS_H
#define HDF5_UTILS_H

#pragma once

#include <hdf5.h>

// Force MPI on or off
#define USE_MPI 0

// Blob "copy" utility
#ifndef HDF5_BLOB
#define HDF5_BLOB
typedef hid_t hdf5_blob;
#endif

/**
 * @brief Extracts an HDF5 object into an in-memory "blob."
 * This function uses the HDF5 "CORE" Virtual File Driver (VFD) to create a virtual 
 * file in RAM. It then copies an object (dataset or group) from the physical 
 * file into this virtual file.
 *
 * @param name The name of the object to extract.
 * @return hdf5_blob A handle (hid_t) to the new in-memory HDF5 file image.
 */
hdf5_blob hdf5_get_blob(const char *name);

/**
 * @brief Writes an in-memory HDF5 blob back to the main file.
 * This function saves the data stored in a temporary RAM-resident HDF5 image 
 * into the permanent file structure. It is often used to save processed 
 * results or updated datasets that were modified in memory for speed.
 *
 * @param blob The handle (hid_t) to the in-memory HDF5 file image.
 * @param name The target name for the object within the main file.
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_write_blob(hdf5_blob blob, const char *name);

/**
 * @brief Deallocates the in-memory HDF5 blob and releases system RAM.
 * 
 * Because the blob was created using the H5FD_CORE driver, the "file" 
 * exists entirely in a memory buffer. Calling this function tells the 
 * HDF5 library that the buffer is no longer needed, effectively 
 * performing a 'free()' operation on that block of memory.
 *
 * @param blob The handle (hid_t) to the in-memory HDF5 file image.
 * @return 0 on success, or a negative error code on failure.
 * 
 */
int hdf5_close_blob(hdf5_blob blob);

// File
/**
 * @brief Creates a new HDF5 file on disk, overwriting any existing file with the same name.
 * This function sets up the environment for writing simulation data. It includes 
 * logic for both single-processor and parallel (MPI) environments.
 *
 * @param fname The string containing the path/name of the file to create (e.g., "data.h5").
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_create(const char *fname);

/**
 * @brief Opens an existing HDF5 file for reading.
 * Unlike ``hdf5_create``, this function expects the file to already exist on disk.
 * It uses the ``Read-Only`` flag to protect the data. Like the creation function,
 * it supports parallel access if MPI is enabled.
 *
 * @param fname The path/name of the existing file to open.
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_open(const char *fname);

/**
 * @brief Flushes pending data to disk and closes the physical HDF5 file.
 * 
 * This is a function for data integrity. It forces the operating 
 * system to write any remaining "cached" data to the hard drive.
 *
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_close();

// Directory
/**
 * @brief Creates a new HDF5 Group at the current path location.
 * This function builds a hierarchical structure within the file. It does 
 * not "move" into the new directory; it simply creates it and then 
 * releases the handle.
 *
 * @param name The name of the new group to create.
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_make_directory(const char *name);

/**
 * @brief Updates the internal "Current Working Directory" tracker.
 * This function acts like the 'cd' (change directory) command in a terminal.
 * It changes the value of the global string hdf5_cur_dir, which is then 
 * prepended to object names in functions like ```hdf5_make_directory```, 
 * ```hdf5_get_blob```, and ```hdf5_write_blob```.
 *
 * @param path The full internal path to set as the current location.
 */
void hdf5_set_directory(const char *path);

// Write
/**
 * @brief Writes a single scalar value as a standalone HDF5 dataset.
 * 
 * This function is used for global simulation parameters that aren't 
 * associated with a specific grid.
 *
 * @param val       Pointer to the variable holding the value.
 * @param name      The name of the dataset to create.
 * @param hdf5_type The HDF5 datatype.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_write_single_val(const void *val, const char *name, hsize_t hdf5_type);


/**
 * @brief Writes a multidimensional "hyperslab" from memory to a dataset in the file.
 * This function can take a 
 * sub-section of an array in RAM and place it into a specific location 
 * within a larger global dataset in the HDF5 file.
 *
 * @param data      Pointer to the source data in RAM.
 * @param rank      Number of dimensions (e.g., 3 for a 3D grid).
 * @param fdims     Total size of the dataset as it will appear in the file.
 * @param fstart    The starting coordinate (offset) in the file dataset.
 * @param fcount    The dimensions of the block being written.
 * @param mdims     Total size of the array as it currently exists in memory.
 * @param mstart    The starting coordinate (offset) in the memory array.
 * @param hdf5_type The datatype.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_write_array(const void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);

// Read
/**
 * @brief Checks for the existence of a link (object) within the HDF5 file.
 * This function performs a path-based look-up. It determines if the internal 
 * 'address' constructed from the current directory and the object name points 
 * to something valid.
 * @param name The name of the object to look for (e.g., "density").
 * @return 1 if the object exists, 0 otherwise.
 */
int hdf5_exists(const char *name);

/**
 * @brief Reads a single scalar value from an HDF5 dataset into memory.
 *
 * @param val       Pointer to the destination variable where the value will be stored.
 * @param name      The name of the dataset to read from.
 * @param hdf5_type The expected HDF5 datatype.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_read_single_val(void *val, const char *name, hsize_t hdf5_type);


/**
 * @brief Reads a multidimensional sub-section (hyperslab) from a file into memory.
 * 
 * This function maps a specific "box" of data inside the HDF5 file to a 
 * specific "box" of an array in RAM.
 *
 * @param data      Pointer to the destination buffer in RAM.
 * @param rank      Dimensionality of the data (e.g., 3 for a 3D grid).
 * @param fdims     The total dimensions of the dataset inside the file.
 * @param fstart    The starting coordinate (offset) in the file.
 * @param fcount    The dimensions of the block to be read.
 * @param mdims     The total dimensions of the destination array in memory.
 * @param mstart    The starting coordinate (offset) in memory.
 * @param hdf5_type The HDF5 datatype.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_read_array(void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);



// Convenience and annotations
/**
 * @brief Creates a fixed-length C-style string datatype for HDF5.
 * This function builds a specific template that tells the HDF5 library to expect a C-style string that is exactly ```len``` bytes long.
 *
 * @param len The maximum number of characters (including the null terminator \0).
 * @return hid_t A handle to the newly created string datatype.
 */
hid_t hdf5_make_str_type(size_t len);

/**
 * @brief Writes a 1D array of fixed-length strings to the HDF5 file.
 * 
 * This function creates a dataset where each entry is a string of a specific 
 * size. It is used to make the data self-documenting by providing names for 
 * primitive variables or coordinate indices.
 *
 * @param data    Pointer to the string array.
 * @param name    The name of the dataset to create.
 * @param str_len The fixed length of each string buffer.
 * @param len     The number of strings in the list.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_write_str_list(const void *data, const char *name, size_t strlen, size_t len);

/**
 * @brief Attaches a scalar attribute (metadata) to an HDF5 object.
 *
 * This function creates a "label" on a specific object. The attribute is 
 * created as a scalar, meaning it holds a single value (e.g., one number 
 * or one fixed-length string) rather than an array.
 *
 * @param att       Pointer to the data value to be stored.
 * @param att_name  The name of the attribute.
 * @param data_name The name of the dataset/group to attach it to.
 * @param hdf5_type The HDF5 datatype of the attribute.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_add_attr(const void *att, const char *att_name, const char *data_name, hsize_t hdf5_type);

/**
 * @brief Attaches a physical unit label to an HDF5 object.
 * This function ensures that your data is self-describing. By labeling 
 * datasets with their units, the output file is much easier to 
 * analyze with external tools.
 *
 * @param name The name of the dataset or group to label.
 * @param unit The string representing the unit.
 * 
 * @return 0 on success, or a negative error code on failure.
 */
int hdf5_add_units(const char *name, const char *unit);

#endif // HDF5_UTILS_H
