"""Utility functions for loading and processing VTI microstructure data."""

import numpy as np
import pyvista as pv


def load_vti(filepath: str) -> dict:
    """
    Load a VTI file and return the data as a numpy array with metadata.
    
    Parameters
    ----------
    filepath : str
        Path to the VTI file.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'grain_ids': 3D numpy array of grain IDs with shape (nx, ny, nz)
        - 'rgb': 4D numpy array of RGB colors with shape (nx, ny, nz, 3)
        - 'gray': 3D numpy array of grayscale values with shape (nx, ny, nz)
        - 'dimensions': tuple of (nx, ny, nz)
        - 'spacing': tuple of voxel spacing
        - 'origin': tuple of origin coordinates
    """
    mesh = pv.read(filepath)
    
    # Get the dimensions from the mesh
    # limit to 101 x 101 x 101
    dims = mesh.dimensions  # (nx, ny, nz)
    
    # Get the grain ID data and reshape to 3D
    # VTI data is stored in Fortran order (x varies fastest)
    grain_ids = mesh['Grain ID']
    grain_ids_3d = grain_ids.reshape(*dims, order='F')
    
    # Get the RGB data and reshape to 4D (nx, ny, nz, 3)
    rgb = mesh['rgbZ']  # shape: (n_points, 3), values in [0, 1]
    rgb_3d = rgb.reshape((*dims, 3), order='F')
    
    # Convert RGB to grayscale using luminosity method
    # gray = 0.299*R + 0.587*G + 0.114*B
    gray_3d = np.dot(rgb_3d, [0.299, 0.587, 0.114])[..., None]
    
    return {
        'grain_ids': grain_ids_3d,
        'rgb': rgb_3d,
        'gray': gray_3d,
        'dimensions': dims,
        'spacing': mesh.spacing,
        'origin': mesh.origin,
    }


def get_xy_slice(data: np.ndarray, z_index: int) -> np.ndarray:
    """
    Extract an XY slice from 3D volumetric data at the given Z index.
    
    Parameters
    ----------
    data : np.ndarray
        3D numpy array with shape (nx, ny, nz).
    z_index : int
        Index along the Z axis to extract the slice from.
        
    Returns
    -------
    np.ndarray
        2D numpy array of shape (nx, ny) representing the XY slice.
        
    Raises
    ------
    IndexError
        If z_index is out of bounds for the data.
    """
    nz = data.shape[2]
    if z_index < 0 or z_index >= nz:
        raise IndexError(f"z_index {z_index} out of bounds for data with {nz} Z slices")
    
    return data[:64, :64, z_index]

