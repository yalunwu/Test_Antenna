"""
openEMS and CSXCAD Environment Configuration
Handles dynamic loading of Windows DLLs, path configuration, and helper utilities.
"""
import os
import sys
import pathlib
import warnings

# Suppress known benign HDF5 runtime version warning (HDF5 2.2.0 vs 2.0.0)
warnings.filterwarnings("ignore", category=UserWarning, module="h5py")

# Determine openEMS installation directory
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
_OPENEMS_DIR = _PROJECT_ROOT / "openEMS"

if not _OPENEMS_DIR.exists():
    # Fallback to local user path
    _OPENEMS_DIR = pathlib.Path(r"c:\Users\Allen\Documents\Test_Antenna\openEMS")

# Configure environment variables required by openEMS and CSXCAD
os.environ["CSXCAD_INSTALL_PATH"] = str(_OPENEMS_DIR)
os.environ["OPENEMS_INSTALL_PATH"] = str(_OPENEMS_DIR)

# Add openEMS folder to DLL search path on Windows
if hasattr(os, "add_dll_directory") and _OPENEMS_DIR.exists():
    try:
        os.add_dll_directory(str(_OPENEMS_DIR))
    except Exception:
        pass

# Also prepend to system PATH for any sub-processes
os.environ["PATH"] = str(_OPENEMS_DIR) + os.pathsep + os.environ.get("PATH", "")

def get_openems_path() -> pathlib.Path:
    """Return the resolved openEMS directory path."""
    return _OPENEMS_DIR

def view_in_appcsxcad(xml_path: str or pathlib.Path) -> None:
    """Launch AppCSXCAD to visually inspect the 3D antenna model and mesh."""
    from CSXCAD import AppCSXCAD_BIN
    xml_str = str(xml_path)
    if os.path.exists(xml_str):
        print(f"Launching AppCSXCAD with {xml_str}...")
        os.system(f'"{AppCSXCAD_BIN}" "{xml_str}"')
    else:
        print(f"Error: XML file {xml_str} not found.")
