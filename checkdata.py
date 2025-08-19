import scipy.io as sio
import numpy as np

file_path = r"C:\Users\Sneha\Desktop\Sneha\ev_soc_project\4. BatteryAgingARC_45_46_47_48\B0045.mat"
mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

def explore(obj, indent=0):
    prefix = "    " * indent
    print(f"{prefix}- Type: {type(obj)}, Shape: {getattr(obj, 'shape', 'N/A')}")
    
    if hasattr(obj, "_fieldnames"):  # MATLAB struct
        print(f"{prefix}  Fields: {obj._fieldnames}")
        for field in obj._fieldnames:
            explore(getattr(obj, field), indent + 1)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:  # Object array
        print(f"{prefix}  Object array with {obj.size} elements")
        for i, item in enumerate(obj):
            print(f"{prefix}  [{i}]")
            explore(item, indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}  List/Tuple with {len(obj)} elements")
        for i, item in enumerate(obj):
            print(f"{prefix}  [{i}]")
            explore(item, indent + 1)

B0045 = mat_data['B0045']
explore(B0045.cycle)


