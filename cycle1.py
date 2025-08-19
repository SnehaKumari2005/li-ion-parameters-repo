import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat = sio.loadmat(r"C:/Users/Sneha/Desktop/Sneha/ev_soc_project/4. BatteryAgingARC_45_46_47_48/B0045.mat")
cycles = mat["B0045"][0, 0]["cycle"][0]
print("Total cycles in dataset:", len(cycles))

capacity = 2.0  

n_points = 1000  
all_voltage, all_current, all_soc = [], [], []
time_ref = None


for idx in range(0, len(cycles), 2):
    cycle_num = idx + 1  
    cycle = cycles[idx]

    
    if "data" not in cycle.dtype.names:
        print(f"Cycle {cycle_num}: MISSING 'data' field")
        continue

    data = cycle["data"][0, 0]
    fields = data.dtype.names

    required = ["Time", "Voltage_measured", "Current_measured"]
    if not all(field in fields for field in required):
        print(f"Cycle {cycle_num}: Missing fields → {[f for f in required if f not in fields]} ")
        continue

    
    time = np.array(data["Time"]).flatten()
    voltage = np.array(data["Voltage_measured"]).flatten()
    current = np.array(data["Current_measured"]).flatten()

    if len(time) == 0 or len(voltage) == 0 or len(current) == 0:
        print(f"Cycle {cycle_num}: Empty data arrays")
        continue

    
    dt = np.diff(time, prepend=time[0])
    Ah_flow = np.cumsum(-current * dt / 3600.0)  # Ah
    soc = 1 - (Ah_flow / capacity)

    
    if time_ref is None:
        time_ref = np.linspace(time.min(), time.max(), n_points)
    voltage_i = np.interp(time_ref, time, voltage)
    current_i = np.interp(time_ref, time, current)
    soc_i = np.interp(time_ref, time, soc)

    
    all_voltage.append(voltage_i)
    all_current.append(current_i)
    all_soc.append(soc_i)

    print(f"Cycle {cycle_num}: Processed")


if all_voltage:
    avg_voltage = np.mean(all_voltage, axis=0)
    avg_current = np.mean(all_current, axis=0)
    avg_soc = np.mean(all_soc, axis=0)

   
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_ref, avg_voltage, label="Average Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_ref, avg_current, label="Average Current", color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_ref, avg_soc, label="Average SoC", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("State of Charge (SoC)")
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No valid odd cycles found for averaging")
