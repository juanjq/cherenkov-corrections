# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob, tables
import pandas as pd

# Importing custom utility functions
sys.path.insert(0, os.getcwd() + "/../scripts/")
import auxiliar as aux

def get_and_write_intensities(im, iM, fname, lengroups=100):
    im, iM = int(im), int(iM)
    
    str_dchecks = "datacheck_"
    dl1_root = "/fefs/aswg/data/real/DL1/*/v0.*/tailcut84/"
    main_name = f"{str_dchecks}dl1_LST-1.Run?????"
    
    total_dl1a_runwise = np.sort(glob.glob(dl1_root + "*/" + f"{main_name}.h5") + glob.glob(dl1_root + f"{main_name}.h5"))

    counts, intensity_points = [], []
    tab = tables.open_file(total_dl1a_runwise[0])
    _bin_edges = tab.root.dl1datacheck.histogram_binning.col("hist_intensity")[0]
    tab.close()
    binning = list((_bin_edges[1:] + _bin_edges[:-1]) / 2)
    
    for i, srun_dcheck in enumerate(total_dl1a_runwise[im:iM]):
        print(f"Analysing... {i:3}/{lengroups}") if i % 10 == 0 else None
        
        tab = tables.open_file(srun_dcheck)
        for inte, time  in zip(tab.root.dl1datacheck.cosmics.col("hist_intensity"),
                               tab.root.dl1datacheck.cosmics.col("elapsed_time")):
            
            counts += list(np.array(inte) / time)
            intensity_points += binning
    
        tab.close()
    
    print("Writting...")
    with open(fname, "wb") as f:
        pickle.dump([intensity_points, counts], f, pickle.HIGHEST_PROTOCOL)
    print("Finished\n")

def write_ws_run_relation(init, end, fname_dcheck_raw, file_ws_db, fname_ws_run_relation, fname_ws_reduced):
    init, end = int(init), int(end)
    
    # Reading the datacheck dictionary
    with open(fname_dcheck_raw, 'rb') as f:
        dict_dcheck_raw = pickle.load(f) 

    # WS reading
    with open(fname_ws_reduced, "rb") as f:
        df_ws, dates_ws = pickle.load(f)
    maxdate_ws = np.max(dates_ws)

    print("Starting to iterate")
    ws_entry, run, srun = [], [], []
    for j, date_dcheck in enumerate(dict_dcheck_raw["time"]):
        if j >= init and j <= end and j:

            if date_dcheck > maxdate_ws:
                run.append(dict_dcheck_raw["run"][j])
                srun.append(dict_dcheck_raw["srun"][j])
                ws_entry.append(np.nan)

            else:
                str_id = str(df_ws.iloc[np.argmin(np.abs(dates_ws - date_dcheck))].name)

                run.append(dict_dcheck_raw["run"][j])
                srun.append(dict_dcheck_raw["srun"][j])
                ws_entry.append(str_id)

    print("Writting...")
    with open(fname_ws_run_relation, "a") as f:
        for r, s, e in zip(run, srun, ws_entry):
            f.write(f"{r}-{s},{e}\n")
    print("Finished\n")
                    
if __name__ == "__main__":
    
    func_name = sys.argv[1]
    func_args = sys.argv[2:]  # Take all arguments after the function name
    
    globals()[func_name](*func_args)  # Unpack the arguments and pass them to the function