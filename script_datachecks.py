import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob, tables, pymongo
import pandas as pd
from pathlib import Path
import lst1_mongodb_tcu

def get_intensity_cut(fname, run=None, cuts_dir=None):
    from lstchain.reco.utils import get_intensity_cut as lst_get_intensity_cut

    base_name = os.path.basename(fname)
    run_label = f"Run {run}" if run is not None else "Unknown Run"
    
    print(f"--- [{run_label}] Processing started ---", flush=True)
    print(f"[{run_label}] File path: {fname}", flush=True)

    try:
        print(f"[{run_label}] Attempting to read 'intensity' column from HDF5...", flush=True)
        dl2_df = pd.read_hdf(fname, key="/dl2/event/telescope/parameters/LST_LSTCam", columns=["intensity"])
        print(f"[{run_label}] Successfully loaded {len(dl2_df):,} events into data frame.", flush=True)
        
        print(f"[{run_label}] Computing intensity cut value via lstchain...", flush=True)
        cut_value = lst_get_intensity_cut(dl2_df)
        int_cut = cut_value
        print(f"[{run_label}] Success! Calculated intensity cut: {int_cut}", flush=True)
        
    except KeyError:
        print(f"[{run_label}] [SKIPPED] Critical key missing in HDF5 file.", flush=True)
        int_cut = np.nan
    except Exception as e:
        print(f"[{run_label}] [ERROR] Process failed with exception: {e}", flush=True)
        int_cut = np.nan

    # Save to text file if running in batch mode
    if run is not None and cuts_dir is not None:
        save_path = Path(cuts_dir, f"{run}.txt")
        print(f"[{run_label}] Exporting result to text file: {save_path}", flush=True)
        save_path.write_text(str(int_cut))
        print(f"[{run_label}] Text file saved successfully.", flush=True)
        
    print(f"--- [{run_label}] Processing finished ---\n", flush=True)
    return int_cut

def query_psf_batch(tstart_iso, tstop_iso, var_name, out_file):
    print(f"--- [Query] Starting {var_name} from {tstart_iso} to {tstop_iso} ---", flush=True)

    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)  # <-- fixes Errno 2

        tstart = datetime.fromisoformat(tstart_iso)
        tstop  = datetime.fromisoformat(tstop_iso)
        print(f"[Query] Connecting to MongoDB at lst101-int:27017...", flush=True)
        client = pymongo.MongoClient("lst101-int:27017")

        print(f"[Query] Fetching data for {var_name}...", flush=True)
        out = lst1_mongodb_tcu.get_entries(client, var_name, astropy_time=False, tstart=tstart, tstop=tstop)
        client.close()

        dates, values = out["time"], out["value"]

        print(f"[Query] Writing {len(values)} entries to {out_file}...", flush=True)
        with open(out_file, "w") as f:
            for t, v in zip(dates, values):
                f.write(f"{pd.Timestamp(t).isoformat()},{v}\n")
        print(f"[Query] Successfully wrote data to {os.path.basename(out_file)}", flush=True)

        print("Finished", flush=True)

    except Exception as e:
        print(f"[Query] [ERROR] Failed to query or write data: {e}", flush=True)
        sys.exit(1)   # <-- non-zero exit: inline mode won't touch .done,
                       #     and this line never prints "Finished" anyway


# def get_and_write_intensities(im, iM, fname, lengroups=100):
#     im, iM = int(im), int(iM)
#     str_dchecks = "datacheck_"
#     dl1_root = "/fefs/aswg/data/real/DL1/*/v0.*/tailcut84/"
#     main_name = f"{str_dchecks}dl1_LST-1.Run?????"

#     total_dl1a_runwise = np.sort(glob.glob(dl1_root + "*/" + f"{main_name}.h5") + glob.glob(dl1_root + f"{main_name}.h5"))

#     counts, intensity_points = [], []
#     tab = tables.open_file(total_dl1a_runwise[0])
#     _bin_edges = tab.root.dl1datacheck.histogram_binning.col("hist_intensity")[0]
#     tab.close()

#     binning = list((_bin_edges[1:] + _bin_edges[:-1]) / 2)

#     for i, srun_dcheck in enumerate(total_dl1a_runwise[im:iM]):
#         if i % 10 == 0:
#             print(f"Analysing... {i:3}/{lengroups}")
#         tab = tables.open_file(srun_dcheck)
#         for inte, time  in zip(tab.root.dl1datacheck.cosmics.col("hist_intensity"),
#                                tab.root.dl1datacheck.cosmics.col("elapsed_time")):
#             counts += list(np.array(inte) / time)
#             intensity_points += binning
#         tab.close()

#     print("Writing...")
#     with open(fname, "wb") as f:
#         pickle.dump([intensity_points, counts], f, pickle.HIGHEST_PROTOCOL)
#     print("Finished\n")


def write_ws_run_relation(init, end, fname_dcheck_flat, fname_ws_reduced, fname_ws_run_relation):
    """
    Matches datacheck subruns to the nearest Weather Station (WS) entries.
    """
    init, end = int(init), int(end)

    # 1. Read datacheck flat table (now Parquet, not Pickle)
    # We only load the columns we need to save memory in the SLURM worker
    df_dcheck = pd.read_parquet(
        fname_dcheck_flat, 
        columns=["obs_id", "subrun", "time"]
    )
    
    # Slice the dataframe for the specific SLURM batch
    df_batch = df_dcheck.iloc[init : end + 1]

    # 2. Read WS reduced data
    with open(fname_ws_reduced, "rb") as f:
        df_ws, dates_ws = pickle.load(f)

    maxdate_ws = np.max(dates_ws)

    print(f"Iterating over batch [{init}:{end}] ({len(df_batch)} rows)...")
    ws_entry, run, srun = [], [], []

    # 3. Match WS times
    for row in df_batch.itertuples(index=False):
        date_dcheck = row.time
        
        run.append(row.obs_id)
        srun.append(row.subrun)

        if date_dcheck > maxdate_ws:
            ws_entry.append(np.nan)
        else:
            # FIX: Convert Pandas Timestamp to numpy.datetime64[ns]
            target_dt = np.datetime64(date_dcheck)
            
            # Now NumPy can handle the array-wide subtraction flawlessly
            idx = np.argmin(np.abs(dates_ws - target_dt))
            str_id = str(df_ws.index[idx])
            ws_entry.append(str_id)

    print("Writing...")
    with open(fname_ws_run_relation, "a") as f:
        for r, s, e in zip(run, srun, ws_entry):
            f.write(f"{r}-{s},{e}\n")
    print("Finished\n")


if __name__ == "__main__":
    func_name = sys.argv[1]
    func_args = sys.argv[2:]
    globals()[func_name](*func_args)