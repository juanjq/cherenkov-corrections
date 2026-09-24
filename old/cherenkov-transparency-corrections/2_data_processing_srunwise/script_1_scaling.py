import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob, tables, copy, subprocess
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

from astropy.coordinates import SkyCoord
from lstchain.io.config  import get_standard_config
from ctapipe.io          import read_table

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

import script_0_utils_scaling as utils_scaling

    
def main_init(input_job_str, file_config, simulate_data=False):
    logger.info(f"Input job string: {input_job_str}")
    
    if isinstance(simulate_data, str):
        simulate_data = simulate_data.lower() == "true"
    
    ########################################
    # Initial configuring and paths creation
    # Reading the configuration file
    with open(file_config, "r") as json_file:
        dict_config = json.load(json_file)
    
    # Other auxiliar scripts
    sys.path.insert(0, dict_config["root_scripts"])
    import geometry as geom
    
    # Extracting the run number from the input string
    obs_id = int(input_job_str.split("_")[0])
    first_last_srun = [int(s) for s in input_job_str.split("_")[1:]]
    if len(first_last_srun) == 1:
        srun_numbers = np.array(first_last_srun)
    else:
        srun_numbers = np.arange(first_last_srun[0], first_last_srun[1] + 1)
    
    ########################################################
    # Binning    
    binning_intensity = np.array(dict_config["binning_intensity"])
    binning_intensity_c = (binning_intensity[1:] * binning_intensity[:-1]) ** 0.5 # Center (log) of binning
    binning_intensity_w = np.diff(binning_intensity) # Width of bins    
    # Mask for the fitting region in the fits
    mask_dcheck_bins_fit = (
        (binning_intensity_c >= dict_config["lims_intensity"][0]) &
        (binning_intensity_c <= dict_config["lims_intensity"][1])
    )
      
    ##########################################################
    # Reading the histogram data from the dl1 ---------------
    dict_srunwise = {} # Will contain: "file_dl1", "data_counts_intensity", "data_rates", "data_delta_rates"

    for srun in srun_numbers:
        dict_srunwise[srun] = {}
        
        # First we find the files, and we store them in dict_srunwise
        query_dl1_srun = np.sort(glob.glob(os.path.join(dict_config["root_dl1"], f"dl1_LST-1.Run{obs_id}.{srun:04}.h5")))
        if len(query_dl1_srun) == 0:
            logger.error(f"No dl2 file found for Run {obs_id}"); sys.exit()
        file_dl1 = query_dl1_srun[0]    
        dict_srunwise[srun]["file_dl1"] = file_dl1
        
        # Reading the file
        table_data = tables.open_file(file_dl1)
        data_counts_intensity, _ = np.histogram(
            table_data.root.dl1.event.telescope.parameters.LST_LSTCam.col("intensity"), 
            bins = binning_intensity,
        ); table_data.close()
        
        # Normalize by time and bin size
        effective_time_srun = dict_config["dicts"]["telapsed"][str(obs_id)][str(srun)]
        dict_srunwise[srun]["data_counts_intensity"] = data_counts_intensity
        dict_srunwise[srun]["data_rates"] = (data_counts_intensity/effective_time_srun/binning_intensity_w)
        dict_srunwise[srun]["data_delta_rates"] = (np.sqrt(data_counts_intensity)/effective_time_srun/binning_intensity_w)
        
    # -----------------------------------------------------------
    # Initially filling with the information we know in advance
    # Empty dictionary to store all the results of one run.
    dict_results = copy.deepcopy(utils_scaling.dict_results_empty)
    dict_results["obs_id"] = obs_id
    
    # First filling the dictionary with ones in the scaled values
    # and saving the number of events stored in each subrun.
    for srun in srun_numbers:
        dict_results["scaled"]["original"][srun] = 1.0
        dict_results["statistics"][srun] = int(np.sum(dict_srunwise[srun]["data_counts_intensity"]))
    
    
    ###############################################
    # INITIAL SCALING
    logger.info(f"\n\nPerforming \"init\" step\n{'-' * 20}\n")
    # Then we read these files and perform the fits
    dict_results = utils_scaling.find_scaling(
        iteration_step = "original", 
        dict_results = dict_results,
        input_job_str = input_job_str,
        dict_config = dict_config, 
        simulate_data = simulate_data,
    )    
    
    # Then filling the next step "scaled" with the calculated one
    for srun in srun_numbers:
        dict_results["scaled"]["upper"][srun] = dict_results["scaling"]["original"][srun]
    
    ###############################################
    # UPPER SCALING
    logger.info(f"\n\nPerforming \"upper\" step\n{'-' * 20}\n")
    dict_results = utils_scaling.find_scaling(
        iteration_step = "upper",
        dict_results = dict_results,
        input_job_str = input_job_str,
        dict_config = dict_config,
        simulate_data = simulate_data,
    )
    
    # Calculating the linear factor to the linear scaling
    for srun in srun_numbers:
    
        # Now putting all together, upper and half
        points_scaling     = np.array([1, dict_results["scaling"]["original"][srun]])
        points_light_yield = np.array(
            [dict_results["light_yield"]["original"][srun], dict_results["light_yield"]["upper"][srun]]
        )
    
        # Finding the final scaling as a line that pass trough the two points we have
        # Then we calculate where the light yield will be 1 in linear approximation
        slope     = (points_light_yield[1] - points_light_yield[0]) / (points_scaling[1] - points_scaling[0])
        intercept = points_light_yield[0] - slope * points_scaling[0]
        linear_scale_factor = 1 / slope - points_light_yield[0] / slope + points_scaling[0]
    
        dict_results["scaled"]["linear"][srun] = linear_scale_factor
    
    ###############################################
    # LINEAR SCALING
    logger.info(f"\n\nPerforming \"linear\" step\n{'-' * 20}\n")
    dict_results = utils_scaling.find_scaling(
        iteration_step = "linear",
        dict_results = dict_results,
        input_job_str = input_job_str,
        dict_config = dict_config,
        simulate_data = simulate_data,
    )
    
    # And finally calculating the final scaling factors as 2deg polynomial interpolation
    for srun in srun_numbers:
    
        # Only calculating for the cases with no flag errors:
        if not dict_results["flag_error"][srun]:
    
            # All 3 points we have
            if simulate_data:
                points_scaling           = np.array([1,     1.2,  1.4])  + np.random.rand(3) * 0.1
                points_light_yield       = np.array([0.7,   0.9,  1.2])  + np.random.rand(3) * 0.1
                points_delta_light_yield = np.array([0.05, 0.05, 0.05])  + np.random.rand(3) * 0.01
            else:
                stages = ["original", "linear", "upper"]
                points_scaling = np.array([dict_results["scaled"][key][srun] for key in stages])
                points_light_yield = np.array([dict_results["light_yield"][key][srun] for key in stages])
                points_delta_light_yield = np.array([dict_results["delta_light_yield"][key][srun] for key in stages])
            
            # Parabola parameters
            srun_a, srun_b, srun_c, srun_delta_a, srun_delta_b, srun_delta_c = geom.parabola_3points(
                *points_scaling, *points_light_yield, *points_delta_light_yield
            )
            
            # ToDo Check and add a Likelihood method instead of this rudimentary one
            range_avg_point = np.mean(points_scaling)
            x0, delta_x0 = geom.get_roots_pol2(
                range_avg_point, 1,*points_scaling, *points_light_yield, *points_delta_light_yield
            )
    
            final_scale_factor = x0
            delta_final_scale_factor = delta_x0
    
        else:
            final_scale_factor = np.nan
            delta_final_scale_factor = np.nan
        
        # Plotting for debuging $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        dict_results["final_scaling"][srun] = final_scale_factor
        dict_results["delta_final_scaling"][srun] = delta_final_scale_factor
    
    ##############################
    # Storing data in a pkl object
    dict_fname = os.path.join(dict_config["root_results"], f"results_job_{input_job_str}.pkl")
    
    # Saving the objects
    with open(dict_fname, "wb") as f:
        pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)
        
    logger.info("SUCCESFULLY FINISHED")


def main_merge(file_config):
    
    ########################################
    # Initial configuring and paths creation
    # Reading the configuration file
    with open(file_config, "r") as json_file:
        dict_config = json.load(json_file)
    
    # Other auxiliar scripts
    sys.path.insert(0, dict_config["root_scripts"])
    import geometry as geom
    import utils
    
    #############################################
    #Reading all the information on the directory
    # All the stored dicts that are inside the results folder
    files_results = np.sort(glob.glob(os.path.join(dict_config["root_results"], "*.pkl")))
    
    # Storing all the run numbers and all the separate dicts
    obs_ids_results, dict_results = [], []
    for file in files_results:
    
        # Reading the dicts using pickle
        with open(file, "rb") as f:
            tmp_dict = pickle.load(f)
    
        obs_ids_results.append(int(os.path.basename(file).split("_")[2]))
        dict_results.append(tmp_dict)

    ##############################################
    # Cleaning some directories with temporal data
    # Iterate over all the entries in the directory
    for dir_to_delete in [dict_config["root_sub_dl1"]]:
        for entry in os.listdir(dir_to_delete):
            entry_path = os.path.join(dir_to_delete, entry)
            # Check if it's a file and delete it
            if os.path.isfile(entry_path):
                os.remove(entry_path)
        
    ##########################
    # Merging the dictionaries
    # Keep only non-repeated runs
    obs_ids_results = np.unique(obs_ids_results)
    
    # We create a empty dict for each run we have information
    dict_runs = {}
    for obs_id in obs_ids_results:
        # Empty dictionary to store all the results of one run.
        tmp = copy.deepcopy(utils_scaling.dict_results_empty)
        tmp["obs_id"] = obs_id
        dict_runs[obs_id] = tmp
        
    # Now we fill this dicts one by one with the empty one
    for d in dict_results:
        obs_id = d["obs_id"]
        dict_runs[obs_id] = utils.merge_dicts(dict_runs[obs_id], d)

    ###################################
    # Checking statistics
    # We don"t trust the fit for subruns with too few events or in which the fit have not suceeded
    # In tose cases we will apply as the final scaling the average with the neighbors.
    for obs_id in dict_runs.keys():
        # The dictionary of one run
        dict_run = dict_runs[obs_id]
        # Statistics object (dict)
        dict_stats = dict_run["statistics"]
        last_srun = max(dict_stats.keys())
    
        # We check subrun by subrun
        for srun in np.sort(list(dict_stats.keys())):
            stats = dict_stats[srun]
            flag = dict_run["flag_error"][srun]
    
            if stats < dict_config["stats_th"] or flag:
                logger.warning(f"\nFor run {obs_id} subrun {srun}:\nN_events = {stats}")
                logger.warning(f"Flag_error = {flag}\nSo interpolating neighbors.")
    
                # Search for the right neighbor until a valid one is found
                right_neighbor = srun + 1
                while right_neighbor <= last_srun and (
                    dict_run["statistics"][right_neighbor] < dict_config["stats_th"] or dict_run["flag_error"][right_neighbor]
                ):
                    right_neighbor += 1
    
                # Search for the left neighbor until a valid one is found
                left_neighbor = srun - 1
                while left_neighbor >= 0 and (
                    dict_run["statistics"][left_neighbor] < dict_config["stats_th"] or dict_run["flag_error"][left_neighbor]
                ):
                    left_neighbor -= 1
    
                # Case of two invalid subruns in a row
                if right_neighbor <= last_srun and left_neighbor >= 0:
                    dict_runs[obs_id]["final_scaling"][srun] = (
                        (dict_runs[obs_id]["final_scaling"][left_neighbor]) / 2 + 
                        (dict_runs[obs_id]["final_scaling"][right_neighbor]) / 2
                    )
                    dict_runs[obs_id]["delta_final_scaling"][srun] = (
                        (dict_runs[obs_id]["delta_final_scaling"][left_neighbor]) / 2 + 
                        (dict_runs[obs_id]["delta_final_scaling"][right_neighbor]) / 2
                    )
                elif right_neighbor <= last_srun:
                    dict_runs[obs_id]["final_scaling"][srun] = dict_runs[obs_id]["final_scaling"][right_neighbor]
                    dict_runs[obs_id]["delta_final_scaling"][srun] = dict_runs[obs_id]["delta_final_scaling"][right_neighbor]
                elif left_neighbor >= 0:
                    dict_runs[obs_id]["final_scaling"][srun] = dict_runs[obs_id]["final_scaling"][left_neighbor]
                    dict_runs[obs_id]["delta_final_scaling"][srun] = dict_runs[obs_id]["delta_final_scaling"][left_neighbor]
                else:
                    logger.warning(f"No valid neighbors found for run {obs_id} subrun {srun}. Unable to interpolate.")

    #############################################
    for ir, obs_id in enumerate(obs_ids_results):

        nsubruns = len(dict_config["dicts"]["sruns"][str(obs_id)])

        # Then calculating the interpolated values
        logger.info(f"Interpolating... Run{obs_id} - {ir / len(obs_ids_results) * 100:.1f}%")
        dict_results = dict_runs[obs_id]
    
        sruns_array = np.sort(list(dict_results["final_scaling"].keys()))
        x_fit = np.cumsum([dict_config["dicts"]["telapsed"][str(obs_id)][str(s)] for s in sruns_array])
        y_fit = np.array([dict_results["final_scaling"][srun] for srun in sruns_array])
        yerr_fit = np.array([dict_results["delta_final_scaling"][srun] for srun in sruns_array])
        
        nan_mask = ~(np.isnan(x_fit) | np.isnan(y_fit) | np.isnan(yerr_fit))

        mask_nan = (np.isnan(x_fit) | np.isnan(y_fit) | np.isnan(yerr_fit))
        mask_zero_error = np.array([e < 1e-4 for e in yerr_fit])
        
        mean_scaling, std_scaling = np.mean(y_fit), np.std(y_fit)
        mask_high_scale = np.array([y > 2.0 or y > mean_scaling + 2 * std_scaling for y in y_fit])
        mask_low_scale  = np.array([y < mean_scaling - 2 * std_scaling for y in y_fit])
        
        mask_fit = ~(mask_zero_error | mask_nan | mask_high_scale | mask_low_scale)
    
        x_fit_masked = x_fit[mask_fit]
        y_fit_masked = y_fit[mask_fit]
        yerr_fit_masked = yerr_fit[mask_fit]
        
        # Performing the fit
        params, pcov, info, _, _ = curve_fit(
            f     = geom.straight_line,
            xdata = x_fit_masked,
            ydata = y_fit_masked,
            sigma = yerr_fit_masked,
            p0    = [1, 0],
            full_output = True,
        )
            
        intercept, slope = params
        delta_intercept = np.sqrt(pcov[0, 0])
        delta_slope     = np.sqrt(pcov[1, 1])
        _chi2           = np.sum(info["fvec"] ** 2)
        ndf             = len(x_fit_masked)
        pvalue          = 1 - chi2.cdf(_chi2, ndf)
        
        fit_run_x0, fit_run_x1 = params
        fit_run_delta_x0 = np.sqrt(pcov[0, 0])
        fit_run_delta_x1 = np.sqrt(pcov[1, 1])
        fit_run_chi2 = np.sum(info["fvec"] ** 2)
        fit_run_ndf = len(x_fit_masked)
        fit_run_pvalue = 1 - chi2.cdf(fit_run_chi2, fit_run_ndf)
        
        # Plotting for debuging $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        
        
        dict_results["interpolation"] = {
            "chi2" : fit_run_chi2, "ndf"  : fit_run_ndf, "pvalue" : fit_run_pvalue,
            "x0" : fit_run_x0, "delta_x0" : fit_run_delta_x0, # intercept
            "x1" : fit_run_x1, "delta_x1" : fit_run_delta_x1, # slope
        }
        
        # Setting a interpolated scaling factor
        for srun in dict_results["final_scaling"].keys():
            
            scaling_interpolated = fit_run_x0 + fit_run_x1 * x_fit[srun]
            
            dict_results["final_scaling_interpolated"][srun] = scaling_interpolated
            dict_results["scaled"]["final"][srun]            = scaling_interpolated

        ####################
        # Storing the object
        dict_fname = os.path.join(dict_config["root_results_final"], f"results_job_{obs_id}.pkl")
        
        with open(dict_fname, "wb") as f:
            pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)

#     ######################################
#     # Cleaning the temporal results folder
#     for file in glob.glob(os.path.join(dict_config["root_results"], "*.pkl")):
#         command_rm = f"rm {file}"
#         subprocess.run(command_rm, shell=True)

    logger.info("SUCCESFULLY FINISHED")

    
def main_final(input_job_str, file_config, simulate_data=False):
    logger.info(f"Input string: {input_job_str}")
    
    if isinstance(simulate_data, str):
        simulate_data = simulate_data.lower() == "true"
    
    ########################################
    # Initial configuring
    # Reading the configuration file
    with open(file_config, "r") as json_file:
        dict_config = json.load(json_file)
    
    # Other auxiliar scripts
    sys.path.insert(0, dict_config["root_scripts"])
    import geometry as geom
    
    # Extracting the run number from the input string
    obs_id = int(input_job_str.split("_")[0])
    first_last_srun = [int(s) for s in input_job_str.split("_")[1:]]
    if len(first_last_srun) == 1:
        srun_numbers = np.array(first_last_srun)
    else:
        srun_numbers = np.arange(first_last_srun[0], first_last_srun[1] + 1)
    
    ########################################################
    # Binning    
    binning_intensity = np.array(dict_config["binning_intensity"])
    binning_intensity_c = (binning_intensity[1:] * binning_intensity[:-1]) ** 0.5 # Center (log) of binning
    binning_intensity_w = np.diff(binning_intensity) # Width of bins    
    # Mask for the fitting region in the fits
    mask_dcheck_bins_fit = (
        (binning_intensity_c >= dict_config["lims_intensity"][0]) &
        (binning_intensity_c <= dict_config["lims_intensity"][1])
    )

    ##########################################################
    # Reading the histogram data from the dl1 ---------------
    dict_srunwise = {} # Will contain: "file_dl1", "data_counts_intensity", "data_rates", "data_delta_rates"

    for srun in srun_numbers:
        dict_srunwise[srun] = {}
        
        # First we find the files, and we store them in dict_srunwise
        query_dl1_srun = np.sort(glob.glob(os.path.join(dict_config["root_dl1"], f"dl1_LST-1.Run{obs_id}.{srun:04}.h5")))
        if len(query_dl1_srun) == 0:
            logger.error(f"No dl2 file found for Run {obs_id}"); sys.exit()
        file_dl1 = query_dl1_srun[0]    
        dict_srunwise[srun]["file_dl1"] = file_dl1
        
        # Reading the file
        table_data = tables.open_file(file_dl1)
        data_counts_intensity, _ = np.histogram(
            table_data.root.dl1.event.telescope.parameters.LST_LSTCam.col("intensity"), 
            bins = binning_intensity,
        ); table_data.close()
        
        # Normalize by time and bin size
        effective_time_srun = dict_config["dicts"]["telapsed"][str(obs_id)][str(srun)]
        dict_srunwise[srun]["data_counts_intensity"] = data_counts_intensity
        dict_srunwise[srun]["data_rates"] = (data_counts_intensity/effective_time_srun/binning_intensity_w)
        dict_srunwise[srun]["data_delta_rates"] = (np.sqrt(data_counts_intensity)/effective_time_srun/binning_intensity_w)

    ##############################################################################################################
    # Scaling the final files and also saving the final results dictionary with all the information o fthe process
    file_tmp_results_input  = os.path.join(dict_config["root_results_final"], f"results_job_{obs_id}.pkl")
    file_tmp_results_output = os.path.join(dict_config["root_results"], f"results_job_{input_job_str}.pkl")
    
    # Reading the object
    with open(file_tmp_results_input, "rb") as f:
        dict_results = pickle.load(f)
    
    logger.info(f"\n\nPerforming \"final\" step\n{'-' * 20}\n")
    dict_results = utils_scaling.find_scaling(
        iteration_step = "final", 
        dict_results = dict_results,
        input_job_str = input_job_str,
        dict_config = dict_config, 
        simulate_data = simulate_data,
    )
    
    # Saving the object again
    with open(file_tmp_results_output, "wb") as f:
        pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)
        
    logger.info("SUCCESFULLY FINISHED")

if __name__ == "__main__":

    func_name = sys.argv[1]
    func_args = sys.argv[2:]
    
    globals()[func_name](*func_args)        