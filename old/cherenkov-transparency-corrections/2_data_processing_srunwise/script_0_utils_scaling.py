import matplotlib.pyplot as plt
import numpy as np
import json, sys, os, copy, tables, subprocess, glob
from scipy.optimize import curve_fit
from scipy.stats import chi2

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

"""
Empty results dictionary to be filled while the processing
"""
dict_results_empty = { 
    "obs_id": None, "file_dl1_final": {}, "statistics": {}, "flag_error" : {},
    "final_scaling": {}, "delta_final_scaling": {}, "final_scaling_interpolated": {}, "interpolation" : {},
}
for key in [
    "scaled", "p0", "delta_p0", "p1", "delta_p1", "chi2", "ndf",
    "pvalue", "light_yield", "delta_light_yield", "scaling", "delta_scaling"
]:
    dict_results_empty[key] = {"original": {}, "upper": {}, "linear": {}, "final": {}}

    
def find_scaling(iteration_step, dict_results, input_job_str, dict_config, simulate_data=False):
    """
    A function to perform scaling and evaluating the results. Returning everything in a updated dictionary

    Input:
    - iteration_step: (str) 
        The iteration step you are in, that can be:
            - "original" - For the original data
            - "upper"    - For the upper limit on the scale factor
            - "linear"   - For the linear intepolation factor and
            - "final"    - For the final scaling and results.
        
    - dict_results: (dict)
        Dictionary with the results of the before step.
        
    - input_job_str (str)
        Input containing run and first and last subrun to be analysed: <obs_id>_<srun_init>_<srun_final>
        
    - other_parameters (dict)
        A dictionary with all other needed configuration parameters.
        
    - simulate_data: (bool)
        If True random data is generated just to fill the values. To run tests.

    """
    
    # Import other auxiliar scripts
    sys.path.insert(0, dict_config["root_scripts"])
    import geometry as geom
    import utils
    
    # Extracting the run number from the input string
    obs_id = int(input_job_str.split("_")[0])
    first_last_srun = [int(s) for s in input_job_str.split("_")[1:]]
    if len(first_last_srun) == 1:
        srun_numbers = np.array(first_last_srun)
    else:
        srun_numbers = np.arange(first_last_srun[0], first_last_srun[1] + 1)

        
    # Zenith corrections to the parameters
    #########################################################
    p0a, p0b, p0c = dict_config["p0a"], dict_config["p0b"], dict_config["p0c"]
    p1a, p1b, p1c = dict_config["p1a"], dict_config["p1b"], dict_config["p1c"]
    srun_zds = np.array([dict_config["dicts"]["zd"][str(obs_id)][str(srun)] for srun in srun_numbers])
    
    corr_factor_p0 = geom.pol2(1, p0a, p0b, p0c) / geom.pol2(np.cos(srun_zds), p0a, p0b, p0c)
    corr_factor_p1 = geom.pol2(1, p1a, p1b, p1c) - geom.pol2(np.cos(srun_zds), p1a, p1b, p1c) 
    
    # Empty arrays to store the fit information
    data_p0, data_delta_p0 = [], []
    data_p1, data_delta_p1 = [], []
    data_chi2, data_pvalue = [], []
    data_ndf = []
    
    # Processing subrun by subrun---------------------------------------------------------------
    for s, srun in enumerate(srun_numbers):    

        # Reading dl1
        #################################################
        # Finding DL1a file for run-subrun
        query_dl1_srun = glob.glob(os.path.join(dict_config["root_dl1"], f"dl1_LST-1.Run{obs_id:05}.{srun:04}.h5"))
        if len(query_dl1_srun) == 0:
            logger.error(f"No DL1a file found for Run {obs_id} - Subrun {srun}"); sys.exit()        
        file_input_dl1 = query_dl1_srun[0]
        
        data_scale_factor = dict_results["scaled"][iteration_step][srun]   # Reading the scaling factor

        # Here we do different things depending on the iteration step
        # ////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////
        # If is the first one i.e. == "original"
        # We do not run lstchain_dl1ab because the data is already scaled
        if iteration_step == "original":
            file_output_dl1 = file_input_dl1

        # ////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////
        # If is the second or third: "upper" or "linear"
        # We perform lstchain_dl1ab but over a subset of the data only to keep it shorter
        elif iteration_step in ["upper", "linear"]:
            logger.info(f"\nProcessing subrun {srun}")

            # Temporal dl1 file that will be overwritten in the next iteration / subrun
            file_output_dl1 = os.path.join(
                dict_config["root_sub_dl1"], f"tmp_dl1_srunwise_run{obs_id}_srun{srun}_{iteration_step}_scaled.h5"
            )

            # If scale is greater than 1 we select a range lower than the upper one
            # otherwise we select a range higher than the upper one
            if data_scale_factor > 1:
                dl1_selected_range = f"{dict_config['lims_intensity_extended']:.2f},{dict_config['lims_intensity'][1]:.2f}"
            else:
                dl1_selected_range = f"{dict_config['lims_intensity'][0]:.2f},inf"

            if not simulate_data:
                logger.info(f"Running lstchain_dl1ab... Scale: {data_scale_factor:.2f}")
                # If the file already exists we delete it
                if os.path.exists(file_output_dl1):
                    os.remove(file_output_dl1)
                
                command_dl1ab  = f"lstchain_dl1ab --input-file {file_input_dl1} --output-file {file_output_dl1} "
                command_dl1ab += f"--config {dict_config['file_config_lstchain_dl2']} --no-image "
                command_dl1ab += f"--light-scaling {data_scale_factor} --intensity-range {dl1_selected_range}"
                logger.debug(command_dl1ab)

                # subprocess.run(command_dl1ab, shell=True)
                # We add an exception because sometimes can fail...
                dl1_creation_suceeded, ntries = False, copy.copy(dict_config["number_tries_dl1"])
                while ntries > 0 and (not dl1_creation_suceeded):
                    subprocess.run(command_dl1ab, shell=True) # Running DL1ab
                    
                    # Now we check if the file exists
                    if os.path.exists(file_output_dl1):
                        dl1_creation_suceeded = True
                    else:
                        logger.error(f"Try {dict_config['number_tries_dl1'] - ntries}/{dict_config['number_tries_dl1']}")
                        logger.error(f"No file created with: {command_dl1ab}"); ntries = ntries - 1
                if not dl1_creation_suceeded:
                    logger.error(f"No file created after {dict_config['number_tries_dl1']} tries."); sys.exit()

        # ////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////
        # If is the last step i.e. "final"
        # The lstchain_dl1ab script is run over all the dataset to generate the final file
        elif iteration_step == "final":
            logger.info(f"\nProcessing subrun {srun}")

            file_output_dl1 = os.path.join(
                dict_config["dir_dl1"], f"Run{obs_id:05}", os.path.basename(file_input_dl1)
            )
            os.makedirs(os.path.join(dict_config["dir_dl1"], f"Run{obs_id:05}"), exist_ok=True)
            
            if not simulate_data:
                logger.info(f"Running lstchain_dl1ab... Scale: {data_scale_factor:.2f}")
                # If the file already exists we delete it
                if os.path.exists(file_output_dl1):
                    os.remove(file_output_dl1)
               
                command_dl1ab  = f"lstchain_dl1ab --input-file {file_input_dl1} --output-file {file_output_dl1} "
                command_dl1ab += f"--config {dict_config['file_config_lstchain_dl2']} --no-image "
                command_dl1ab += f"--light-scaling {data_scale_factor}" # No intensity range is used in this case
                logger.info(command_dl1ab)

                # subprocess.run(command_dl1ab, shell=True)
                # We add an exception because sometimes can fail...
                dl1_creation_suceeded, ntries = False, copy.copy(dict_config["number_tries_dl1"])
                while ntries > 0 and (not dl1_creation_suceeded):
                    subprocess.run(command_dl1ab, shell=True) # Running DL1ab

                    # Now we check if the file exists
                    if os.path.exists(file_output_dl1):
                        dl1_creation_suceeded = True
                    else:
                        logger.error(f"Try {dict_config['number_tries_dl1'] - ntries}/{dict_config['number_tries_dl1']}")
                        logger.error(f"No file created with: {command_dl1ab}"); ntries = ntries - 1
                if not dl1_creation_suceeded:
                    logger.error(f"No file created after {dict_config['number_tries_dl1']} tries."); sys.exit()
    
            # We store this info also in the dictionary in the final case
            dict_results["file_dl1_final"][srun] = file_output_dl1

        # Defining the intensity binning parameters
        binning_intensity = np.array(dict_config["binning_intensity"])
        binning_intensity_c = (binning_intensity[1:] * binning_intensity[:-1]) ** 0.5 # Center (log) of binning
        binning_intensity_w = np.diff(binning_intensity) # Width of bins
        # Mask for the fitting region in the fits
        mask_dcheck_bins_fit = (
            (binning_intensity_c >= dict_config["lims_intensity"][0]) &
            (binning_intensity_c <= dict_config["lims_intensity"][1])
        )
    
        #################################################################
        # Reading the dl1 file
        #################################################################
        if not simulate_data:
            table_data = tables.open_file(file_output_dl1)
            data_counts_intensity, _ = np.histogram(
                table_data.root.dl1.event.telescope.parameters.LST_LSTCam.col("intensity"), 
                bins = binning_intensity,
            ); table_data.close()
        else:
            # Simulated example data where we add random noise
            simdata = np.loadtxt(os.path.join(dict_config["root_objects"], "simulated_data_base.txt"), dtype=int)
            
            dict_multipliers = {"original": 1.00, "upper": 0.30, "linear": 0.15, "final": 0.18}
            if iteration_step in dict_multipliers:
                data_counts_intensity = simdata * dict_multipliers[iteration_step] + np.random.rand(100) * 100
        
        # Calculating the non binning dependent transformation [ev / s / p.e.]
        effective_time_srun = dict_config["dicts"]["telapsed"][str(obs_id)][str(srun)]
        
        data_rates = data_counts_intensity / effective_time_srun / binning_intensity_w
        # The error on the rates is just the statistical error: sqrt(counts) / Delta t / Delta I
        data_delta_rates = np.sqrt(data_counts_intensity) / effective_time_srun / binning_intensity_w


        #################################################################
        # Performing the fit
        #################################################################
        # Displacing the X-coordinates to the center of the fit, in order to decorrelate the fit
        x_fit = binning_intensity_c[mask_dcheck_bins_fit] / dict_config["ref_intensity"]
        y_fit = data_rates[mask_dcheck_bins_fit]
        yerr_fit = data_delta_rates[mask_dcheck_bins_fit]
        
        try:
            params, pcov, info, _, _ = curve_fit(
                f     = geom.powerlaw,
                xdata = x_fit,
                ydata = y_fit,
                sigma = yerr_fit,
                p0    = [dict_config["ref_p0"], dict_config["ref_p1"]],
                full_output = True,
            )
        
            srun_p0, srun_delta_p0 = params[0], np.sqrt(pcov[0, 0])
            srun_p1, srun_delta_p1 = params[1], np.sqrt(pcov[1, 1])
            srun_chi2 = np.sum(info["fvec"] ** 2)
            srun_ndf = sum(mask_dcheck_bins_fit)
            srun_pvalue = 1 - chi2.cdf(srun_chi2, srun_ndf)
            dict_results["flag_error"][srun] = False

        # If the fit is not successful we return nan 
        # values and activate the error flag
        except RuntimeError:
            logger.error(f"For run {obs_id} and subrun {srun}, the fit failed due to RuntimeError.")
            srun_p0, srun_p1 = np.nan, np.nan
            srun_delta_p0, srun_delta_p1 = np.nan, np.nan
            srun_chi2 = np.nan
            srun_ndf = sum(mask_dcheck_bins_fit)
            srun_pvalue = np.nan
            dict_results["flag_error"][srun] = True
            
        # Plotting for debuging $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        fig, ax = plt.subplots()

        ax.plot(binning_intensity_c, data_rates, color="k", label="All data")
        ax.plot(x_fit * dict_config["ref_intensity"], y_fit, "+", color="r", label="Fit points")

        x = np.logspace(2, 3.6, 100)
        ax.plot(x, geom.powerlaw(x / dict_config["ref_intensity"], srun_p0, srun_p1), 
                color="darkorange", ls="-", label="Fit PWL")

        ax.plot(x, geom.powerlaw(x / dict_config["ref_intensity"], dict_config["ref_p0"] / corr_factor_p0[s], 
            dict_config["ref_p1"] - corr_factor_p1[s]), color="r", ls="--", label="Ref PWL (ZD)")
        ax.plot(x, geom.powerlaw(x / dict_config["ref_intensity"], dict_config["ref_p0"], dict_config["ref_p1"]), 
                color="k", ls="--", label="Ref PWL")

        ax.axvline(dict_config["ref_intensity"], ls=":", color="k", label=f"Ref I {dict_config['ref_intensity']} p.e.")
        ax.axhline(srun_p0, ls="--", color="r", label=f"P0 = {srun_p0:.2f}")
        ax.axvspan(*dict_config["lims_intensity"], color="0.2", alpha=0.3, label="Fit Region")

        str_title = f"Step {iteration_step}, Scale {dict_results['scaled'][iteration_step][srun]:.2f}"
        ax.set_title(f"Run {obs_id} Subrun {srun}, {str_title}")
        ax.loglog(); ax.legend()
        ax.set_xlim(0.9e2, 3e3); ax.set_ylim(3e-2, 2e1)
        ax.set_xlabel("Intensity [p.e.]"); ax.set_ylabel("Rate [ev/s/p.e.]")
        plt.savefig(os.path.join(dict_config["root_tmp_plots"], f"tmp_{iteration_step}_{obs_id}.{srun}.png"))
        plt.show()
        
        # Those parameters do not need ZD correction
        dict_results["chi2"][iteration_step][srun]   = srun_chi2
        dict_results["ndf"][iteration_step][srun]    = srun_ndf
        dict_results["pvalue"][iteration_step][srun] = srun_pvalue
        dict_results["scaled"][iteration_step][srun] = data_scale_factor
    
        data_p0.append(srun_p0); data_delta_p0.append(srun_delta_p0)
        data_p1.append(srun_p1); data_delta_p1.append(srun_delta_p1)
        data_chi2.append(srun_chi2); data_ndf.append(srun_ndf)
        data_pvalue.append(srun_pvalue)
    
    # Convert to numpy arrays
    data_p0, data_delta_p0 = np.array(data_p0), np.array(data_delta_p0)
    data_p1, data_delta_p1 = np.array(data_p1), np.array(data_delta_p1)
    data_chi2, data_ndf, data_pvalue = np.array(data_chi2), np.array(data_ndf), np.array(data_pvalue)
  
    # Zenith corrections to the parameters
    #########################################################    
    data_corr_p0 = data_p0 * corr_factor_p0
    data_corr_p1 = data_p1 + corr_factor_p1
    
    data_corr_delta_p0 = data_delta_p0 * corr_factor_p0
    data_corr_delta_p1 = data_delta_p1
    
    # Calculating the needed light yield  
    data_light_yield, data_delta_light_yield = geom.calc_light_yield(
        p0_fit = data_corr_p0,
        p1_fit = data_corr_p1, 
        sigma_p0_fit = data_corr_delta_p0, 
        sigma_p1_fit = data_corr_delta_p1, 
        p0_ref = dict_config["ref_p0"],
    )
    logger.info(f"Step \"{iteration_step}\" mean(LY) = {np.mean(data_light_yield):.3f}")
    
    # Scalings to apply
    data_scaling       = (1 / data_light_yield)
    data_delta_scaling = (1 / data_light_yield) ** 4 * data_delta_light_yield
    
    # Adding to dictionary
    for i, srun in enumerate(srun_numbers):
        dict_results["p0"][iteration_step][srun]       = data_corr_p0[i]
        dict_results["delta_p0"][iteration_step][srun] = data_corr_delta_p0[i]
        dict_results["p1"][iteration_step][srun]       = data_corr_p1[i]
        dict_results["delta_p1"][iteration_step][srun] = data_corr_delta_p1[i]  
        dict_results["light_yield"][iteration_step][srun]       = data_light_yield[i]
        dict_results["delta_light_yield"][iteration_step][srun] = data_delta_light_yield[i]
        dict_results["scaling"][iteration_step][srun]           = data_scaling[i]
        dict_results["delta_scaling"][iteration_step][srun]     = data_delta_scaling[i]

    return dict_results
    