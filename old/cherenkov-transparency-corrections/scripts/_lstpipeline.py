import numpy as np
import glob, os, sys

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def find_run_datacheck(run, dchecking=False):
    str_dchecks = "" if not dchecking else "datacheck_"
    dl1_root = "/fefs/aswg/data/real/DL1/*/v*/tailcut84/"
    main_name = f"{str_dchecks}dl1_LST-1.Run?????"
    total_dl1a_runwise = glob.glob(dl1_root + "*/" + f"{main_name}.h5") + glob.glob(dl1_root + f"{main_name}.h5")

    logger.info(f"\nFinding datacheck for Run {run}...")
        
    # Checking for files of this certain run
    runfiles = []
    for rf in total_dl1a_runwise:
        if f"{run:05}" in rf:
            runfiles.append(rf)
    
    # Checking runs we have, not, or we have duplicated
    if len(runfiles) == 0:
        raise ValueError(f"Run {run:5} not found in {dl1_root}")
    
    elif len(runfiles) > 1:
        logger.debug(f"Run {run:5} presented {len(runfiles)} files:")
        versions = []
        lengths_versions = []
        for i, runfile in enumerate(runfiles):
            str_version = runfile.split("/")[7][1:]
            lengths_versions.append(len(str_version))
            str_parts = str_version.split(".")
            str_parts_float = [float(''.join(filter(str.isdigit, str_parts[iii]))) for iii in range(len(str_parts))]
            for pi, part in enumerate(str_parts_float):
                if pi == 0:
                    final_str = f"{part:04.0f}."
                else:
                    final_str = final_str + f"{part:04.0f}"                
            versions.append(float(final_str)) 
    
        version_index = 0
        for i in range(1, len(versions)):
            if versions[i] > versions[version_index]:
                version_index = i
            elif versions[i] == versions[version_index] and lengths_versions[i] < lengths_versions[version_index]:
                version_index = i
        
        for i, runfile in enumerate(runfiles):
            selected = "(SELECTED)" if i == version_index else ""
            logger.debug(f"--> {runfile} {selected}")
    
        run_path  = runfiles[version_index]
    
    else:
        run_path  = runfiles[0]

    return run_path


def add_dl1_paths_to_dict(DICT, dl1_root, dchecking=False):
    """
    Add DL1 file paths to a dictionary.

    Args:
        DICT (dict): The dictionary to which the DL1 file paths will be added.
        dl1_root (str): The root directory of the DL1 files.
        dchecking (bool, optional): Whether to perform data checking. Defaults to False.

    Returns:
        dict: The updated dictionary with DL1 file paths added.

    """
    str_dchecks = "" if not dchecking else "datacheck_"
    log_errors  = ""

    main_name = f"{str_dchecks}dl1_LST-1.Run?????"
    # Finding all datacheck files for run-wise and subrun-wise
    total_dl1a_runwise    = glob.glob(dl1_root + "*/" + f"{main_name}.h5")      + glob.glob(dl1_root + f"{main_name}.h5")
    total_dl1a_subrunwise = glob.glob(dl1_root + "*/" + f"{main_name}.????.h5") + glob.glob(dl1_root + f"{main_name}.????.h5")
    # logger.info(f"DL1 files:  Found {len(total_dl1a_runwise):4} run-wise and {len(total_dl1a_subrunwise):6} subrun-wise")
    # logger.info(f"Datachecks: Found {len(total_dcheck_runwise):4} run-wise and {len(total_dcheck_subrunwise):6}  subrun_wise\n")

    for run in DICT.keys():

        logger.info(f"\nAdding dl1 {str_dchecks[:-1]} data to dictionary (Run {run})...")
        
        # Checking for files of this certain run
        runfiles = []
        for rf in total_dl1a_runwise:
            if f"{run:05}" in rf:
                runfiles.append(rf)

        # Checking runs we have, not, or we have duplicated
        if len(runfiles) == 0:
            raise ValueError(f"Run {run:5} not found in {dl1_root}")
        
        elif len(runfiles) > 1:
            logger.debug(f"Run {run:5} presented {len(runfiles)} files:")
            versions = []
            lengths_versions = []
            for i, runfile in enumerate(runfiles):
                str_version = runfile.split("/")[7][1:]
                lengths_versions.append(len(str_version))
                str_parts = str_version.split(".")
                str_parts_float = [float(''.join(filter(str.isdigit, str_parts[iii]))) for iii in range(len(str_parts))]
                for pi, part in enumerate(str_parts_float):
                    if pi == 0:
                        final_str = f"{part:04.0f}."
                    else:
                        final_str = final_str + f"{part:04.0f}"                
                versions.append(float(final_str)) 

            version_index = 0
            for i in range(1, len(versions)):
                if versions[i] > versions[version_index]:
                    version_index = i
                elif versions[i] == versions[version_index] and lengths_versions[i] < lengths_versions[version_index]:
                    version_index = i
            
            for i, runfile in enumerate(runfiles):
                selected = "(SELECTED)" if i == version_index else ""
                logger.debug(f"--> {runfile} {selected}")

            run_path  = runfiles[version_index]

        else:
            run_path  = runfiles[0]        

        # Checking subruns
        _subrun_files, _subrun_num = [], []
        for f in total_dl1a_subrunwise:
            if f"{run:05}" in f:
                _subrun_files.append(f)
                _subrun_num.append(int(f.split(".")[-2]))
        
        # Sorting subruns
        subrun_num, subrun_files = sort_based( _subrun_files, _subrun_num)

        # first we make sure there is at least one subrun
        if len(subrun_files) == 0:
            raise ValueError(f"No subrun found in {dl1_root}.")    

        # checking we have all subruns, or if some is repeated
        _subrun_dict_ = {}
        for i in range(subrun_num[-1]+1):
            _subrun_dict_[i] = []

        for i, srunfile in enumerate(subrun_files):
            subrun = subrun_num[i]
            _subrun_dict_[subrun].append(srunfile)

        # Check if there is more than one file associated with the same subrun
        subrun_dict  = {} # this will be the final dict with only one file per subrun
        subrun_paths = []
        for subrun, files in _subrun_dict_.items():

            if len(files) == 0:
                raise ValueError(f"Subrun {subrun:04} not found in {dl1_root}.")
            elif len(files) > 1:
                logger.debug(f"Subrun {subrun:04} presented {len(files)} files:")

                versions = []
                lengths_versions = []
                for i, srunfile in enumerate(files):
                    str_version = srunfile.split("/")[7][1:]
                    lengths_versions.append(len(str_version))
                    str_parts = str_version.split(".")
                    str_parts_float = [float(''.join(filter(str.isdigit, str_parts[iii]))) for iii in range(len(str_parts))]
                    for pi, part in enumerate(str_parts_float):
                        if pi == 0:
                            final_str = f"{part:04.0f}."
                        else:
                            final_str = final_str + f"{part:04.0f}"                
                    versions.append(float(final_str)) 

                version_index = 0
                for i in range(1, len(versions)):
                    if versions[i] > versions[version_index]:
                        version_index = i
                    elif versions[i] == versions[version_index] and lengths_versions[i] < lengths_versions[version_index]:
                        version_index = i

                for i, subrunfile in enumerate(files):
                    selected = "(SELECTED)" if i == version_index else ""
                    logger.debug(f"--> {subrunfile} {selected}")

                subrun_dict[subrun] = files[version_index]
                subrun_paths.append(files[version_index])
            else:
                subrun_dict[subrun] = files[0]
                subrun_paths.append(files[0])
        
        # checking if the branch of dict exists
        str_dchecks_dict = "dl1a" if not dchecking else "dchecks"
        try: 
            DICT[run][str_dchecks_dict]["runwise"]  = run_path
            DICT[run][str_dchecks_dict]["srunwise"] = subrun_paths
        except KeyError:
            DICT[run][str_dchecks_dict] = {
                "runwise"  : run_path, 
                "srunwise" : subrun_paths,
            }
                
    logger.info(f"...Finished adding dl1 data to dictionary")
    return DICT


def angular_dist(az1, az2):
    """
    Calculate the angular distance between two azimuth angles.

    Parameters:
    az1 (float): The first azimuth angle in degrees.
    az2 (float): The second azimuth angle in degrees.

    Returns:
    float: The angular distance between the two azimuth angles.
    """
    angular_distance_abs = abs(az1 - az2)
    return min(angular_distance_abs, 360 - angular_distance_abs)

def add_rf_node(DICT, rfs_root, dict_source):
    """
    Add MC and RF nodes to the given dictionary.

    Parameters:
    - DICT (dict): The dictionary to which the MC and RF nodes will be added.
    - rfs_root (str): The root directory of the RF nodes.
    - dict_source (dict): The dictionary containing the source information.

    Returns:
    - dictionary: The updated dictionary (DICT) and a dictionary of nodes (dict_nodes).
    """
    
    # finding the RF nodes in DEC
    rfs_decs = os.listdir(rfs_root)
    rf_nodes_dec = np.array([float(d.split("_")[-1][:-2] + "." + d.split("_")[-1][-2:]) for d in rfs_decs])

    dist_rfs = np.abs(rf_nodes_dec - dict_source["dec"].value)
    closest_rf_node = rfs_decs[np.argmin(dist_rfs)]

    for run in DICT.keys():

        DICT[run]["simulations"] = {
            "rf" : rfs_root + closest_rf_node,
        }

    return DICT

def sort_based(x_array, ref_array):
    """
    Sorts the array ref_array in ascending order and rearranges the array x_array based on the sorted ref_array values.

    Parameters:
    x_array (array-like): The array to be rearranged based on ref_array.
    ref_array (array-like): The reference array used for sorting.

    Returns:
    tuple: A tuple containing two arrays. The first array is the sorted ref_array, and the second array is x_array 
    rearranged based on the sorted ref_array values.
    """
    return np.sort(ref_array), np.array([x for ref, x in sorted(zip(ref_array, x_array))])