#!/bin/bash

# The file with the list of jobs to be sent
file="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/high_level_analysis/objects/test_scan_flux_dispersion_config_jobs.txt"
# python
root_scripts="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/high_level_analysis/"
python_script="$root_scripts""test_scan_flux_dispersion_script.py"

# Read the run numbers from the input text file
input_strs=$(cat $file)

for run_str in $input_strs; do
    # Create a temporary bash script for each run after merge
    tmp_script_final="$root_scripts""job_tmp.sh"
    echo "#!/bin/bash" > $tmp_script_final
    echo "python $python_script '$run_str'" >> $tmp_script_final

    # Submit the job with dependency on the merge job
    job_id=$(sbatch -p short $tmp_script_final | awk '{print $4}')
    echo "Final job submitted for energy $run_str, with Job ID $job_id"
done