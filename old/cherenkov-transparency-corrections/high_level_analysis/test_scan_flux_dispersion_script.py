import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle, sys, os, json
import astropy.units as u
import copy
import pandas as pd
pd.set_option("display.max_columns", None)

from astropy.coordinates import SkyCoord
from matplotlib.dates    import DayLocator, MonthLocator, DateFormatter
from regions             import PointSkyRegion
from astropy.time        import Time
from scipy.stats         import chi2, norm

from gammapy.modeling.models import create_crab_spectral_model, SkyModel, LogParabolaSpectralModel
from gammapy.estimators      import FluxPointsEstimator, LightCurveEstimator, FluxPoints
from gammapy.modeling        import Fit
from gammapy.datasets        import Datasets, SpectrumDataset
from gammapy.makers          import SpectrumDatasetMaker, WobbleRegionsFinder, ReflectedRegionsBackgroundMaker, SafeMaskMaker
from gammapy.maps            import MapAxis, RegionGeom, Map, TimeMapAxis
from gammapy.data            import DataStore

# import scripts
sys.path.insert(0, os.getcwd() + "/../scripts/")
import auxiliar as aux
import plotting


def main(limit):

    limit = float(limit)
    # Root path of this script
    root = os.getcwd() + "/"
    # Objects directory
    root_objects = root + "objects/"
    # Data directory
    root_data = root + "../../data/"
    
    # Gammapy configuration file
    config_gammapy_path = root_objects + "config_gammapy_analysis.json"
    
    # Data labels
    dataset_labels = ["standard", "scaled", "paper"]
    # Color plots
    color_plots = ["b", "r", "gray"]
    
    # Path of dl3 data
    dl3_dirs = ["/fefs/aswg/workspace/juan.jimenez/data/cherenkov_transparency_corrections/crab/dl3/",
                "/fefs/aswg/workspace/juan.jimenez/data/cherenkov_transparency_corrections/crab/dl3_scaled/",
                "/fefs/aswg/workspace/juan.jimenez/data/tests/crab/dl3_paper/"]
    
    compute_datasets = True

    pkl_paths_dataset    = []
    pkl_paths_lightcurve = []
    for label in dataset_labels:
        pkl_paths_dataset.append(os.path.join(root, "objects", f"tmp_dataset_{label}.pkl"))
        pkl_paths_lightcurve.append(os.path.join(root, "objects", f"tmp_lightcurve_{label}.pkl"))
    
    # Reading the configuration for gammapy we created
    with open(config_gammapy_path, "r") as json_file:
        config_gammapy = json.load(json_file)
    
    # Saving part of the configuration in variables
    target_name   = config_gammapy["target_name"]
    n_off_regions = config_gammapy["n_off_regions"]
    _e_reco = config_gammapy["e_reco"]
    _e_true = config_gammapy["e_true"]
    
    e_reco_min, e_reco_max, e_reco_bin_p_dec = _e_reco["min"], _e_reco["max"], _e_reco["bins_p_dec"]
    e_true_min, e_true_max, e_true_bin_p_dec = _e_true["min"], _e_true["max"], _e_true["bins_p_dec"]
    
    # Energy for the lightcurve
    e_lc_min = limit * u.TeV
    e_lc_max = config_gammapy["e_lc"]["max"] * u.TeV
    print("ENERGIES: ", e_lc_min, " - ", e_lc_max, " TeV")
    
    print(f"\nLoading config file...\nSource: {target_name}")
    print(f"Reco Energy limits: {_e_reco['min']:}-{_e_reco['max']:} TeV ({_e_reco['bins_p_dec']} bins)")
    print(f"True Energy limits: {_e_true['min']:}-{_e_true['max']:} TeV ({_e_true['bins_p_dec']} bins)")
    print(f"LC Integration lim: {e_lc_min.value:}-{e_lc_max.value:} TeV")
    
    obs_ids      = [] # All run numbers directly read from the directory
    observations = [] # Observation information for each dataset
    
    for dl3_dir in dl3_dirs:
        # Opening all the dl3 data in a path
        total_data_store = DataStore.from_dir(dl3_dir)
        
        # Taking the obs ids
        _obs_ids = total_data_store.obs_table["OBS_ID"].data
        _obs_ids = _obs_ids[:]
        
        # Then we get the observation information from the total data store
        _observations = total_data_store.get_observations(
            _obs_ids,
            required_irf=["aeff", "edisp", "rad_max"]
        )
    
        obs_ids.append(_obs_ids)
        observations.append(_observations)
    
        print(f"\nReading {dl3_dir}...\nN runs: {len(_obs_ids)} ({min(_obs_ids)}-{max(_obs_ids)})")
        print(f"Livetime: {total_data_store.obs_table['LIVETIME'].data.sum()/3600:.2f} h")
    # display(total_data_store.obs_table[:5])
    
    # Defining target position and ON reion
    target_position = SkyCoord.from_name(target_name, frame="icrs")
    on_region = PointSkyRegion(target_position)
    
    # ============================ #
    # estimated energy axes
    energy_axis = MapAxis.from_energy_bounds(
        e_reco_min, e_reco_max, 
        nbin=e_reco_bin_p_dec, per_decade=True, 
        unit="TeV", name="energy"
    )
    # ============================ #
    # estimated energy axes
    energy_axis_true = MapAxis.from_energy_bounds(
        e_true_min, e_true_max, 
        nbin=e_true_bin_p_dec, per_decade=True, 
        unit="TeV", name="energy_true"
    )
    # ============================ #
    # Energy for the spectrum
    e_fit_min = energy_axis.edges[0].value
    e_fit_max = energy_axis.edges[-1].value
    e_fit_bin_p_dec = e_reco_bin_p_dec
    
    # Just to have a separate MapAxis for spectral fit energy range
    energy_fit_edges = MapAxis.from_energy_bounds(
        e_fit_min, e_fit_max, 
        nbin=e_fit_bin_p_dec, per_decade=True, 
        unit="TeV"
    ).edges
    
    # ============================ #
    
    # fig, ax = plt.subplots(figsize=(10,1))
    # for i in range(len(energy_fit_edges)-1):
    #     if i % 2 == 0:
    #         ax.axvspan(energy_fit_edges[i].value, energy_fit_edges[i+1].value, color="gray",)
    #     else:
    #         ax.axvspan(energy_fit_edges[i].value, energy_fit_edges[i+1].value, color="lightgray")
    
    # ax.axvspan(None, None, color="gray", label="SED bins")
    # ax.axvspan(e_lc_min.value, e_lc_max.value, facecolor="none", edgecolor="k", hatch="/", label="LC integration lims")
    # ax.set_yticks([])
    # ax.legend(loc=3)
    # ax.set_xscale("log")
    # ax.set_xlabel("Energy [TeV]")
    # plt.show()

    # geometry defining the ON region and SpectrumDataset based on it
    geom = RegionGeom.create(
        region=on_region, 
        axes=[energy_axis]
    )
    
    
    # -------------------------------------------------------
    # creating an empty dataset
    dataset_empty = SpectrumDataset.create(
        geom=geom, 
        energy_axis_true=energy_axis_true
    )
    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False,
        selection=["counts", "exposure", "edisp"]
    )
    # -------------------------------------------------------
    
    
    # tell the background maker to use the WobbleRegionsFinder
    region_finder = WobbleRegionsFinder(n_off_regions=n_off_regions)
    bkg_maker = ReflectedRegionsBackgroundMaker(region_finder=region_finder)
    
    
    datasets = []
    stacked_datasets = []
    for _observations, label, pkl_path in zip(observations, dataset_labels, pkl_paths_dataset):
        if compute_datasets:
            # -------------------------------------------------------
            # The final object will be stored as a Datasets object
            _datasets = Datasets()
            for obs in _observations:
                _dataset = copy.copy(dataset_maker).run(
                    dataset=copy.copy(dataset_empty).copy(name=str(obs.obs_id)),
                    observation=obs
                )
                dataset_on_off = bkg_maker.run(
                    dataset=_dataset, 
                    observation=obs
                )
                _datasets.append(dataset_on_off) 
    
            # Stacking the datasets in one
            _stacked_dataset = Datasets(_datasets).stack_reduce()
    
            datasets.append(_datasets)
            stacked_datasets.append(_stacked_dataset)
            # -------------------------------------------------------
    
            # # Storing objects
            # with open(pkl_path, 'wb') as f:
            #     pickle.dump(_datasets, f, pickle.HIGHEST_PROTOCOL)
        # else:
        #     with open(pkl_path, "rb") as f:
        #         _datasets = pickle.load(f)
        #     _stacked_dataset = Datasets(_datasets).stack_reduce()
            
            datasets.append(_datasets)
            stacked_datasets.append(_stacked_dataset)
            
        print(f"dataset = {label}")
        print(_stacked_dataset)

    # defining the model we want to fit and the starting values
    spectral_model = LogParabolaSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        alpha=2,
        beta=0.1,
        reference=1 * u.TeV,
    )
    # we will use the crab model in general
    models = []
    for stacked_dataset in stacked_datasets:
        _model = SkyModel(
            spectral_model=spectral_model, 
            name="crab"
        )
        # We set the model of all datasets to log parabola
        stacked_dataset.models = _model
        models.append(_model)

    best_fit_models = []
    for stacked_dataset, model in zip(stacked_datasets, models):
        # Now we run the fit to extract the parameters of the model
        fit = Fit()
        result = fit.run(datasets=stacked_dataset)
        best_fit_model = model
        # display(stacked_dataset.models.to_parameters_table())
    
        best_fit_models.append(best_fit_model.copy())
    
    # then extracting the flux points from the data
    fpe = FluxPointsEstimator(
        energy_edges=energy_fit_edges, 
        source=target_name, 
        selection_optional="all"
    )
    
    flux_points = []
    for stacked_dataset in stacked_datasets:
        fpe = FluxPointsEstimator(
            energy_edges=energy_fit_edges, 
            source=target_name, 
            selection_optional="all"
        )
        
        # We apply the flux point estiation from the datasets
        _flux_points = fpe.run(datasets=stacked_dataset)
        # display(_flux_points.to_table(sed_type="dnde", formatted=True)[:3])
        flux_points.append(_flux_points)


    # Create the LC Estimator for each run
    lc_maker_1d = LightCurveEstimator(
        energy_edges=[e_lc_min, e_lc_max], 
        reoptimize=False, # Re-optimizing other free model parameters (not belonging to the source)
        source="crab", 
        selection_optional="all" # Estimates asymmetric errors, upper limits and fit statistic profiles
    )
    
    for dataset, model in zip(datasets, models):
        model.parameters["alpha"].frozen = True
        model.parameters["beta"].frozen  = True
        # Assigning the fixed parameters model to each dataset
        for data in dataset:
            data.models = model
    
    print(f"LC will be estimated from {e_lc_min:} to {e_lc_max:} TeV")
    lcs_runwise = []
    lightcurves = []
    for pkl_path, dataset in zip(pkl_paths_lightcurve, datasets):
        if compute_datasets:
            _lc_runwise = lc_maker_1d.run(dataset)
            # Storing object
        #     with open(pkl_path, 'wb') as f:
        #         pickle.dump(_lc_runwise, f, pickle.HIGHEST_PROTOCOL)
        # else:
        #     with open(pkl_path, "rb") as f:
        #         _lc_runwise = pickle.load(f)
        _lightcurve = _lc_runwise.to_table(sed_type="flux", format="lightcurve")
        
        lcs_runwise.append(_lc_runwise)
        lightcurves.append(_lightcurve)
    
    def weighted_average(table, sys_error=0):
        val = table["flux"]
        uncertainty = np.sqrt((sys_error * table["flux"])**2 + table["flux_err"]**2)
        return (val/uncertainty**2).sum() / (1/uncertainty**2).sum(), np.sqrt(1/np.sum(1/uncertainty**2))
    
    def calculate_chi2_pvalue(table, sys_error=0):
        uncertainty = np.sqrt((sys_error * table["flux"])**2 + table["flux_err"]**2)
        flux = table["flux"]
        mean_flux = (flux/uncertainty**2).sum() / (1/uncertainty**2).sum()
        mean_flux_err = np.sqrt(1/np.sum(1/uncertainty**2))
        print(f"Weighted mean flux: {mean_flux:.3e} +/- {mean_flux_err:.3e} cm-2 s-1")
        
        chi2_value = np.sum((table["flux"] - mean_flux)**2/uncertainty**2)
        ndf = len(table["flux"]) - 1
        pvalue = chi2.sf(x=chi2_value, df=ndf)
        print(f"Chi2: {chi2_value:.1f}, ndf: {ndf}, P-value: {pvalue:.2e}")
        return chi2_value, ndf, pvalue
    
    mean_flux = []
    mean_flux_err = []
    chi2_val, pvalue, ndf = [], [], []
    for lightcurve, label in zip(lightcurves, dataset_labels):
        _mean_flux, _mean_flux_err = weighted_average(lightcurve)
        print(f"\nFor {label} analysis")
        _chi2_val, _ndf, _pvalue = calculate_chi2_pvalue(lightcurve, sys_error=0.0)
        mean_flux.append(_mean_flux)
        mean_flux_err.append(_mean_flux_err)
        chi2_val.append(_chi2_val)
        pvalue.append(_pvalue)
        ndf.append(_ndf)
    
    time_min, time_max = [], []
    delta_time, time_center = [], []
    flux, flux_stat_err = [], []
    run_num = []
    for observation, lightcurve in zip(observations, lightcurves):
        _time_min = Time(np.hstack(lightcurve["time_min"]), format='mjd').datetime
        _time_max = Time(np.hstack(lightcurve["time_max"]), format='mjd').datetime
        time_min.append(_time_min)
        time_max.append(_time_max)
        delta_time.append(_time_max - _time_min)
        time_center.append(_time_min + (_time_max - _time_min) / 2)
        # Flux and flux error
        flux.append(np.hstack(lightcurve["flux"]))
        flux_stat_err.append(np.hstack(lightcurve["flux_err"]))
        # run numbers
        run_num.append([int(n) for n in observation.ids])
    
    def calculate_chi2_pvalue_array(flux, flux_err, sys_error=0):
        uncertainty = np.sqrt((sys_error * flux)**2 + flux_err**2)
        mean_flux = (flux/uncertainty**2).sum() / (1/uncertainty**2).sum()
        mean_flux_err = np.sqrt(1/np.sum(1/uncertainty**2))
        
        chi2_value = np.sum((flux - mean_flux)**2/uncertainty**2)
        ndf = len(flux) - 1
        pvalue = chi2.sf(x=chi2_value, df=ndf)
        return chi2_value, ndf, pvalue
    
    Niter = 300
    sys = np.linspace(0, 0.12, Niter)
    chi2_sys, pvalue_sys, sigma_sys = [], [], []
    
    for i in range(len(flux)):
        _chi2_sys, _pvalue_sys, _sigma_sys = [], [], []
        for s in sys:
            _chi2_value, _ndf, _pvalue = calculate_chi2_pvalue_array(flux[i], flux_stat_err[i], sys_error=s)
        
            _chi2_sys.append(_chi2_value)
            _pvalue_sys.append(_pvalue)
            _sigma_sys.append(norm.ppf(1 - _pvalue))
        chi2_sys.append(_chi2_sys)
        pvalue_sys.append(_pvalue_sys)
        sigma_sys.append(_sigma_sys)
    
    sys_error = []
    for i in range(len(flux)):
        # Calculating the systematic error and the difference
        for j in range(len(sys)-1):
            if pvalue_sys[i][j] < 0.5 and pvalue_sys[i][j+1] >= 0.5:
                flag = j
        sys_error.append(sys[flag])
        print(f"Systematic uncertainty ({dataset_labels[i]}) = {sys[flag]*100:.2f}%")

    dict_total = {
        "run_nums" : run_num,
        "time" : time_min,
        "flux" : flux,
        "flux_err" : flux_stat_err,
        "integrating_energy" : limit,
        "sys_error" : sys_error,
    }
    
    
    # Saving the object
    fname_dict = root + f"objects/tests_flux_dispersion/result_energy_{limit:.4f}TeV.pkl"
    with open(fname_dict, 'wb') as f:
        pickle.dump(dict_total, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    input_str = sys.argv[1]
    main(input_str)

