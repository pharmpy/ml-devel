# Retrieve input/training data for one model.
# This will be used by the prediction code in Pharmpy and should
# also be used to retrieve data in the training/development process


import pharmpy.modeling as modeling
import numpy as np
import pandas as pd


def create_dataset(model):
    idcol = model.dataset.pharmpy.id_label
    nids = modeling.get_number_of_individuals(model)
    nobs = modeling.get_number_of_observations(model)
    nobsi = modeling.get_number_of_observations_per_individual(model)
    cwres = model.modelfit_results.residuals['CWRES']

    # Max ratio of abs(ETAi) and omegai
    variance_omegas = model.random_variables.etas.variance_parameters
    omega_estimates = model.modelfit_results.parameter_estimates[variance_omegas]
    abs_ebes = model.modelfit_results.individual_estimates.abs()
    ebe_ratio = abs_ebes / list(omega_estimates)
    max_ebe_ratio = ebe_ratio.max(axis=1)

    # exp(OFVi / nobsi) / exp(OFV / nobs)
    iofv = model.modelfit_results.individual_ofv
    ofv_ratio = np.exp(iofv / nobsi) / np.exp(model.modelfit_results.ofv / nobs)

    # mean(ETA / OMEGA)
    cov = model.modelfit_results.individual_estimates_covariance
    etc_diag = pd.DataFrame([np.diag(y) for y in cov], columns=cov.iloc[0].columns)
    etc_ratio = etc_diag / list(omega_estimates)
    mean_etc_ratio = etc_ratio.mean(axis=1)
    mean_etc_ratio.index = ofv_ratio.index

    # max((abs(indcov - mean(cov))) / sd(cov))
    cov_names = [str(x) for x in model.covariates]
    covariates = model.dataset[cov_names + [idcol]].set_index(idcol)
    mean_covs = covariates.groupby(idcol).mean()
    cov_ratio = (abs(mean_covs - mean_covs.mean()) / mean_covs.std()).mean(axis=1)

    df = pd.DataFrame({
        'nids': nids,
        'nobs': nobs,
        'nobs_subj': nobsi / (nobs / nids),
        'ncovs': len(model.covariates),
        'cov_ratio': cov_ratio,
        'max_cwres': abs(cwres).groupby(idcol).max(),
        'median_cwres': abs(cwres).groupby(idcol).median(),
        'max_ebe_ratio': max_ebe_ratio,
        'ofv_ratio' : ofv_ratio,
        'etc_ratio': mean_etc_ratio,
        })
    return df

model = read_model("/home/rikard/testing/pheno_real.mod")
data = create_dataset(model)