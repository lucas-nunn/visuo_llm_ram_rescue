import pickle
import numpy as np

from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import get_model_rdms
from nsd_visuo_semantics.utils.tf_utils import corr_rdms, sort_spheres
from nsd_visuo_semantics.searchlight_analyses.tf_searchlight import tf_searchlight as tfs



model_rdms, model_names = get_model_rdms(models_dir, subj, filt=MODEL_NAME)  # (filt should be a wildcard to catch correct model rdms, careful not to catch other models)
batchg = BatchGen(model_rdms, all_conditions)

saved_samples_file = 'lucas_exploration/results/searchlight_respectedsampling_correlation/subj01/saved_sampling/subj01_nsd-allsubstim_sampling.npy'
subj_sample_pool = np.load(saved_samples_file, allow_pickle=True)

for choices in subj_sample_pool:
    model_rdms_sample = np.asarray(batchg.index_rdms(choices))

brain_maps = corr_rdms(brain_sl_rdms_sample, model_rdms_sample)
