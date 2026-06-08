import pickle
import numpy as np

from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import get_model_rdms, get_subject_conditions, get_masks
from nsd_visuo_semantics.utils.tf_utils import corr_rdms, sort_spheres, chunking
from nsd_visuo_semantics.searchlight_analyses.tf_searchlight import tf_searchlight as tfs


batch_size = 250
targetspace = "func1pt8mm"

sl_indices = 'lucas_exploration/results/searchlight/subj01/subj01-func1pt8mm-6rad-searchlight_indices.npy'
sl_centers = 'lucas_exploration/results/searchlight/subj01/subj01-func1pt8mm-6rad-searchlight_centers.npy'

with open(sl_indices, "rb") as fp:
    all_indices = pickle.load(fp)
    
with open(sl_centers, "rb") as fp:
    center_indices = pickle.load(fp)

sorted_indices = sort_spheres(all_indices)

mask = get_masks("data/NSD", "subj01", targetspace)
n_voxels = list(mask.shape)

betas_file = 'lucas_exploration/results/searchlight//betas/subj01_betas_average_func1pt8mm.npy'
betas = np.load(betas_file, allow_pickle=True)

saved_samples_file = 'lucas_exploration/results/searchlight_respectedsampling_correlation/subj01/saved_sampling/subj01_nsd-allsubstim_sampling.npy'
subj_sample_pool = np.load(saved_samples_file, allow_pickle=True)

conditions, conditions_sampled, subj_sample = get_subject_conditions("data/NSD", "subj01", 5, keep_only_3repeats=True)
subj_n_images = len(subj_sample)
all_conditions = range(subj_n_images)
subj_n_samples = int(subj_n_images // 100)
    
model_rdms, model_names = get_model_rdms("lucas_exploration/results/serialised_models_correlation/all-mpnet-base-v2", "subj01", filt="all-mpnet-base-v2")
batchg = BatchGen(model_rdms, all_conditions)

rdms_sort = []
for i, ind in enumerate(sorted_indices):
    chunks = chunking(ind, batch_size)
    for c, chunk in enumerate(chunks):
        rdms_sort.append(center_indices[chunk.astype(np.int32)])  # this is where the sorting mentioned above happens
rdms_sort = np.hstack(rdms_sort).astype(int)

for j, choices in enumerate(subj_sample_pool):
    betas_sampled = betas[:, :, :, choices]
    betas_sampled = betas_sampled.astype(np.float32)

    model_rdms_sample = np.asarray(batchg.index_rdms(choices))
    brain_sl_rdms_sample = tfs(betas_sampled, all_indices, sorted_indices, batch_size)

    del betas_sampled

    print("corr")
    brain_maps = corr_rdms(brain_sl_rdms_sample, model_rdms_sample)

    del brain_sl_rdms_sample

    print("here we go")
    brain_vols = []
    for map_i in range(len(model_rdms)):
        brain_map = brain_maps[:, map_i]
        brain_vect = np.zeros(np.prod(n_voxels))
        brain_vect[rdms_sort] = brain_map.squeeze()  # insert corr map in the right brain locations (each corr ends up in the right voxel)
        brain_vols.append(np.reshape(brain_vect, n_voxels))  # reshape to original xyz fmrivolume
    brain_vols = np.asarray(brain_vols)  # vols is plural because there may be more than 1 model. when using 1 model there is just one vol

    # save correlation vol for that sample
    np.save(f'subj01_nsd-all-mpnet-base-v2_func1pt8mm_sample-{j}.npy', brain_vols)
