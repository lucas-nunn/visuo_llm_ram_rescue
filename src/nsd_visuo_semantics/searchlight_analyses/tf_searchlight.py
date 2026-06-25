import numpy as np
import tensorflow as tf

from nsd_visuo_semantics.utils.tf_utils import chunking as ch
from nsd_visuo_semantics.utils.tf_utils import compute_rdm_batch
from numpy.lib.format import open_memmap


def tf_searchlight(betas_sampled, indices, sorted_indices, batch_size=50):
    """[volumetric searchlight on GPU]

    Args:
        betas_sampled ([array]): volume of beta conditions
                                 (x, y, z, n_conditions).
        indices ([array]):       list of searchlight sphere indices
        sorted_indices ([type]): list of searchlight spheres,
                                 sorted by n_voxels.
        batch_size (int, opt):   how large a chunk to send to GPU.
                                 Defaults to 50.

    Returns:
        rdms [array]:
    """
    x, y, z, n_conditions = betas_sampled.shape

    # unroll the betas
    betas_unrolled = tf.convert_to_tensor(betas_sampled.reshape((-1, n_conditions)))

    # now chunk the betas given the sphere sizes and batch_size
    # and for each chunk, compute a batch of rdm computations.
    # start_time = time.time()
    rdms = []
    for i, ind in enumerate(sorted_indices):
        print(f'processing sphere {i} out of {len(sorted_indices)}')
        chunks = ch(ind, batch_size)

        for c, chunk in enumerate(chunks):
            # line below is from a previous version where we were not using sorted indices
            # here, we DO use them. i.e., we get a list of sorted_indices, where all
            # indices have the same n_voxels (this occurs because we remove NaN voxels).
            # we go over these lsits one by one, and compute their RDMs
            # indices (all_indices) is a Python list of per-sphere index arrays,
            # so fancy-index it via a comprehension; `indices[chunk]` would TypeError.
            # sl = tf.gather(betas_unrolled, np.stack(indices[chunk.astype(np.int32)]))
            these_indices = [indices[i] for i in chunk]
            sl = tf.gather(betas_unrolled, np.stack(these_indices))
            t = tf.transpose(sl, perm=[0, 2, 1])
            rdm = np.asarray(compute_rdm_batch(t))
            rdms.append(rdm)

    # elapsed_time = time.time() - start_time
    # print(
    #     'elapsedtime: ',
    #     f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    # )

    # return to CPU

    return np.vstack(rdms)


def tf_searchlight_corr(
    betas_sampled,
    indices,
    sorted_indices,
    model_rdms,
    batch_size=50,
):
    """Compute searchlight/model RDM correlations without storing brain RDMs.

    The legacy pipeline first materialises every searchlight RDM as a
    ``n_centers x n_condition_pairs`` array and only then correlates it with
    the model RDMs. For NSD this array is several GB, and ``np.vstack`` needs a
    second equally large allocation. Instead, normalise and correlate each
    searchlight batch while it is still a TensorFlow tensor. Only the small
    ``n_centers x n_models`` correlation matrix is returned to CPU memory.

    Rows are emitted in exactly the same sphere-size/chunk order as
    :func:`tf_searchlight`, so the existing ``rdms_sort`` mapping remains
    valid.
    """
    _, _, _, n_conditions = betas_sampled.shape

    betas_unrolled = tf.convert_to_tensor(
        betas_sampled.reshape((-1, n_conditions)),
        dtype=tf.float32,
    )

    model_rdms = tf.convert_to_tensor(
        np.asarray(model_rdms, dtype=np.float32),
        dtype=tf.float32,
    )
    model_rdms = model_rdms - tf.reduce_mean(
        model_rdms, axis=1, keepdims=True
    )
    model_rdms /= tf.sqrt(
        tf.einsum("ij,ij->i", model_rdms, model_rdms)
    )[:, None]

    correlations = []
    for i, ind in enumerate(sorted_indices):
        print(f"processing sphere {i} out of {len(sorted_indices)}")
        chunks = ch(ind, batch_size)

        for chunk in chunks:
            these_indices = [indices[j] for j in chunk]
            searchlights = tf.gather(
                betas_unrolled, np.stack(these_indices)
            )
            patterns = tf.transpose(searchlights, perm=[0, 2, 1])
            brain_rdms = compute_rdm_batch(patterns)

            brain_rdms = brain_rdms - tf.reduce_mean(
                brain_rdms, axis=1, keepdims=True
            )
            brain_rdms /= tf.sqrt(
                tf.einsum("ij,ij->i", brain_rdms, brain_rdms)
            )[:, None]
            batch_correlations = tf.einsum(
                "ik,jk->ij", brain_rdms, model_rdms
            )
            correlations.append(np.asarray(batch_correlations))

            del searchlights, patterns, brain_rdms, batch_correlations

    return np.vstack(correlations)
