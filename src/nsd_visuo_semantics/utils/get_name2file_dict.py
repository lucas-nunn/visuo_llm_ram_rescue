'''Helper that returns a dictionary with the paths to the embeddings and DNN activities files'''

import glob
import os
import re


def _add_sdvae_embedding_files(modelname2file, saved_embeddings_dir):
    """Register Stable Diffusion VAE feature files created in lucas_exploration.

    Expected filenames are emitted by lucas_exploration/stable_diffusion_vae_embeddings.ipynb:
    nsd_sdvae_ft_mse_latents_256px.npy
    nsd_sdvae_ft_mse_pca256_256px.npy

    Model names with the image-size suffix are always added. For 256px files we also add
    short aliases, e.g. sdvae_ft_mse_pca256, mirroring the simple MPNET-style names.
    """
    sdvae_pattern = os.path.join(saved_embeddings_dir, "nsd_sdvae_ft_mse_*.npy")
    for file_path in sorted(glob.glob(sdvae_pattern)):
        file_name = os.path.basename(file_path)
        match = re.fullmatch(r"nsd_sdvae_ft_mse_(latents|pca\d+)_(\d+)px\.npy", file_name)
        if match is None:
            continue

        feature_name, image_size = match.groups()
        model_name = f"sdvae_ft_mse_{feature_name}"
        sized_model_name = f"{model_name}_{image_size}px"
        modelname2file[sized_model_name] = file_path
        if image_size == "256":
            modelname2file[model_name] = file_path


def _add_betavae_embedding_files(modelname2file, saved_embeddings_dir):
    """Register NSD beta-VAE feature files created in lucas_exploration.

    Expected full-run filenames are emitted by
    lucas_exploration/beta_vae_nsd.py:
    nsd_betavae_beta4_seed0_zmean_128px.npy

    Partial smoke-test exports include a rows suffix and are intentionally not
    registered for RDM construction.
    """
    betavae_pattern = os.path.join(
        saved_embeddings_dir, "nsd_betavae_beta*_seed*_zmean_*px.npy"
    )
    for file_path in sorted(glob.glob(betavae_pattern)):
        file_name = os.path.basename(file_path)
        match = re.fullmatch(
            r"nsd_(betavae_beta[^_]+_seed\d+_zmean)_(\d+)px\.npy",
            file_name,
        )
        if match is None:
            continue

        model_name, image_size = match.groups()
        sized_model_name = f"{model_name}_{image_size}px"
        modelname2file[sized_model_name] = file_path
        if image_size == "128":
            modelname2file[model_name] = file_path


def _add_simple_betavae_embedding_files(modelname2file, saved_embeddings_dir):
    """Register simple beta-VAE feature files created in lucas_exploration.

    Expected filenames are emitted by
    lucas_exploration/betavae_nsd_embeddings.py, e.g.:
    nsd_simplebetavae_beta4_z32_seed0_64px.npy

    These come from the small paper-faithful beta-VAE in lucas_exploration/VAE.py
    (64px, sigmoid/BCE, [0, 1] inputs) and are kept distinct from the 128px
    betavae_*_zmean family above. Both the size-suffixed name and a short alias
    are registered. Partial smoke-test exports carry a rows suffix and are
    intentionally not matched.
    """
    pattern = os.path.join(
        saved_embeddings_dir, "nsd_simplebetavae_beta*_z*_seed*_*px.npy"
    )
    for file_path in sorted(glob.glob(pattern)):
        file_name = os.path.basename(file_path)
        match = re.fullmatch(
            r"nsd_(simplebetavae_beta[^_]+_z\d+_seed\d+)_(\d+)px\.npy",
            file_name,
        )
        if match is None:
            continue
        model_name, image_size = match.groups()
        modelname2file[f"{model_name}_{image_size}px"] = file_path
        modelname2file[model_name] = file_path


def _add_pixel_embedding_files(modelname2file, saved_embeddings_dir):
    """Register raw-pixel feature files created in lucas_exploration.

    Expected filenames are emitted by
    lucas_exploration/pixel_nsd_embeddings.py, e.g.:
    nsd_pixels_rgb_64px.npy
    nsd_pixels_gray_128px.npy

    These are the flattened-pixel baseline for the searchlight (low-level image
    statistics, no model). Both the size-suffixed name and a short alias are
    registered. Partial smoke-test exports carry a rows suffix and are
    intentionally not matched.
    """
    pattern = os.path.join(saved_embeddings_dir, "nsd_pixels_*px.npy")
    for file_path in sorted(glob.glob(pattern)):
        file_name = os.path.basename(file_path)
        match = re.fullmatch(
            r"nsd_(pixels_(?:rgb|gray)_\d+px)\.npy",
            file_name,
        )
        if match is None:
            continue
        sized_model_name = match.group(1)
        modelname2file[sized_model_name] = file_path
        # also register a colorspace-only alias (e.g. pixels_rgb), pointing at the
        # last-sorted size when several resolutions exist.
        modelname2file[sized_model_name.rsplit("_", 1)[0]] = file_path


def get_name2file_dict(saved_embeddings_dir, saved_dnn_activities_dir,
                       ecoset_saved_dnn_activities_dir):

    # specify where each set of nsd embeddings is saved
    modelname2file = {
        # basic models
        "mpnet": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings.pkl",
        "multihot": f"{saved_embeddings_dir}/nsd_multihot.pkl",
        "fasttext_categories": f"{saved_embeddings_dir}/nsd_fasttext_CATEGORY_mean_embeddings.pkl",
        "fasttext_nouns": f"{saved_embeddings_dir}/nsd_fasttext_NOUNS_embeddings.pkl",
        "fasttext_verbs": f"{saved_embeddings_dir}/nsd_fasttext_VERBS_embeddings.pkl",
        "fasttext_all": f"{saved_embeddings_dir}/nsd_fasttext_ALLWORDS_embeddings.pkl",
        "glove_categories": f"{saved_embeddings_dir}/nsd_glove_CATEGORY_mean_embeddings.pkl",
        "glove_nouns": f"{saved_embeddings_dir}/nsd_glove_NOUNS_embeddings.pkl",
        "glove_verbs": f"{saved_embeddings_dir}/nsd_glove_VERBS_embeddings.pkl",
        "glove_all": f"{saved_embeddings_dir}/nsd_glove_ALLWORDS_embeddings.pkl",
        "CLIP_ViT_text": f"{saved_embeddings_dir}/nsd_CLIP-vit_mean_embeddings.pkl",
        "CLIP_ViT_images": f"{saved_dnn_activities_dir}/CLIP-vit_nsd_image_features.pkl",
        "CLIP_RN50_text": f"{saved_embeddings_dir}/nsd_CLIP-rn50_mean_embeddings.pkl",
        "CLIP_RN50_images": f"{saved_dnn_activities_dir}/CLIP-rn50_nsd_image_features.pkl",
        "thingsvision_cornet-s": f"{saved_dnn_activities_dir}/thingsvision_cornet-s_nsd_image_features.pkl",
        "brainscore_alexnet": f"{saved_dnn_activities_dir}/brainscore_alexnet_nsd_image_features.pkl",
        "brainscore_resnet50_julios": f"{saved_dnn_activities_dir}/brainscore_resnet50_julios_nsd_image_features.pkl",
        "resnext101_32x8d_wsl": f"{saved_dnn_activities_dir}/resnext101_32x8d_wsl_nsd_image_features.pkl",
        "google_simclrv1_rn50": f"{saved_dnn_activities_dir}/google_simclrv1_rn50_nsd_image_features.pkl",
        "timm_nf_resnet50": f"{saved_dnn_activities_dir}/timm_nf_resnet50_nsd_image_features.pkl",
        "konkle_alexnetgn_ipcl_ref01": f"{saved_dnn_activities_dir}/konkle_alexnetgn_ipcl_ref01_nsd_image_features.pkl",  # these are with inputs in [0,255] before the transform (I was not sure which to use)
        "konkle_alexnetgn_supervised_ref12_augset1_5x": f"{saved_dnn_activities_dir}/konkle_alexnetgn_supervised_ref12_augset1_5x_nsd_image_features.pkl",
        "mpnetWordAvg_all": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_ALLWORDS_embeddings.pkl",
        "taskonomy_scenecat_resnet50": f"{saved_dnn_activities_dir}/taskonomy_scenecat_resnet50_nsd_image_features.pkl",
        "guse": f"{saved_embeddings_dir}/nsd_guse_mean_embeddings.pkl",
        "all-mpnet-base-v2": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings.pkl",  # this is a duplicate of the line above, both names work
        "mpnet_resnet50_finalLayer": f"{saved_dnn_activities_dir}/mpnet_resnet50_finalLayer_nsd_image_features.pkl",
        "multihot_resnet50_finalLayer": f"{saved_dnn_activities_dir}/multihot_resnet50_finalLayer_nsd_image_features.pkl",
        "sceneCateg_resnet50_finalLayer": f"{saved_dnn_activities_dir}/sceneCateg_resnet50_finalLayer_nsd_image_features.pkl",
        'mpnet_scrambled': f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings_scrambled.pkl",
        "sdvae_ft_mse_latents": f"{saved_embeddings_dir}/nsd_sdvae_ft_mse_latents_256px.npy",
        "sdvae_ft_mse_latents_256px": f"{saved_embeddings_dir}/nsd_sdvae_ft_mse_latents_256px.npy",
        "sdvae_ft_mse_pca256": f"{saved_embeddings_dir}/nsd_sdvae_ft_mse_pca256_256px.npy",
        "sdvae_ft_mse_pca256_256px": f"{saved_embeddings_dir}/nsd_sdvae_ft_mse_pca256_256px.npy",

        # DNNs trained on ecoset activities
        "dnn_ecoset_category": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_category_post_gn_epoch80.h5",
        "dnn_ecoset_fasttext": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_fasttext_post_gn_epoch80.h5",
    }

    # DNN activities
    for epoch in [0, 100, 200]:
        for modelname in ["multihot_rec", "mpnet_rec"]: 
            modelname2file[f"dnn_{modelname}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_nsd_activations_epoch{epoch}.h5"
            for seed in range(1,11):
                modelname2file[f"dnn_{modelname}_seed{seed}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_seed{seed}_nsd_activations_epoch{epoch}.h5"

    # word types embeddings
    WORD_TYPES = ['nouns', 'verbs', 'noun', 'verb', 'prepositions', 'adjectives', 'adverbs']

    # sentence embeddings on (lists of) words
    for mpnet_moniker in ["mpnet", "all-mpnet-base-v2"]:
        mpnet_full_name = "all-mpnet-base-v2"
        modelname2file[f"{mpnet_moniker}_category_all"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_CATEGORY_concatString_mean_embeddings_allCats.pkl"
        for wt in WORD_TYPES:
            modelname2file[f'{mpnet_moniker}_{wt}'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_{wt.upper()}_embeddings.pkl"

    SENTENCE_EMBEDDING_MODEL_TYPES = ['multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
                                      'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2', 
                                      'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2',
                                      'GUSE_transformer', 'GUSE_DAN', 'USE_CMLM_Base', 'T5']
    for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
        modelname2file[SENTENCE_EMBEDDING_MODEL_TYPE] = f"{saved_embeddings_dir}/nsd_{SENTENCE_EMBEDDING_MODEL_TYPE}_mean_embeddings.pkl"

    _add_sdvae_embedding_files(modelname2file, saved_embeddings_dir)
    _add_betavae_embedding_files(modelname2file, saved_embeddings_dir)
    _add_simple_betavae_embedding_files(modelname2file, saved_embeddings_dir)
    _add_pixel_embedding_files(modelname2file, saved_embeddings_dir)

    return modelname2file
