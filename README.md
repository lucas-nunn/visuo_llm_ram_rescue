## what

- fork of: https://github.com/adriendoerig/visuo_llm.git
- exploratory replication-ish
- run fully on [this PC](https://pcpartpicker.com/list/4KpcH3)
- mostly dealing with RAM bottleneck!

## results

### searchlight correlations

- Pearson correlation of model RDM vs brain RDM for each voxel
- voxel RDMs calculated in searchlight manner (6 voxel radius sphere centered on each voxel)
- see [paper](https://www.nature.com/articles/s42256-025-01072-0#Sec7) for details

**subject 1, 20 sessions, 25 sampled 100x100 RDMs:**

mpnet validation
![all-mpnet-base-v2 vs human brain](lucas_exploration/figures/all-mpnet-base-v2_subj01_20_sessions.png)

custom mini beta VAE trained on MS COCO 2014 train set
![beta VAE vs human brain](lucas_exploration/figures/simplebetavae_beta4_z32_seed0_64px_subj01.png)

pretrained stable diffusion VAE
![stable diffusion VAE vs human brain](lucas_exploration/figures/sdvae_ft_mse_latents_subj01.png)

pixel space of NSD images
![pixel space vs human brain](lucas_exploration/figures/pixels_rgb_64px_subj01_20_sessions.png)

**subject 2, 20 sessions, 25 sampled 100x100 RDMs:**

mpnet validation
![all-mpnet-base-v2 vs human brain](lucas_exploration/figures/all-mpnet-base-v2_subj02_20_sessions.png)

pixel space of NSD images
![pixel space vs human brain](lucas_exploration/figures/pixels_rgb_64px_subj02_20_sessions.png)

## Citation

```bibtex
@article{doerig2024visualrepresentationshumanbrain,
      title={Visual representations in the human brain are aligned with large language models},
      author={Adrien Doerig and Tim C Kietzmann and Emily Allen and Yihan Wu and Thomas Naselaris and Kendrick Kay and Ian Charest},
      year={2024},
      eprint={2209.11737},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2209.11737},
}
```
