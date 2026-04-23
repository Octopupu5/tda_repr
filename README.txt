This archive is intended to be unpacked into a checked-out tda-repr source tree:

<repo-root>/saved_runs/
  figures_runs/exp_*/...

Unpack the archive so folder names match exactly.

To rebuild the reproduction artifact tree from the archived runs:

    ./reproduction_cases.sh

It produces:
- reproduction/diploma_pictures/ (RU figures)
- reproduction/latex_pictures/ (EN Figure_*.png rendered from saved_runs/figures_runs)
- reproduction/tables/layer_candidates_*.tex (three case-study tables)

Run folders referenced by the case-study pipeline:

- exp_20260419_160324_cv_imagenette_resnet18_ft-full
- exp_20260418_211916_nlp_smol-summarize_smollm2-135m_ft-full
- exp_20260505_033618_cv_bloodmnist_efficientnet_b0_ft-full
- exp_20260408_114511_nlp_trec6_distilbert_ft-full
- exp_20260403_042415_nlp_trec6_distilbert_ft-full
- exp_20260404_182533_cv_imagenette_efficientnet_b0_ft-full
- exp_20260505_194724_cv_bloodmnist_convnext_tiny_ft-full
