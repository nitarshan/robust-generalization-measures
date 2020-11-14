# Map dataset names to script parameters
DATASET_cifar10 := cifar10
DATASET_svhn := svhn
DATASET_all := cifar10_svhn


# Produce the same figures as in the paper
paper:
	@echo "Building paper figures:"

	@echo "... Figure 1"
	@make figure-cdf-all-measures dataset=all min_ess=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp__ds_cifar10_svhn__mess_12.000000__filternoise_True_cdf_per_hp.pdf figure_1.pdf

	@echo "... Figure 2"
	@make figure-triangle-cdf dataset=all measure=complexity.pacbayes_mag_flatness hp=hp.model_depth min_ess=12 filter_noise=True
	@mv figure_triangle_cdf__ds_cifar10_svhn__mess_12.000000__gm_pacbayes.mag.flatness__filternoise_True_hp_hp.model_depth.pdf figure_2.pdf

	@echo "... Figure 3"
	@make figure-supp-envs-vs-miness dataset=all
	@mv figure_miness_remaining_environments_ds_cifar10_svhn_range_0_100.pdf figure_3.pdf

	@echo "... Figure 4"
	@make figure-supp-monte-carlo-ablation dataset=all min_ess=12  
	@ls figure_monte_carlo_noise_ablation__*__ds_cifar10_svhn__mess_12.000000_cdf.pdf | cut -d'_' -f7 | xargs -I{} mv figure_monte_carlo_noise_ablation__{}__ds_cifar10_svhn__mess_12.000000_cdf.pdf figure_4_{}.pdf

	@echo "... Figure 8"
	@make figure-cdf-all-measures dataset=svhn min_ess=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp__ds_svhn__mess_12.000000__filternoise_True_cdf_per_hp.pdf figure_8.pdf

	@echo "... Figure 9a"
	@make figure-triangle-cdf dataset=svhn measure=complexity.pacbayes_orig hp=hp.model_depth min_ess=12 filter_noise=True
	@mv figure_triangle_cdf__ds_svhn__mess_12.000000__gm_pacbayes.orig__filternoise_True_hp_hp.model_depth.pdf figure_9a.pdf

	@echo "... Figure 9b"
	@make figure-triangle-cdf dataset=svhn measure=complexity.pacbayes_orig hp=hp.model_width min_ess=12 filter_noise=True
	@mv figure_triangle_cdf__ds_svhn__mess_12.000000__gm_pacbayes.orig__filternoise_True_hp_hp.model_width.pdf figure_9b.pdf

	@echo "... Figure 10"
	@make figure-cdf-all-measures dataset=cifar10 min_ess=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp__ds_cifar10__mess_12.000000__filternoise_True_cdf_per_hp.pdf figure_10.pdf

	@echo "... Figure 11"
	@make figure-supp-cdf-all-measures-easy dataset=all min_ess=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp_easy_envs__ds_cifar10_svhn__mess_12.000000__filternoise_True_cdf_per_hp.pdf figure_11.pdf


# Precompute the environments to speed up plotting
precompute: .require_dataset .require_filter_noise
	$(eval filter_noise_ := $(shell tr '[:upper:]' '[:lower:]' <<< $(filter_noise)))
	@test ! -f environment_cache/precomputations__filternoise$(filter_noise_)__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found. Precomputing environments..." && \
		python3 create_environments.py $(DATASET_$(dataset)) $(filter_noise_) \
	|| echo "Precomputation file found."


# Main text figure: CDF of all environment sign errors per HP
figure-cdf-all-measures: precompute
figure-cdf-all-measures: .require_dataset .require_min_ess .require_filter_noise
	python3 figure_cdf_all_measures.py $(DATASET_$(dataset)) $(min_ess) $(filter_noise)


# Main text figure: CDF of environment sign errors for pairs of HP values
figure-triangle-cdf: precompute
figure-triangle-cdf: .require_dataset .require_measure .require_hp .require_min_ess .require_filter_noise
	python3 figure_triangle_cdf.py $(DATASET_$(dataset)) $(measure) $(hp) $(min_ess) $(filter_noise)


# Supplementary figure: CDF of all environment sign errors for easier environments
figure-supp-cdf-all-measures-easy: precompute
figure-supp-cdf-all-measures-easy: .require_dataset .require_min_ess .require_filter_noise
	python supp_figure_easier_envs.py $(DATASET_$(dataset)) $(min_ess) $(filter_noise)


# Supplementary figure: number of environments remaining at various min effective sample sizes
figure-supp-envs-vs-miness: lower = 0
figure-supp-envs-vs-miness: upper=100
figure-supp-envs-vs-miness: .require_dataset
	python supp_figure_envs_remaining.py $(DATASET_$(dataset)) $(lower) $(upper)


# Supplementary figure: ablation study of Monte Carlo noise filtering
figure-supp-monte-carlo-ablation: .require_dataset .require_min_ess
	@test ! -f environment_cache/precomputations__filternoisetrue__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found (with noise filtering). Precomputing..." && \
		python3 supp_figure_monte_carlo_ablation.py $(DATASET_$(dataset)) true \
	|| echo "Precomputation file found (with noise filtering)."
	@test ! -f environment_cache/precomputations__filternoisefalse__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found (without noise filtering). Precomputing..." && \
		python3 create_environments.py $(DATASET_$(dataset)) false \
	|| echo "Precomputation file found (without noise filtering)."
	python supp_figure_monte_carlo_ablation.py $(DATASET_$(dataset))  $(min_ess)


.require_dataset:
ifndef dataset
	$(error dataset parameter is required. Add dataset=[cifar10, svhn, all] to your command.)
endif


.require_hp:
ifndef hp
	$(error hp parameter is required. The hyperparameter to consider. Add hp=[hp.lr, hp.model_width, hp.model_depth] to your command.)
endif


.require_measure:
ifndef measure
	$(error measure parameter is required. The complexity measure for which to make the plot. Add measure=<value> to your command.)
endif


.require_min_ess:
ifndef min_ess
	$(error min_ess parameter is required. This is the minimum effective sample size in an environment for it to be considered in the analysis. Add min_ess=<value> to your command.)
endif


.require_filter_noise:
ifndef filter_noise
	$(error filter_noise parameter is required. This enables/disables Monte Carlo noise filtering. It should be enabled unless you are running some kind of ablation study. Add filter_noise=[True|False] to your command.)
endif
