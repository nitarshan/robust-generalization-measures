# Map dataset names to script parameters
DATASET_cifar10 := cifar10
DATASET_svhn := svhn
DATASET_all := cifar10_svhn

.PHONY: precompute

# Precompute the environments to speed up plotting
precompute: .require_dataset .require_filter_noise
	$(eval filter_noise_ := $(shell tr '[:upper:]' '[:lower:]' <<< $(filter_noise)))
	@test ! -f environment_cache/precomputations__filternoise$(filter_noise_)__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found. Precomputing environments..." && \
		python3 create_environments.py $(DATASET_$(dataset)) $(tau) $(filter_noise_) \
	|| echo "Precomputation file found."

# Main paper Figure 1: CDF of all environment sign errors per HP
figure-cdf-all-measures: create-environments
figure-cdf-all-measures: .require_dataset .require_tau .require_min_weight
	python3 figure_cdf_all_measures.py $(DATASET_$(dataset)) $(tau) $(min_weight) true

# Main paper Figure 2: CDF of environment sign errors for pairs of HP values
figure-cdf-all-measures: create-environments
figure-triangle-cdf: .require_dataset .require_measure .require_hp .require_tau .require_min_weight
	python3 figure_triangle_cdf.py $(DATASET_$(dataset)) $(measure) $(hp) $(tau) $(min_weight) true

# Supplementary Figure 3: number of environments remaining at various min effective sample sizes
figure-supp-envs-vs-miness: lower = 0
figure-supp-envs-vs-miness: upper=100
figure-supp-envs-vs-miness: .require_dataset
	python supp_figure_envs_remaining.py $(DATASET_$(dataset)) $(lower) $(upper) true
	# TODO: adapt to noise filter

# Supplementary Figure 9: CDF of all environment sign errors for easier environments
figure-supp-cdf-all-measures-easy: create-environments
	python supp_figure_easier_envs.py $(DATASET_$(dataset)) $(tau) $(min_weight)
	# TODO: adapt to noise filter

# Ablation study of Monte Carlo noise filtering
monte-carlo-ablation: .require_dataset .require_tau .require_min_weight
	@test ! -f environment_cache/precomputations__tau$(tau)__filternoisetrue__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found (with noise filtering). Precomputing..." && \
		python3 create_environments.py $(DATASET_$(dataset)) $(tau) true \
	|| echo "Precomputation file found (with noise filtering)."
	@test ! -f environment_cache/precomputations__tau$(tau)__filternoisefalse__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found (without noise filtering). Precomputing..." && \
		python3 create_environments.py $(DATASET_$(dataset)) $(tau) false \
	|| echo "Precomputation file found (without noise filtering)."
	python fig_monte_carlo_noise_ablation.py $(DATASET_$(dataset)) $(tau) $(min_weight)

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

.require_min_weight:
ifndef min_weight
	$(error min_weight parameter is required. This is the minimum sum of Hoeffding weights in an environment for it to be considered in the analysis. Add min_weight=<value> to your command.)
endif

.require_filter_noise:
ifndef filter_noise
	$(error filter_noise parameter is required. This enables/disables Monte Carlo noise filtering. It should be enabled unless you are running some kind of ablation study. Add filter_noise=[True|False] to your command.)
endif