# Map dataset names to script parameters
DATASET_cifar10 := cifar10
DATASET_svhn := svhn
DATASET_all := cifar10_svhn

.PHONY: precompute figure-cdf-all-measures


# Produce the same figures as in the paper
paper:
	@echo "Building paper figures:"

	@echo "... Figure 1"
	@make figure-cdf-all-measures dataset=all min_weight=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp__ds_cifar10_svhn__mw_12.000000_cdf_per_hp.pdf figure_1.pdf

	@echo "... Figure 8"
	@make figure-cdf-all-measures dataset=svhn min_weight=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp__ds_svhn__mw_12.000000_cdf_per_hp.pdf figure_8.pdf

	@echo "... Figure 10"
	@make figure-cdf-all-measures dataset=cifar10 min_weight=12 filter_noise=True
	@mv figure__signerror_cdf_per_hp__ds_cifar10__mw_12.000000_cdf_per_hp.pdf figure_10.pdf


# Precompute the environments to speed up plotting
precompute: .require_dataset .require_filter_noise
	$(eval filter_noise_ := $(shell tr '[:upper:]' '[:lower:]' <<< $(filter_noise)))
	@test ! -f environment_cache/precomputations__filternoise$(filter_noise_)__$(DATASET_$(dataset)).pkl && \
		echo "Precomputation file not found. Precomputing environments..." && \
		python3 create_environments.py $(DATASET_$(dataset)) $(tau) $(filter_noise_) \
	|| echo "Precomputation file found."


# Main figure: CDF of all environment sign errors per HP
figure-cdf-all-measures: precompute
figure-cdf-all-measures: .require_dataset .require_min_weight
	python3 figure_cdf_all_measures.py $(DATASET_$(dataset)) $(min_weight) true


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