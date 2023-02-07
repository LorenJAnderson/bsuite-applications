export PYTHONPATH := $(shell pwd)

all: example_1_1 example_5_1 example_5_2


example_1_1:
	@echo "Running Experiments for example 1.1"
	@python3 examples/1_model_selection/1.1_algorithm_domain_alignment.py

example_5_1:
	@echo "Running Experiments for example 5.1"
	@python3 examples/5_model_alteration/5.1_ppo_v_ppornn.py

example_5_2:
	@echo "Running Experiments for example 5.1"
	@python3 examples/5_model_alteration/5.2_dqn_v_qrdqn.py

.PHONY: *
