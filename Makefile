export PYTHONPATH := $(shell pwd)

all: example_1_1


example_1_1:
	@echo "Running Experiments for example 1.1"
	@python3 examples/1_model_selection/1.1_algorithm_domain_alignment.py



.PHONY: *
