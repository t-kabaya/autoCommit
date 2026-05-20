.PHONY: help install download-model ft-gemma2b ft-test clean disk-usage

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make download-model  - Download Gemma 2B model to models folder"
	@echo "  make ft-gemma2b      - Fine-tune Gemma 2B (full training)"
	@echo "  make ft-test         - Fine-tune Gemma 2B (quick test with 100 samples)"
	@echo "  make disk-usage      - Check disk usage of project and models"
	@echo "  make clean           - Clean up generated files"

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

download-model:
	. venv/bin/activate && python download_model.py --model_name google/gemma-2-2b-it --output_dir models/gemma-2b

ft-gemma2b:
	. venv/bin/activate && cd FT && python finetune_gemma2b.py \
		--model_name google/gemma-2-2b-it \
		--output_dir ../models/gemma-2b-commitpack-ft \
		--num_samples 1000 \
		--num_epochs 3 \
		--batch_size 4 \
		--learning_rate 2e-4

ft-test:
	. venv/bin/activate && cd FT && python finetune_gemma2b.py \
		--model_name google/gemma-2-2b-it \
		--output_dir ../models/gemma-2b-commitpack-ft-test \
		--num_samples 100 \
		--num_epochs 1 \
		--batch_size 2 \
		--learning_rate 2e-4

disk-usage:
	@echo "=== System Disk Space ==="
	@df -h .
	@echo ""
	@echo "=== Project Disk Usage ==="
	@du -sh .
	@echo ""
	@echo "=== Models Directory Usage ==="
	@if [ -d "models" ]; then \
		du -h --max-depth=2 models | sort -h; \
	else \
		echo "models directory does not exist"; \
	fi
	@echo ""
	@echo "=== Virtual Environment Usage ==="
	@if [ -d "venv" ]; then \
		du -sh venv; \
	else \
		echo "venv directory does not exist"; \
	fi

clean:
	rm -rf models/gemma-2b-commitpack-ft*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
