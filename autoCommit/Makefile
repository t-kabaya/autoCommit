.PHONY: help install download-model ft-gemma2b ft-test clean disk-usage check-memory

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make download-model  - Download Gemma 2B model to models folder"
	@echo "  make ft-gemma2b      - Fine-tune Gemma 2B (full training)"
	@echo "  make ft-test         - Fine-tune Gemma 2B (quick test with 100 samples)"
	@echo "  make disk-usage      - Check disk usage of project and models"
	@echo "  make check-memory    - Check system memory and GPU memory for fine-tuning"
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

# 
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

check-memory:
	@echo "=== System RAM Memory ==="
	@free -h
	@echo ""
	@echo "=== Memory Usage Summary ==="
	@echo "Total RAM: $$(free -h | awk '/^Mem:/ {print $$2}')"
	@echo "Used RAM:  $$(free -h | awk '/^Mem:/ {print $$3}')"
	@echo "Free RAM:  $$(free -h | awk '/^Mem:/ {print $$4}')"
	@echo "Available: $$(free -h | awk '/^Mem:/ {print $$7}')"
	@echo ""
	@echo "=== GPU Memory (if available) ==="
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | \
		awk -F', ' '{printf "GPU %s: %s\n  Total: %s MB\n  Used:  %s MB\n  Free:  %s MB\n\n", $$1, $$2, $$3, $$4, $$5}'; \
	else \
		echo "No NVIDIA GPU detected (nvidia-smi not found)"; \
	fi
	@echo "=== Fine-Tuning Memory Requirements ==="
	@echo "Gemma 2B with 4-bit quantization (LoRA):"
	@echo "  - Minimum RAM:  16 GB (without GPU)"
	@echo "  - Recommended RAM: 32 GB"
	@echo "  - GPU VRAM (if using CUDA): 8-12 GB"
	@echo "  - Batch size 2: ~6-8 GB VRAM"
	@echo "  - Batch size 4: ~10-12 GB VRAM"
	@echo ""
	@echo "Current batch sizes in Makefile:"
	@echo "  - ft-gemma2b: batch_size=4 (requires ~10-12 GB VRAM)"
	@echo "  - ft-test:    batch_size=2 (requires ~6-8 GB VRAM)"
	@echo ""
	@echo "=== Diagnosis: Why Fine-Tuning Might Stop ==="
	@TOTAL_RAM=$$(free -h | awk '/^Mem:/ {print $$2}' | sed 's/Gi//'); \
	AVAIL_RAM=$$(free -h | awk '/^Mem:/ {print $$7}' | sed 's/Gi//'); \
	if command -v nvidia-smi >/dev/null 2>&1; then \
		GPU_VRAM=$$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); \
		GPU_VRAM_GB=$$(echo "scale=1; $$GPU_VRAM/1024" | bc); \
		echo "Current system: RAM=$${TOTAL_RAM}GB (Available=$${AVAIL_RAM}GB), GPU VRAM=$${GPU_VRAM_GB}GB free"; \
		if [ $$(echo "$$TOTAL_RAM < 16" | bc) -eq 1 ]; then \
			echo "⚠️  WARNING: RAM is less than 16GB - This is likely causing FT to stop!"; \
			echo "   Solution: Use 'make ft-test' with batch_size=2 or smaller"; \
		fi; \
		if [ $$(echo "$$GPU_VRAM < 6000" | bc) -eq 1 ]; then \
			echo "⚠️  WARNING: GPU VRAM is less than 6GB - Out of Memory error likely!"; \
			echo "   Solution: Reduce batch_size to 1 or use CPU-only mode"; \
		elif [ $$(echo "$$GPU_VRAM < 10000" | bc) -eq 1 ]; then \
			echo "⚠️  CAUTION: GPU VRAM is tight for batch_size=4"; \
			echo "   Recommendation: Use 'make ft-test' with batch_size=2"; \
		else \
			echo "✓ GPU VRAM is sufficient for batch_size=2-4"; \
		fi; \
	else \
		echo "No GPU detected - Running on CPU requires more RAM"; \
		if [ $$(echo "$$TOTAL_RAM < 32" | bc) -eq 1 ]; then \
			echo "⚠️  WARNING: CPU-only mode requires 32GB+ RAM"; \
			echo "   Current RAM may cause OOM errors"; \
		fi; \
	fi

clean:
	rm -rf models/gemma-2b-commitpack-ft*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
