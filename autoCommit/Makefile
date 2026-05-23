.PHONY: help install download-model ft-gemma2b ft-gemma2b-better ft-test test-gemma2-output clean disk-usage check-memory

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make download-model  - Download Gemma 2B model to models folder"
	@echo "  make ft-gemma2b      - Fine-tune Gemma 2B (1000 samples, 3 epochs)"
	@echo "  make ft-gemma2b-better - Fine-tune Gemma 2B (5000 samples, 3 epochs) - RECOMMENDED"
	@echo "  make ft-test         - Fine-tune Gemma 2B (quick test: 100 samples, 1 epoch)"
	@echo "  make test-gemma2-output - Test fine-tuned Gemma 2B model with 3 sample inputs"
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

# Quick test with minimal samples (for testing only - not recommended for actual use)
ft-test:
	. venv/bin/activate && cd FT && python finetune_gemma2b.py \
		--model_name google/gemma-2-2b-it \
		--output_dir ../models/gemma-2b-commitpack-ft-test \
		--num_samples 100 \
		--num_epochs 1 \
		--batch_size 2 \
		--learning_rate 2e-4

# Better training configuration with more samples
ft-gemma2b-better:
	. venv/bin/activate && cd FT && python finetune_gemma2b.py \
		--model_name google/gemma-2-2b-it \
		--output_dir ../models/gemma-2b-commitpack-ft-better \
		--num_samples 5000 \
		--num_epochs 3 \
		--batch_size 2 \
		--learning_rate 2e-4

# Test fine-tuned Gemma 2B model with 3 sample diffs and output commit messages
# Input 1: New feature (authentication check)
# Input 2: Bug fix (null check for price)
# Input 3: Refactoring (SQL injection prevention)
test-gemma2-output:
	@echo "=== Testing Fine-tuned Gemma 2B Model Output ==="
	@echo ""
	@echo "Input 1: Add new feature (authentication check)"
	@echo "Diff:"
	@echo "  +    if user.is_authenticated():"
	@echo "  +        return dashboard()"
	@echo "  +    else:"
	@echo "  +        return login_page()"
	@echo ""
	@echo "Expected:"
	@echo "  feat: Add authentication check to main function"
	@echo ""
	@echo "Actual Output:"
	@. venv/bin/activate && python -c "import warnings; warnings.filterwarnings('ignore'); import os; os.environ['TRANSFORMERS_VERBOSITY']='error'; from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; \
		model = AutoModelForCausalLM.from_pretrained('./models/gemma-2b-commitpack-ft-test', local_files_only=True, device_map='auto', torch_dtype=torch.float16); \
		tokenizer = AutoTokenizer.from_pretrained('./models/gemma-2b-commitpack-ft-test', local_files_only=True); \
		diff = '''diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ @@\n-\n+    if user.is_authenticated():\n+        return dashboard()\n+    else:\n+        return login_page()\n'''; \
		inputs = tokenizer(diff, return_tensors='pt').to(model.device); \
		outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7); \
		result = tokenizer.decode(outputs[0], skip_special_tokens=True); \
		lines = result.replace(diff, '').strip().split('\\n'); \
		result = lines[0] if lines else ''; \
		print('  ' + result)" 2>/dev/null
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Input 2: Fix bug (null check for price)"
	@echo "Diff:"
	@echo "  -    return sum(item.price for item in items)"
	@echo "  +    return sum(item.price for item in items if item.price is not None)"
	@echo ""
	@echo "Expected:"
	@echo "  fix: Add null check for item price in calculate_total"
	@echo ""
	@echo "Actual Output:"
	@. venv/bin/activate && python -c "import warnings; warnings.filterwarnings('ignore'); import os; os.environ['TRANSFORMERS_VERBOSITY']='error'; from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; \
		model = AutoModelForCausalLM.from_pretrained('./models/gemma-2b-commitpack-ft-test', local_files_only=True, device_map='auto', torch_dtype=torch.float16); \
		tokenizer = AutoTokenizer.from_pretrained('./models/gemma-2b-commitpack-ft-test', local_files_only=True); \
		diff = '''diff --git a/utils.py b/utils.py\n--- a/utils.py\n+++ b/utils.py\n@@ @@\n-    return sum(item.price for item in items)\n+    return sum(item.price for item in items if item.price is not None)\n'''; \
		inputs = tokenizer(diff, return_tensors='pt').to(model.device); \
		outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7); \
		result = tokenizer.decode(outputs[0], skip_special_tokens=True); \
		lines = result.replace(diff, '').strip().split('\\n'); \
		result = lines[0] if lines else ''; \
		print('  ' + result)" 2>/dev/null
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Input 3: Refactor code (SQL injection prevention)"
	@echo "Diff:"
	@echo "  -    def get_user(self, id):"
	@echo "  -        return self.query(f\"SELECT * FROM users WHERE id={id}\")"
	@echo "  +    def get_user(self, user_id):"
	@echo "  +        return self.query(\"SELECT * FROM users WHERE id=?\", (user_id,))"
	@echo ""
	@echo "Expected:"
	@echo "  refactor: Use parameterized query to prevent SQL injection in get_user"
	@echo ""
	@echo "Actual Output:"
	@. venv/bin/activate && python -c "import warnings; warnings.filterwarnings('ignore'); import os; os.environ['TRANSFORMERS_VERBOSITY']='error'; from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; \
		model = AutoModelForCausalLM.from_pretrained('./models/gemma-2b-commitpack-ft-test', local_files_only=True, device_map='auto', torch_dtype=torch.float16); \
		tokenizer = AutoTokenizer.from_pretrained('./models/gemma-2b-commitpack-ft-test', local_files_only=True); \
		diff = '''diff --git a/db.py b/db.py\n--- a/db.py\n+++ b/db.py\n@@ @@\n-    def get_user(self, id):\n-        return self.query(f\"SELECT * FROM users WHERE id={id}\")\n+    def get_user(self, user_id):\n+        return self.query(\"SELECT * FROM users WHERE id=?\", (user_id,))\n'''; \
		inputs = tokenizer(diff, return_tensors='pt').to(model.device); \
		outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7); \
		result = tokenizer.decode(outputs[0], skip_special_tokens=True); \
		lines = result.replace(diff, '').strip().split('\\n'); \
		result = lines[0] if lines else ''; \
		print('  ' + result)" 2>/dev/null
	@echo ""
	@echo "=== Test Complete ==="

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
