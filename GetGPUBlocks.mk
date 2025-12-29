all: $(EXECUTABLE)

# Define the temporary executable name
PROBE_EXE = ./gpu_block_probe

# Define where your config file lives
CONFIG_FILE = $(SRC_DIR)/config.h

# ---------------------------------------------------------
# LOGIC: Check the GPU_TUNING variable from the main Makefile
# ---------------------------------------------------------
ifeq ($(BLOCK_TUNING),1)

# === OPTION A: Tuning is ENABLED (1) ===
configure_gpu_blocks:
	@echo " [CONFIG] Checking GPU specifications..."
	
	@# 1. Compile the probe
	@$(NVCC) $(SRC_DIR)/query_blocks.cu -o $(PROBE_EXE) -w
	
	@# 2. Run logic strictly inside the shell
	@DETECTED_BLOCKS=$$($(PROBE_EXE)); \
	if [ -z "$$DETECTED_BLOCKS" ]; then \
		echo " [ERROR] Failed to query GPU. Leaving config.h unchanged."; \
	else \
		if grep -q "#define N_BLOCKS $$DETECTED_BLOCKS" $(CONFIG_FILE); then \
			echo " [CONFIG] N_BLOCKS is already set to $$DETECTED_BLOCKS. Skipping update."; \
		else \
			echo " [CONFIG] Updating N_BLOCKS to $$DETECTED_BLOCKS in $(CONFIG_FILE)..."; \
			sed -i "s/#define N_BLOCKS .*/#define N_BLOCKS $$DETECTED_BLOCKS/" $(CONFIG_FILE); \
		fi; \
	fi
	
	@# 3. Clean up
	@rm -f $(PROBE_EXE)

else

# === OPTION B: Tuning is DISABLED (0) ===
configure_gpu_blocks:
	@echo " [CONFIG] GPU Tuning is disabled (GPU_TUNING=0). Using existing value for number of blocks found in config.h."

endif