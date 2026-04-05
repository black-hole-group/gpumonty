# /*
#  * GPUmonty - GetGPUBlocks.mk
#  * Copyright (C) 2026 Pedro Naethe Motta
#  *
#  * This program is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
#  */
all: $(EXECUTABLE)


PROBE_EXE = ./gpu_block_probe
CONFIG_FILE = $(SRC_DIR)/config.h

# Don't run the GPU check if the user is just running 'make clean'
ifneq ($(MAKECMDGOALS),clean)

ifeq ($(BLOCK_TUNING),1)
$(info [CONFIG] Checking GPU specifications...)

# We use $(shell) to execute the probe BEFORE Make calculates dependencies.
# This ensures that if config.h is updated, its timestamp changes in time
# for Make to trigger a full recompile of all dependent .o files.
_GPU_SETUP := $(shell \
    $(NVCC) -Wno-deprecated-gpu-targets $(SRC_DIR)/query_blocks.cu -o $(PROBE_EXE) -w; \
    DETECTED_BLOCKS=$$($(PROBE_EXE)); \
    if [ -z "$$DETECTED_BLOCKS" ]; then \
        echo " [ERROR] Failed to query GPU. Leaving config.h unchanged."; \
    else \
        if grep -q "\#define N_BLOCKS $$DETECTED_BLOCKS" $(CONFIG_FILE); then \
            echo " [CONFIG] N_BLOCKS is already set to $$DETECTED_BLOCKS. Skipping update."; \
        else \
            echo " [CONFIG] Updating N_BLOCKS to $$DETECTED_BLOCKS in $(CONFIG_FILE)..."; \
            sed -i "s/\#define N_BLOCKS .*/\#define N_BLOCKS $$DETECTED_BLOCKS/" $(CONFIG_FILE); \
        fi; \
    fi; \
    rm -f $(PROBE_EXE) \
)

$(info $(_GPU_SETUP))

else
$(info [CONFIG] GPU Tuning is disabled (BLOCK_TUNING=0). Using existing config.h.)
endif

endif 

