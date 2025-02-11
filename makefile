# Compilers
CC  = gcc
CCU = nvcc
ASM = nasm

# Compiler flags
CFLAGS    = -std=c99 -lm
CUFLAGS   = -std=c++11
ASMFLAGS  = -f elf64

# Directories
C_DIR   = C_SOLUTIONS
CU_DIR  = CUDA_SOLUTIONS
ASM_DIR = ASM_SOLUTIONS
EXEC_DIR = exec

# Source files
CSRC   = $(wildcard $(C_DIR)/*.c)
ASMSRC = $(wildcard $(ASM_DIR)/*.asm)
CUSRC  = $(wildcard $(CU_DIR)/*.cu)

# Executable targets
C_EXES   = $(patsubst $(C_DIR)/%.c, $(EXEC_DIR)/%, $(CSRC))
ASM_EXES = $(patsubst $(ASM_DIR)/%.asm, $(EXEC_DIR)/%, $(ASMSRC))
CU_EXES  = $(patsubst $(CU_DIR)/%.cu, $(EXEC_DIR)/%, $(CUSRC))
EXES     = $(C_EXES) $(ASM_EXES) $(CU_EXES)

# Phony targets
.PHONY: all clean

# Main target
all: $(EXES)

# Directory creation
$(EXEC_DIR):
	@mkdir -p $@

# Compilation rules
$(EXEC_DIR)/%: $(C_DIR)/%.c | $(EXEC_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(EXEC_DIR)/%: $(ASM_DIR)/%.asm | $(EXEC_DIR)
	$(ASM) $(ASMFLAGS) $< -o $(@).o
	$(CC) $(@).o -o $@ -no-pie

$(EXEC_DIR)/%: $(CU_DIR)/%.cu | $(EXEC_DIR)
	$(CCU) $(CUFLAGS) -o $@ $<

# Clean
clean:
	rm -rf $(EXEC_DIR)
