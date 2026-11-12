NVCC ?= nvcc
ARCH ?= sm_75
STD ?= c++17
FLAGS ?= -O3 -std=$(STD) --use_fast_math -Xcompiler -Wall,-Wextra
BUILD := build

all: $(BUILD)/rmsnorm

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/rmsnorm: rmsnorm.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) $(FLAGS) -o $@ $<

check: $(BUILD)/rmsnorm
	$(BUILD)/rmsnorm --x ../tests/golden/raw/rmsnorm_fp32_4096x512/x.f32.bin \
		--w ../tests/golden/raw/rmsnorm_fp32_4096x512/w.f32.bin \
		--y ../tests/golden/raw/rmsnorm_fp32_4096x512/y.f32.bin \
		--rows 4096 --cols 512 --eps 1e-06 --reps 200 --warmup 20 --key rmsnorm_fp32_4096x512

clean:
	rm -rf $(BUILD)

.PHONY: all check clean
