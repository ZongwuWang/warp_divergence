all:
	nvcc -O0 warp_diverges.cu -o warp_diverges -arch=sm_80
.PHONY: clean

clean:
	@rm warp_diverges warp_diverges2
