CU_INPUTS=output/instance_xor/output.cu
CPP_INPUTS=output/main.cpp

OUTPUT_PATH=output/out

COMPILER_FLAGS=-ccbin=/opt/cuda/bin -L /opt/cuda/lib/ -I output/include -arch sm_61 -std=c++20

.PHONY: default
default: run

.PHONY: build
build:
	nvcc $(CU_INPUTS) $(CPP_INPUTS) $(COMPILER_FLAGS) -o $(OUTPUT_PATH) -O3

.PHONY: build_debug
build_debug:
	nvcc $(CU_INPUTS) $(CPP_INPUTS) $(COMPILER_FLAGS) -o $(OUTPUT_PATH) -O1 -g

.PHONY: upload
upload: build
	scp $(OUTPUT_PATH) compute@colin.compute.srv:/srv/compute

.PHONY: upload_debug
upload_debug: build_debug
	scp $(OUTPUT_PATH) compute@colin.compute.srv:/srv/compute

.PHONY: run
run: upload
	ssh compute@colin.compute.srv "cd /srv/compute  && source /etc/profile && echo \"-- OUTPUT --\" && ./out"

.PHONY: debug
debug: upload_debug
	ssh compute@colin.compute.srv "cd /srv/compute && source /etc/profile && echo \"-- OUTPUT --\" && cuda-gdb ./out"

.PHONY: profile
profile: upload
	ssh compute@colin.compute.srv "cd /srv/compute && source /etc/profile && echo \"-- OUTPUT --\" && nvprof ./out"