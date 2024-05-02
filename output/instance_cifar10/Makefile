no_regen:
	nvcc --std=c++20 output.cu main.cpp -I ../include/ -o out
	scp out compute@colin.compute.srv:/srv/compute/out

build:
	../../cmake-build-debug/pkmlc cifar10.pkml
	nvcc --std=c++20 output.cu main.cpp -I ../include/ -o out -O3
	scp out compute@colin.compute.srv:/srv/compute/out

build_debug:
	../../cmake-build-debug/pkmlc cifar10.pkml
	nvcc --std=c++20 output.cu main.cpp -I ../include/ -o out -g -G
	scp out compute@colin.compute.srv:/srv/compute/out

run: build
	ssh compute@colin.compute.srv "cd /srv/compute && ./out"

debug: build_debug
	ssh compute@colin.compute.srv "cd /srv/compute && ./out"