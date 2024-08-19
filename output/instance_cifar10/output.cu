#include <fstream>
#include "output.hpp"
#include "modules/fully_connected.pkmlm/fully_connected.hpp"
#include "modules/sigmoid.pkmlm/sigmoid.hpp"

using l0_size=PKML::Dimension<32,32,3>;
using l0_1_size=PKML::Dimension<40,40>;
using l1_2_size=PKML::Dimension<40,40>;
using l2_3_size=PKML::Dimension<40,40>;
using l3_4_size=PKML::Dimension<40,40>;
using l4_5_size=PKML::Dimension<20,20>;
using l5_6_size=PKML::Dimension<20,20>;
using l6_7_size=PKML::Dimension<10>;
using l7_size=PKML::Dimension<10>;

using l0_t=FullyConnected<l0_size,l0_1_size,FullyConnected_params{.learning_rate=0.001,}>;
using l1_t=Sigmoid<l0_1_size,l1_2_size,Sigmoid_params{}>;
using l2_t=FullyConnected<l1_2_size,l2_3_size,FullyConnected_params{.learning_rate=0.001,}>;
using l3_t=Sigmoid<l2_3_size,l3_4_size,Sigmoid_params{}>;
using l4_t=FullyConnected<l3_4_size,l4_5_size,FullyConnected_params{.learning_rate=0.001,}>;
using l5_t=Sigmoid<l4_5_size,l5_6_size,Sigmoid_params{}>;
using l6_t=FullyConnected<l5_6_size,l6_7_size,FullyConnected_params{.learning_rate=0.001,}>;
using l7_t=Sigmoid<l6_7_size,l7_size,Sigmoid_params{}>;

PKML::float_t * buf0;
PKML::float_t * buf0_1;
PKML::float_t * buf1_2;
PKML::float_t * buf2_3;
PKML::float_t * buf3_4;
PKML::float_t * buf4_5;
PKML::float_t * buf5_6;
PKML::float_t * buf6_7;
PKML::float_t * buf7;

PKML::float_t * l0_alloc;
PKML::float_t * l1_alloc;
PKML::float_t * l2_alloc;
PKML::float_t * l3_alloc;
PKML::float_t * l4_alloc;
PKML::float_t * l5_alloc;
PKML::float_t * l6_alloc;
PKML::float_t * l7_alloc;

static constexpr PKML::float_t *& input_buffer = buf0;
static constexpr PKML::float_t *& output_buffer = buf7;

InstanceCifar10::Network::Network() {
cudaMalloc((void **) &buf0, l0_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf0_1, l0_1_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf1_2, l1_2_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf2_3, l2_3_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf3_4, l3_4_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf4_5, l4_5_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf5_6, l5_6_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf6_7, l6_7_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf7, l7_size::element_product * sizeof(PKML::float_t));

if constexpr (l0_t::memory_requirement > 0) cudaMalloc((void **) &l0_alloc, l0_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l1_t::memory_requirement > 0) cudaMalloc((void **) &l1_alloc, l1_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l2_t::memory_requirement > 0) cudaMalloc((void **) &l2_alloc, l2_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l3_t::memory_requirement > 0) cudaMalloc((void **) &l3_alloc, l3_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l4_t::memory_requirement > 0) cudaMalloc((void **) &l4_alloc, l4_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l5_t::memory_requirement > 0) cudaMalloc((void **) &l5_alloc, l5_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l6_t::memory_requirement > 0) cudaMalloc((void **) &l6_alloc, l6_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l7_t::memory_requirement > 0) cudaMalloc((void **) &l7_alloc, l7_t::memory_requirement * sizeof(PKML::float_t));

l0_t::init(l0_alloc);
l1_t::init(l1_alloc);
l2_t::init(l2_alloc);
l3_t::init(l3_alloc);
l4_t::init(l4_alloc);
l5_t::init(l5_alloc);
l6_t::init(l6_alloc);
l7_t::init(l7_alloc);
}

InstanceCifar10::Network::~Network() {
cudaFree(buf0);
cudaFree(buf0_1);
cudaFree(buf1_2);
cudaFree(buf2_3);
cudaFree(buf3_4);
cudaFree(buf4_5);
cudaFree(buf5_6);
cudaFree(buf6_7);
cudaFree(buf7);

if constexpr (l0_t::memory_requirement > 0) cudaFree(l0_alloc);
if constexpr (l1_t::memory_requirement > 0) cudaFree(l1_alloc);
if constexpr (l2_t::memory_requirement > 0) cudaFree(l2_alloc);
if constexpr (l3_t::memory_requirement > 0) cudaFree(l3_alloc);
if constexpr (l4_t::memory_requirement > 0) cudaFree(l4_alloc);
if constexpr (l5_t::memory_requirement > 0) cudaFree(l5_alloc);
if constexpr (l6_t::memory_requirement > 0) cudaFree(l6_alloc);
if constexpr (l7_t::memory_requirement > 0) cudaFree(l7_alloc);
}

template<std::size_t total_threads>
__device__ void forward_d(
const std::size_t thread_index,
PKML::float_t * const _buf0,
PKML::float_t * const _buf0_1,
PKML::float_t * const _buf1_2,
PKML::float_t * const _buf2_3,
PKML::float_t * const _buf3_4,
PKML::float_t * const _buf4_5,
PKML::float_t * const _buf5_6,
PKML::float_t * const _buf6_7,
PKML::float_t * const _buf7,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
PKML::float_t * const _l6_alloc,
PKML::float_t * const _l7_alloc) {
PKML::thread_gate<total_threads, l1_2_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
_buf0_1[thread_index] = l0_t::forward_gated(thread_index, _buf0, _l0_alloc);
_buf1_2[thread_index] = l1_t::forward_ungated(_buf0_1[thread_index], _l1_alloc);
});
__syncthreads();
PKML::thread_gate<total_threads, l3_4_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
_buf2_3[thread_index] = l2_t::forward_gated(thread_index, _buf1_2, _l2_alloc);
_buf3_4[thread_index] = l3_t::forward_ungated(_buf2_3[thread_index], _l3_alloc);
});
__syncthreads();
PKML::thread_gate<total_threads, l5_6_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
_buf4_5[thread_index] = l4_t::forward_gated(thread_index, _buf3_4, _l4_alloc);
_buf5_6[thread_index] = l5_t::forward_ungated(_buf4_5[thread_index], _l5_alloc);
});
__syncthreads();
PKML::thread_gate<total_threads, l7_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
_buf6_7[thread_index] = l6_t::forward_gated(thread_index, _buf5_6, _l6_alloc);
_buf7[thread_index] = l7_t::forward_ungated(_buf6_7[thread_index], _l7_alloc);
});
__syncthreads();
}

template<std::size_t total_threads>
__device__ void propogate_d(
const std::size_t thread_index,
PKML::float_t * const _buf0,
PKML::float_t * const _buf0_1,
PKML::float_t * const _buf1_2,
PKML::float_t * const _buf2_3,
PKML::float_t * const _buf3_4,
PKML::float_t * const _buf4_5,
PKML::float_t * const _buf5_6,
PKML::float_t * const _buf6_7,
PKML::float_t * const _buf7,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
PKML::float_t * const _l6_alloc,
PKML::float_t * const _l7_alloc,
const PKML::float_t * const correct
) {
forward_d<total_threads>(
thread_index,
_buf0,
_buf0_1,
_buf1_2,
_buf2_3,
_buf3_4,
_buf4_5,
_buf5_6,
_buf6_7,
_buf7,
_l0_alloc,
_l1_alloc,
_l2_alloc,
_l3_alloc,
_l4_alloc,
_l5_alloc,
_l6_alloc,
_l7_alloc);
__shared__ PKML::float_t intermediate_costs[total_threads];
PKML::float_t * intermediate_costs_ptr = intermediate_costs;
PKML::thread_gate<total_threads, l7_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, correct, _buf7]() {
intermediate_costs_ptr[thread_index] = PKML::Math::sub(_buf7[thread_index], correct[thread_index]);
});
PKML::thread_gate<total_threads, l7_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
PKML::float_t intermediate = intermediate_costs_ptr[thread_index];
intermediate = PKML::Math::mul(intermediate, l7_t::backward_ungated(_buf6_7[thread_index], _buf7[thread_index], _l7_alloc));
l6_t::backward_gated(thread_index, intermediate_costs_ptr, _buf5_6, intermediate, _l6_alloc);
});
__syncthreads();
PKML::thread_gate<total_threads, l5_6_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
PKML::float_t intermediate = intermediate_costs_ptr[thread_index];
intermediate = PKML::Math::mul(intermediate, l5_t::backward_ungated(_buf4_5[thread_index], _buf5_6[thread_index], _l5_alloc));
l4_t::backward_gated(thread_index, intermediate_costs_ptr, _buf3_4, intermediate, _l4_alloc);
});
__syncthreads();
PKML::thread_gate<total_threads, l3_4_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
PKML::float_t intermediate = intermediate_costs_ptr[thread_index];
intermediate = PKML::Math::mul(intermediate, l3_t::backward_ungated(_buf2_3[thread_index], _buf3_4[thread_index], _l3_alloc));
l2_t::backward_gated(thread_index, intermediate_costs_ptr, _buf1_2, intermediate, _l2_alloc);
});
__syncthreads();
PKML::thread_gate<total_threads, l1_2_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5_6, _buf6_7, _buf7, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc, _l6_alloc, _l7_alloc]() {
PKML::float_t intermediate = intermediate_costs_ptr[thread_index];
intermediate = PKML::Math::mul(intermediate, l1_t::backward_ungated(_buf0_1[thread_index], _buf1_2[thread_index], _l1_alloc));
l0_t::backward_gated(thread_index, intermediate_costs_ptr, _buf0, intermediate, _l0_alloc);
});
__syncthreads();
}

template<std::size_t total_threads>
__global__ void forward_k(
PKML::float_t * const _buf0,
PKML::float_t * const _buf0_1,
PKML::float_t * const _buf1_2,
PKML::float_t * const _buf2_3,
PKML::float_t * const _buf3_4,
PKML::float_t * const _buf4_5,
PKML::float_t * const _buf5_6,
PKML::float_t * const _buf6_7,
PKML::float_t * const _buf7,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
PKML::float_t * const _l6_alloc,
PKML::float_t * const _l7_alloc) {
const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
forward_d<total_threads>(
thread_index,
_buf0,
_buf0_1,
_buf1_2,
_buf2_3,
_buf3_4,
_buf4_5,
_buf5_6,
_buf6_7,
_buf7,
_l0_alloc,
_l1_alloc,
_l2_alloc,
_l3_alloc,
_l4_alloc,
_l5_alloc,
_l6_alloc,
_l7_alloc);
}

void InstanceCifar10::Network::forward() {
forward_k<6 * 512><<<6, 512>>>(
buf0,
buf0_1,
buf1_2,
buf2_3,
buf3_4,
buf4_5,
buf5_6,
buf6_7,
buf7,
l0_alloc,
l1_alloc,
l2_alloc,
l3_alloc,
l4_alloc,
l5_alloc,
l6_alloc,
l7_alloc);
}

template<std::size_t total_threads>
__global__ void train_k(
std::size_t iterations,
std::size_t mult,
PKML::float_t * const _buf0,
PKML::float_t * const _buf0_1,
PKML::float_t * const _buf1_2,
PKML::float_t * const _buf2_3,
PKML::float_t * const _buf3_4,
PKML::float_t * const _buf4_5,
PKML::float_t * const _buf5_6,
PKML::float_t * const _buf6_7,
PKML::float_t * const _buf7,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
PKML::float_t * const _l6_alloc,
PKML::float_t * const _l7_alloc,
const InstanceCifar10::Dataset::TrainingSet * const dataset,
const std::size_t dataset_size
) {
const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
for (std::size_t i = 0; i < iterations; i++) {
const InstanceCifar10::Dataset::TrainingSet & training_set = dataset[(i * mult) % dataset_size];
PKML::thread_gate<total_threads, l0_size::element_product>(thread_index, [thread_index, training_set, _buf0]() {
_buf0[thread_index] = training_set.inputs[thread_index]; // possible optimization by moving _buf0 pointer
});
__syncthreads();
propogate_d<total_threads>(
thread_index,
_buf0,
_buf0_1,
_buf1_2,
_buf2_3,
_buf3_4,
_buf4_5,
_buf5_6,
_buf6_7,
_buf7,
_l0_alloc,
_l1_alloc,
_l2_alloc,
_l3_alloc,
_l4_alloc,
_l5_alloc,
_l6_alloc,
_l7_alloc,
training_set.outputs
);
}
}

void InstanceCifar10::Network::train(std::size_t iterations, std::size_t mult, InstanceCifar10::Dataset & dataset) {
train_k<6 * 512><<<6, 512>>>(
iterations,
mult,
buf0,
buf0_1,
buf1_2,
buf2_3,
buf3_4,
buf4_5,
buf5_6,
buf6_7,
buf7,
l0_alloc,
l1_alloc,
l2_alloc,
l3_alloc,
l4_alloc,
l5_alloc,
l6_alloc,
l7_alloc,
dataset._data,
dataset.size()
);
cudaDeviceSynchronize();
}

template<std::size_t total_threads>
__global__ void evaluate_k(
PKML::float_t * const _buf0,
PKML::float_t * const _buf0_1,
PKML::float_t * const _buf1_2,
PKML::float_t * const _buf2_3,
PKML::float_t * const _buf3_4,
PKML::float_t * const _buf4_5,
PKML::float_t * const _buf5_6,
PKML::float_t * const _buf6_7,
PKML::float_t * const _buf7,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
PKML::float_t * const _l6_alloc,
PKML::float_t * const _l7_alloc,
const InstanceCifar10::Dataset::TrainingSet * const dataset,
const std::size_t dataset_size,
PKML::float_t * cost_ptr
) {
const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
PKML::float_t cost = 0;
for (std::size_t i = 0; i < dataset_size; i++) {
const InstanceCifar10::Dataset::TrainingSet & training_set = dataset[i];
PKML::thread_gate<total_threads, l0_size::element_product>(thread_index, [thread_index, training_set, _buf0]() {
_buf0[thread_index] = training_set.inputs[thread_index]; // possible optimization by moving _buf0 pointer
});
__syncthreads();
forward_d<total_threads>(
thread_index,
_buf0,
_buf0_1,
_buf1_2,
_buf2_3,
_buf3_4,
_buf4_5,
_buf5_6,
_buf6_7,
_buf7,
_l0_alloc,
_l1_alloc,
_l2_alloc,
_l3_alloc,
_l4_alloc,
_l5_alloc,
_l6_alloc,
_l7_alloc);
PKML::thread_gate<total_threads, l6_7_size::element_product>(thread_index, [thread_index, &cost, dataset_size, training_set, _buf7]() {
cost += (training_set.outputs[thread_index] - _buf7[thread_index]) / dataset_size;
});
}
*cost_ptr = cost;
}

PKML::float_t InstanceCifar10::Network::evaluate(InstanceCifar10::Dataset & dataset) {
PKML::float_t * dev_cost;
cudaMalloc((void **) &dev_cost, sizeof(PKML::float_t));
evaluate_k<6 * 512><<<6, 512>>>(
buf0,
buf0_1,
buf1_2,
buf2_3,
buf3_4,
buf4_5,
buf5_6,
buf6_7,
buf7,
l0_alloc,
l1_alloc,
l2_alloc,
l3_alloc,
l4_alloc,
l5_alloc,
l6_alloc,
l7_alloc,
dataset._data,
dataset.size(),
dev_cost
);
PKML::float_t cost;
cudaMemcpy(&cost, dev_cost, sizeof(float), cudaMemcpyDeviceToHost);
return cost;
}

void InstanceCifar10::Network::save(const char * path) {
std::ofstream fs(path);
if constexpr (l0_t::memory_requirement != 0) {
float * temp = new float[l0_t::memory_requirement];
cudaMemcpy(temp, l0_alloc, l0_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l0_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l1_t::memory_requirement != 0) {
float * temp = new float[l1_t::memory_requirement];
cudaMemcpy(temp, l1_alloc, l1_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l1_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l2_t::memory_requirement != 0) {
float * temp = new float[l2_t::memory_requirement];
cudaMemcpy(temp, l2_alloc, l2_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l2_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l3_t::memory_requirement != 0) {
float * temp = new float[l3_t::memory_requirement];
cudaMemcpy(temp, l3_alloc, l3_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l3_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l4_t::memory_requirement != 0) {
float * temp = new float[l4_t::memory_requirement];
cudaMemcpy(temp, l4_alloc, l4_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l4_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l5_t::memory_requirement != 0) {
float * temp = new float[l5_t::memory_requirement];
cudaMemcpy(temp, l5_alloc, l5_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l5_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l6_t::memory_requirement != 0) {
float * temp = new float[l6_t::memory_requirement];
cudaMemcpy(temp, l6_alloc, l6_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l6_t::memory_requirement * sizeof(float));
delete[] temp;
}
if constexpr (l7_t::memory_requirement != 0) {
float * temp = new float[l7_t::memory_requirement];
cudaMemcpy(temp, l7_alloc, l7_t::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);
fs.write((const char *) temp, l7_t::memory_requirement * sizeof(float));
delete[] temp;
}
}

void InstanceCifar10::Network::load(const char * path) {
std::ifstream fs(path);
if constexpr (l0_t::memory_requirement != 0) {
float * temp = new float[l0_t::memory_requirement];
fs.read((char *) temp, l0_t::memory_requirement * sizeof(float));
cudaMemcpy(l0_alloc, temp, l0_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l1_t::memory_requirement != 0) {
float * temp = new float[l1_t::memory_requirement];
fs.read((char *) temp, l1_t::memory_requirement * sizeof(float));
cudaMemcpy(l1_alloc, temp, l1_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l2_t::memory_requirement != 0) {
float * temp = new float[l2_t::memory_requirement];
fs.read((char *) temp, l2_t::memory_requirement * sizeof(float));
cudaMemcpy(l2_alloc, temp, l2_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l3_t::memory_requirement != 0) {
float * temp = new float[l3_t::memory_requirement];
fs.read((char *) temp, l3_t::memory_requirement * sizeof(float));
cudaMemcpy(l3_alloc, temp, l3_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l4_t::memory_requirement != 0) {
float * temp = new float[l4_t::memory_requirement];
fs.read((char *) temp, l4_t::memory_requirement * sizeof(float));
cudaMemcpy(l4_alloc, temp, l4_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l5_t::memory_requirement != 0) {
float * temp = new float[l5_t::memory_requirement];
fs.read((char *) temp, l5_t::memory_requirement * sizeof(float));
cudaMemcpy(l5_alloc, temp, l5_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l6_t::memory_requirement != 0) {
float * temp = new float[l6_t::memory_requirement];
fs.read((char *) temp, l6_t::memory_requirement * sizeof(float));
cudaMemcpy(l6_alloc, temp, l6_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
if constexpr (l7_t::memory_requirement != 0) {
float * temp = new float[l7_t::memory_requirement];
fs.read((char *) temp, l7_t::memory_requirement * sizeof(float));
cudaMemcpy(l7_alloc, temp, l7_t::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);
delete[] temp;
}
}

void InstanceCifar10::Network::copy_outputs(PKML::float_t * dst) {
cudaMemcpy(dst, output_buffer, l7_size::element_product * sizeof(PKML::float_t), cudaMemcpyDeviceToHost);
}

void InstanceCifar10::Network::copy_inputs(PKML::float_t * src) {
cudaMemcpy(input_buffer, src, l0_size::element_product * sizeof(PKML::float_t), cudaMemcpyHostToDevice);
}

InstanceCifar10::Dataset::Dataset(): _size(0), _capacity(1) {
if (cudaMalloc((void **) &_data, _capacity * sizeof(TrainingSet)) == cudaErrorMemoryAllocation) throw std::bad_alloc();
}

InstanceCifar10::Dataset::~Dataset() {
cudaFree(_data);
}

void InstanceCifar10::Dataset::push_back(const TrainingSet & value) {
if (_size == _capacity) {
_capacity *= 2;
TrainingSet * temp_mem;
if (cudaMalloc((void **) &temp_mem, _capacity * sizeof(TrainingSet)) == cudaErrorMemoryAllocation) throw std::bad_alloc();
cudaMemcpy(temp_mem, _data, _size * sizeof(TrainingSet), cudaMemcpyDeviceToDevice);
cudaFree(_data);
_data = temp_mem;
}
cudaMemcpy(&_data[_size++], &value, sizeof(TrainingSet), cudaMemcpyHostToDevice);
}

void InstanceCifar10::Dataset::pull_set(std::size_t index, TrainingSet & set) const {
cudaMemcpy(&set, &_data[index], sizeof(TrainingSet), cudaMemcpyDeviceToHost);
}

