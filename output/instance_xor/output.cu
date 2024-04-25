#include "output.hpp"
#include "modules/fully_connected.pkmlm/fully_connected.hpp"
#include "modules/sigmoid.pkmlm/sigmoid.hpp"

using l0_size=PKML::Dimension<2>;
using l0_1_size=PKML::Dimension<2>;
using l1_2_size=PKML::Dimension<2>;
using l2_3_size=PKML::Dimension<2>;
using l3_4_size=PKML::Dimension<2>;
using l4_5_size=PKML::Dimension<2>;
using l5_size=PKML::Dimension<2>;

using l0_t=FullyConnected<l0_size,l0_1_size,FullyConnected_params{.learning_rate=0.1,}>;
using l1_t=Sigmoid<l0_1_size,l1_2_size,Sigmoid_params{}>;
using l2_t=FullyConnected<l1_2_size,l2_3_size,FullyConnected_params{.learning_rate=0.1,}>;
using l3_t=Sigmoid<l2_3_size,l3_4_size,Sigmoid_params{}>;
using l4_t=FullyConnected<l3_4_size,l4_5_size,FullyConnected_params{.learning_rate=0.1,}>;
using l5_t=Sigmoid<l4_5_size,l5_size,Sigmoid_params{}>;

PKML::float_t * buf0;
PKML::float_t * buf0_1;
PKML::float_t * buf1_2;
PKML::float_t * buf2_3;
PKML::float_t * buf3_4;
PKML::float_t * buf4_5;
PKML::float_t * buf5;

PKML::float_t * l0_alloc;
PKML::float_t * l1_alloc;
PKML::float_t * l2_alloc;
PKML::float_t * l3_alloc;
PKML::float_t * l4_alloc;
PKML::float_t * l5_alloc;

static constexpr PKML::float_t *& input_buffer = buf0;
static constexpr PKML::float_t *& output_buffer = buf5;

InstanceXor::Network::Network() {
cudaMalloc((void **) &buf0, l0_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf0_1, l0_1_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf1_2, l1_2_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf2_3, l2_3_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf3_4, l3_4_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf4_5, l4_5_size::element_product * sizeof(PKML::float_t));
cudaMalloc((void **) &buf5, l5_size::element_product * sizeof(PKML::float_t));

if constexpr (l0_t::memory_requirement > 0) cudaMalloc((void **) &l0_alloc, l0_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l1_t::memory_requirement > 0) cudaMalloc((void **) &l1_alloc, l1_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l2_t::memory_requirement > 0) cudaMalloc((void **) &l2_alloc, l2_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l3_t::memory_requirement > 0) cudaMalloc((void **) &l3_alloc, l3_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l4_t::memory_requirement > 0) cudaMalloc((void **) &l4_alloc, l4_t::memory_requirement * sizeof(PKML::float_t));
if constexpr (l5_t::memory_requirement > 0) cudaMalloc((void **) &l5_alloc, l5_t::memory_requirement * sizeof(PKML::float_t));

l0_t::init(l0_alloc);
l1_t::init(l1_alloc);
l2_t::init(l2_alloc);
l3_t::init(l3_alloc);
l4_t::init(l4_alloc);
l5_t::init(l5_alloc);
}

InstanceXor::Network::~Network() {
cudaFree(buf0);
cudaFree(buf0_1);
cudaFree(buf1_2);
cudaFree(buf2_3);
cudaFree(buf3_4);
cudaFree(buf4_5);
cudaFree(buf5);

if constexpr (l0_t::memory_requirement > 0) cudaFree(l0_alloc);
if constexpr (l1_t::memory_requirement > 0) cudaFree(l1_alloc);
if constexpr (l2_t::memory_requirement > 0) cudaFree(l2_alloc);
if constexpr (l3_t::memory_requirement > 0) cudaFree(l3_alloc);
if constexpr (l4_t::memory_requirement > 0) cudaFree(l4_alloc);
if constexpr (l5_t::memory_requirement > 0) cudaFree(l5_alloc);
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
PKML::float_t * const _buf5,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
const PKML::float_t * const correct
) {
PKML::thread_gate<total_threads, l0_1_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc]() {
_buf0_1[thread_index] = l0_t::forward_gated(thread_index, _buf0, _l0_alloc);
_buf1_2[thread_index] = l1_t::forward_ungated(_buf0_1[thread_index], _l1_alloc);
});
PKML::thread_gate<total_threads, l2_3_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc]() {
_buf2_3[thread_index] = l2_t::forward_gated(thread_index, _buf1_2, _l2_alloc);
_buf3_4[thread_index] = l3_t::forward_ungated(_buf2_3[thread_index], _l3_alloc);
});
PKML::thread_gate<total_threads, l4_5_size::element_product>(thread_index, [thread_index, _buf0, _buf0_1, _buf1_2, _buf2_3, _buf3_4, _buf4_5, _buf5, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, _l4_alloc, _l5_alloc]() {
_buf4_5[thread_index] = l4_t::forward_gated(thread_index, _buf3_4, _l4_alloc);
_buf5[thread_index] = l5_t::forward_ungated(_buf4_5[thread_index], _l5_alloc);
});
}

template<std::size_t total_threads>
__global__ void train_k(
std::size_t iterations,
PKML::float_t * const _buf0,
PKML::float_t * const _buf0_1,
PKML::float_t * const _buf1_2,
PKML::float_t * const _buf2_3,
PKML::float_t * const _buf3_4,
PKML::float_t * const _buf4_5,
PKML::float_t * const _buf5,
PKML::float_t * const _l0_alloc,
PKML::float_t * const _l1_alloc,
PKML::float_t * const _l2_alloc,
PKML::float_t * const _l3_alloc,
PKML::float_t * const _l4_alloc,
PKML::float_t * const _l5_alloc,
const InstanceXor::Dataset::TrainingSet * const dataset,
const std::size_t dataset_size
) {
const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
for (std::size_t i = 0; i < iterations; i++) {
const InstanceXor::Dataset::TrainingSet & training_set = dataset[i % dataset_size];
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
_buf5,
_l0_alloc,
_l1_alloc,
_l2_alloc,
_l3_alloc,
_l4_alloc,
_l5_alloc,
training_set.outputs
);
}
}

void InstanceXor::Network::train(std::size_t iterations, InstanceXor::Dataset & dataset) {
train_k<1 * 100><<<1, 100>>>(
iterations,
buf0,
buf0_1,
buf1_2,
buf2_3,
buf3_4,
buf4_5,
buf5,
l0_alloc,
l1_alloc,
l2_alloc,
l3_alloc,
l4_alloc,
l5_alloc,
dataset._data,
dataset.size()
);
cudaDeviceSynchronize();
}

void InstanceXor::Network::copy_outputs(PKML::float_t * dst) {
cudaMemcpy(dst, output_buffer, 2 * sizeof(PKML::float_t), cudaMemcpyDeviceToHost);
}

void InstanceXor::Network::copy_inputs(PKML::float_t * src) {
cudaMemcpy(input_buffer, src, 2 * sizeof(PKML::float_t), cudaMemcpyHostToDevice);
}

InstanceXor::Dataset::Dataset(): _size(0), _capacity(1) {
if (cudaMalloc((void **) &_data, _capacity * sizeof(TrainingSet)) == cudaErrorMemoryAllocation) throw std::bad_alloc();
}

InstanceXor::Dataset::~Dataset() {
cudaFree(_data);
}

void InstanceXor::Dataset::push_back(const TrainingSet & value) {
if (_size == _capacity) {
_capacity *= 2;
TrainingSet * temp_mem;if (cudaMalloc((void **) &temp_mem, _capacity * sizeof(TrainingSet)) == cudaErrorMemoryAllocation) throw std::bad_alloc();
cudaMemcpy(temp_mem, _data, _size * sizeof(TrainingSet), cudaMemcpyDeviceToDevice);
cudaFree(_data);
_data = temp_mem;
}
}

