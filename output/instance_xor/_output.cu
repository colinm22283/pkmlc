#include "_output.hpp"

#include "modules/fully_connected.pkmlm/fully_connected.hpp"
#include "modules/sigmoid.pkmlm/sigmoid.hpp"

using l0_size = PKML::Dimension<2>;
using l01_size = PKML::Dimension<2>;
using l12_size = PKML::Dimension<2>;
using l23_size = PKML::Dimension<2>;
using l3_size = PKML::Dimension<2>;

using l0_t = FullyConnected<l0_size, l01_size, FullyConnected_params { .learning_rate = 0.1, }>;
using l1_t = Sigmoid<l01_size, l12_size, Sigmoid_params { }>;
using l2_t = FullyConnected<l12_size, l23_size, FullyConnected_params { .learning_rate = 0.1, }>;
using l3_t = Sigmoid<l23_size, l3_size, Sigmoid_params { }>;

PKML::float_t * buf0;
PKML::float_t * buf01;
PKML::float_t * buf12;
PKML::float_t * buf23;
PKML::float_t * buf3;

PKML::float_t * l0_alloc;
PKML::float_t * l1_alloc;
PKML::float_t * l2_alloc;
PKML::float_t * l3_alloc;

inline PKML::float_t * input_buffer() noexcept { return buf0; };
inline PKML::float_t * output_buffer() noexcept { return buf3; };

InstanceXor::Network::Network() {
    cudaMalloc((void **) &buf0, l0_size::element_product * sizeof(PKML::float_t));
    cudaMalloc((void **) &buf01, l01_size::element_product * sizeof(PKML::float_t));
    cudaMalloc((void **) &buf12, l12_size::element_product * sizeof(PKML::float_t));
    cudaMalloc((void **) &buf23, l23_size::element_product * sizeof(PKML::float_t));
    cudaMalloc((void **) &buf3, l3_size::element_product * sizeof(PKML::float_t));

    if constexpr (l0_t::memory_requirement > 0) cudaMalloc((void **) &l0_alloc, l0_t::memory_requirement * sizeof(PKML::float_t));
    if constexpr (l1_t::memory_requirement > 0) cudaMalloc((void **) &l1_alloc, l1_t::memory_requirement * sizeof(PKML::float_t));
    if constexpr (l2_t::memory_requirement > 0) cudaMalloc((void **) &l2_alloc, l2_t::memory_requirement * sizeof(PKML::float_t));
    if constexpr (l3_t::memory_requirement > 0) cudaMalloc((void **) &l3_alloc, l3_t::memory_requirement * sizeof(PKML::float_t));

    l0_t::init(l0_alloc);
    l1_t::init(l1_alloc);
    l2_t::init(l2_alloc);
    l3_t::init(l3_alloc);
}
InstanceXor::Network::~Network() {}

template<uint32_t total_threads>
__global__ void epoch_k(
    PKML::float_t * const _buf0,
    PKML::float_t * const _buf01,
    PKML::float_t * const _buf12,
    PKML::float_t * const _buf23,
    PKML::float_t * const _buf3,
    PKML::float_t * const _l0_alloc,
    PKML::float_t * const _l1_alloc,
    PKML::float_t * const _l2_alloc,
    PKML::float_t * const _l3_alloc,
    const PKML::float_t * const costs
) {
    const uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    PKML::thread_gate<total_threads, l12_size::element_product>(thread_index, [thread_index, _buf0, _buf01, _buf12, _buf23, _buf3, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc]() {
        _buf01[thread_index] = l0_t::forward_gated(thread_index, _buf0, _l0_alloc);
        _buf12[thread_index] = l1_t::forward_ungated(_buf01[thread_index], _l1_alloc);
    });

    __syncthreads();

    PKML::thread_gate<total_threads, l3_size::element_product>(thread_index, [thread_index, _buf0, _buf01, _buf12, _buf23, _buf3, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc]() {
        _buf23[thread_index] = l2_t::forward_gated(thread_index, _buf12, _l2_alloc);
        _buf3[thread_index] = l3_t::forward_ungated(_buf23[thread_index], _l3_alloc);
    });

    __shared__ PKML::float_t intermediate_costs[total_threads];
    PKML::float_t * intermediate_costs_ptr = intermediate_costs;

    PKML::thread_gate<total_threads, l3_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, costs]() { intermediate_costs_ptr[thread_index] = costs[thread_index]; });

    PKML::thread_gate<total_threads, l3_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, _buf0, _buf01, _buf12, _buf23, _buf3, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, costs]() {
        PKML::float_t intermediate = intermediate_costs_ptr[thread_index];

        intermediate *= l3_t::backward_ungated(_buf23[thread_index], _buf3[thread_index], _l3_alloc);
        l2_t::backward_gated(thread_index, intermediate_costs_ptr, _buf12, intermediate, _l2_alloc);
    });
    __syncthreads();

    PKML::thread_gate<total_threads, l12_size::element_product>(thread_index, [thread_index, intermediate_costs_ptr, _buf0, _buf01, _buf12, _buf23, _buf3, _l0_alloc, _l1_alloc, _l2_alloc, _l3_alloc, costs]() {
        PKML::float_t intermediate = intermediate_costs_ptr[thread_index];

        intermediate *= l1_t::backward_ungated(_buf01[thread_index], _buf12[thread_index], _l1_alloc);
        l0_t::backward_gated(thread_index, intermediate_costs_ptr, _buf0, intermediate, _l0_alloc);
    });
}
void InstanceXor::Network::run(std::size_t iterations, const PKML::float_t * correct) {
    float out[l3_size::element_product];
    copy_outputs(out);

    for (int i = 0; i < l3_size::element_product; i++) out[i] -= correct[i];

    float * costs;
    cudaMalloc((void **) &costs, l3_size::element_product * sizeof(float));
    cudaMemcpy(costs, out, l3_size::element_product * sizeof(float), cudaMemcpyHostToDevice);

    for (std::size_t i = 0; i < iterations; i++) {
        epoch_k<1 * 100><<<1, 100>>>(buf0, buf01, buf12, buf23, buf3, l0_alloc, l1_alloc, l2_alloc, l3_alloc, costs);
    }

    cudaFree(costs);
}

void InstanceXor::Network::copy_outputs(PKML::float_t * dst) {
    cudaMemcpy(dst, output_buffer(), 2 * sizeof(PKML::float_t), cudaMemcpyDeviceToHost);
}
void InstanceXor::Network::copy_inputs(PKML::float_t * src) {
    cudaMemcpy(input_buffer(), src, 2 * sizeof(PKML::float_t), cudaMemcpyHostToDevice);
}