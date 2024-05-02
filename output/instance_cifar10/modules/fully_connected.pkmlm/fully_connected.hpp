#pragma once

#include <cmath>
#include <memory>

#include <pkml.hpp>

struct FullyConnected_params {
    PKML::float_t learning_rate;
};

template<typename input_dimension, typename output_dimension, FullyConnected_params params>
struct FullyConnected {
    struct allocation_t {
        PKML::float_t weights[output_dimension::element_product][input_dimension::element_product];
        PKML::float_t biases[output_dimension::element_product];
    };

    static inline void init(PKML::float_t * alloc) {
        allocation_t * host = new allocation_t;
        for (std::size_t i = 0; i < output_dimension::element_product; i++) {
            for (std::size_t j = 0; j < input_dimension::element_product; j++) {
                host->weights[i][j] = 0.001;
            }
        }
        for (std::size_t i = 0; i < output_dimension::element_product; i++) host->biases[i] = 0.001;

        cudaMemcpy(alloc, host, sizeof(allocation_t), cudaMemcpyHostToDevice);

        delete host;
    }

    __device__ static inline PKML::float_t forward_gated(uint32_t thread_index, PKML::float_t * input, PKML::float_t * alloc) {
        allocation_t & allocation = *((allocation_t *) alloc);

        PKML::float_t sum = 0;
//#pragma unroll
        for (std::size_t i = 0; i < input_dimension::element_product; i++) {
            sum = PKML::Math::fma(input[i], allocation.weights[thread_index][i], sum);
        }
        return sum;
    }

    __device__ static inline void backward_gated(uint32_t thread_index, PKML::float_t * costs, PKML::float_t * input, PKML::float_t cost, PKML::float_t * alloc) {
        allocation_t & allocation = *((allocation_t *) alloc);

//#pragma unroll
        for (std::size_t i = 0; i < input_dimension::element_product; i++) {
            costs[i] = PKML::Math::fma(
                cost,
                allocation.weights[thread_index][i],
                costs[i]
            );

            allocation.weights[thread_index][i] = PKML::Math::fma(
                -params.learning_rate,
                PKML::Math::mul(cost, input[i]),
                allocation.weights[thread_index][i]
            );
        }

        allocation.biases[thread_index] = PKML::Math::fma(
            -params.learning_rate,
            cost,
            allocation.biases[thread_index]
        );
    }

    static constexpr std::size_t memory_requirement = sizeof(allocation_t) / sizeof(PKML::float_t);
};