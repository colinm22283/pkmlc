#pragma once

#include <cmath>

#include <pkml.hpp>

struct ReLu_params { };

template<typename input_dimension, typename output_dimension, ReLu_params params>
struct ReLu {
    static inline void init(PKML::float_t * alloc) { }

    __device__ static inline PKML::float_t forward_ungated(PKML::float_t input, PKML::float_t * alloc) {
        if (input >= 0) return input;
        else return 0;
    }

    __device__ static inline PKML::float_t backward_ungated(PKML::float_t input, PKML::float_t output, PKML::float_t * alloc) {
        if (input >= 0) return 1;
        else return 0;
    }

    static constexpr std::size_t memory_requirement = 0;
};