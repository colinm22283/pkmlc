#pragma once

#include <cmath>

#include <pkml.hpp>

struct Tanh_params { };

template<typename input_dimension, typename output_dimension, Tanh_params params>
struct Tanh {
    static inline void init(PKML::float_t * alloc) { }

    __device__ static inline PKML::float_t forward_ungated(PKML::float_t input, PKML::float_t * alloc) {
        return PKML::Math::tanh(input);
    }

    __device__ static inline PKML::float_t backward_ungated(PKML::float_t input, PKML::float_t output, PKML::float_t * alloc) {
        PKML::float_t sec = PKML::Math::sech(input);
        return sec * sec;
    }

    static constexpr std::size_t memory_requirement = 0;
};