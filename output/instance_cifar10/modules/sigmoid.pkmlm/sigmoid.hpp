#pragma once

#include <cmath>

#include <pkml.hpp>

struct Sigmoid_params { };

template<typename input_dimension, typename output_dimension, Sigmoid_params params>
struct Sigmoid {
    static inline void init(PKML::float_t * alloc) { }

    __device__ static inline PKML::float_t forward_ungated(PKML::float_t input, PKML::float_t * alloc) {
        return PKML::Math::div(1, PKML::Math::add(1, PKML::Math::exp(PKML::Math::neg(input))));
    }

    __device__ static inline PKML::float_t backward_ungated(PKML::float_t input, PKML::float_t output, PKML::float_t * alloc) {
        PKML::float_t exp = PKML::Math::exp(PKML::Math::neg(input));
        PKML::float_t base = PKML::Math::add(
            1,
            PKML::Math::add(
                PKML::Math::mul(2, exp),
                PKML::Math::mul(exp, exp)
            )
        );
        return PKML::Math::div(exp, base);
    }

    static constexpr std::size_t memory_requirement = 0;
};