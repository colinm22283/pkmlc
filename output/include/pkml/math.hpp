#pragma once

#include <cmath>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace PKML {
    template<typename T>
    struct _Math { };

//    template<>
//    struct _Math<half> {
//        [[nodiscard]] __device__ static inline half add(half a, half b) noexcept { return __hadd(a, b); }
//        [[nodiscard]] __device__ static inline half div(half a, half b) noexcept { return __hdiv(a, b); }
//        [[nodiscard]] __device__ static inline half fma(half a, half b, half c) noexcept { return __hfma(a, b, c); }
//        [[nodiscard]] __device__ static inline half mul(half a, half b) noexcept { return __hmul(a, b); }
//        [[nodiscard]] __device__ static inline half neg(half x) noexcept { return __hneg(x); }
//        [[nodiscard]] __device__ static inline half sub(half a, half b) noexcept { return __hsub(a, b); }
//        [[nodiscard]] __device__ static inline half exp(half x) noexcept { return hexp(x); }
//    };

    template<>
    struct _Math<float> {
        [[nodiscard]] __device__ static inline float add(float a, float b) noexcept { return a + b; }
        [[nodiscard]] __device__ static inline float div(float a, float b) noexcept { return a / b; }
        [[nodiscard]] __device__ static inline float fma(float a, float b, float c) noexcept { return a * b + c; }
        [[nodiscard]] __device__ static inline float mul(float a, float b) noexcept { return a * b; }
        [[nodiscard]] __device__ static inline float neg(float a) noexcept { return -a; }
        [[nodiscard]] __device__ static inline float sub(float a, float b) noexcept { return a - b; }
        [[nodiscard]] __device__ static inline float exp(float a) noexcept { return std::exp(a); }
    };

    using Math = _Math<PKML::float_t>;
}