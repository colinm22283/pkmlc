#pragma once

namespace PKML {
    template<uint32_t total_threads, uint32_t desired_threads>
    __device__ constexpr void thread_gate(uint32_t thread_index, auto functor) noexcept {
        static_assert(total_threads >= desired_threads, "Not enough threads!");

        if constexpr (total_threads == desired_threads) functor();
        else if (thread_index < desired_threads) functor();
    }
}