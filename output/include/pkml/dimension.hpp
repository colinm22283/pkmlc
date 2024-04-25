#pragma once

#include <cstdint>

namespace PKML {
    template<std::size_t... components>
    struct Dimension {
        template<std::size_t _first, std::size_t... _rest>
        struct _dimension_indexer_t {
            template<std::size_t index>
            static consteval std::size_t get() {
                static_assert(!(sizeof...(_rest) == 0 && index != 0), "Index out of range!");

                if constexpr (index == 0) return _first;
                else _dimension_indexer_t<_rest...>::template get<index - 1>();
            }
        };

        Dimension() = delete;

        static constexpr std::size_t size = sizeof...(components);

        template<std::size_t index>
        static constexpr std::size_t get = _dimension_indexer_t<components...>::template get<index>();

        static constexpr std::size_t element_product = (components * ...);
    };
}