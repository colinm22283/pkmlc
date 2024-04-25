#pragma once

#include <cuda_fp16.h>
namespace PKML { using float_t = float; }

namespace InstanceXor {
    class Network {
    public:
        Network();
        ~Network();

        void run(std::size_t iterations, const PKML::float_t * correct);

        void copy_outputs(PKML::float_t * dst);
        void copy_inputs(PKML::float_t * src);
    };

    class Dataset {
    public:
        Dataset();
        ~Dataset();


    };
};