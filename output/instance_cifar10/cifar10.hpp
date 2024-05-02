#pragma once

#include <fstream>

namespace Cifar10 {
    inline void load_datafile(auto & dataset, const char * path) {
        std::ifstream fs(path);
        if (!fs.is_open()) throw std::runtime_error("Can't open file");

        for (int i = 0; i < 10000; i++) {
            float inputs[3072];
            float outputs[10];

            unsigned char index;
            fs.read((char *) &index, sizeof(unsigned char));
            for (unsigned char j = 0; j < 10; j++) {
                if (j == index) outputs[j] = 1;
                else outputs[j] = 0;
            }

            for (int j = 0; j < 3072; j++) {
                unsigned char temp;
                fs.read((char *) &temp, sizeof(unsigned char));
                inputs[j] = (float) temp / 255.0f;
            }

            dataset.copy_back(inputs, outputs);
        }
    }
}