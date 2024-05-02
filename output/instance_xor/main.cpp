#include <iostream>
#include <fstream>

#include "output.hpp"

int main() {
    std::cout << "PKML Test\n";

    InstanceXor::Dataset dataset;
    InstanceXor::Network network;

    std::cout << "Constructing dataset...\n";

    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 1, 0 },
        .outputs = { 1 },
    });
    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 0, 1 },
        .outputs = { 1 },
    });
    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 0, 0 },
        .outputs = { 0 },
    });
    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 1, 1 },
        .outputs = { 0 },
    });

    std::cout << "Starting training...\n";

    std::ofstream csv("xor.csv");
    csv << "Cost";

    for (std::size_t i = 0; i < 1000; i++) {
        float cost = 0;

        for (std::size_t j = 0; j < dataset.size(); j++) {
            InstanceXor::Dataset::TrainingSet set;
            dataset.pull_set(j, set);

            PKML::float_t temp[2];

            network.copy_inputs(set.inputs);
            network.forward();
            network.copy_outputs(temp);

            for (std::size_t k = 0; k < 2; k++) {
                PKML::float_t test = (temp[k] - set.outputs[k]);
                cost += (test * test) / dataset.size();
            }
        }

        std::cout << "Total cost: " << cost << "\n";

        csv << ",\n" << cost;

        network.train(4000, 1, dataset);
    }

//    network.train(20000000, 1, dataset);

    for (std::size_t i = 0; i < dataset.size(); i++) {
        InstanceXor::Dataset::TrainingSet set;
        dataset.pull_set(i, set);

        PKML::float_t temp[1];

        network.copy_inputs(set.inputs);
        network.forward();
        network.copy_outputs(temp);

        std::cout << "IN:  [ " << set.inputs[0] << ", " << set.inputs[1] << " ]\n";
        std::cout << "OUT: [ " << (float) temp[0] << " ]\n";
    }
}