#include <iostream>

#include "instance_xor/output.hpp"

int main() {
    std::cout << "PKML Test\n";

    InstanceXor::Dataset dataset;
    InstanceXor::Network network;

    PKML::float_t host[2];

    std::cout << "Constructing dataset...\n";

    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 1, 0 },
        .outputs = { 1, 0 },
    });
    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 0, 1 },
        .outputs = { 1, 0 },
    });
    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 0, 0 },
        .outputs = { 0, 1 },
    });
    dataset.push_back(InstanceXor::Dataset::TrainingSet {
        .inputs = { 1, 1 },
        .outputs = { 0, 1 },
    });

    std::cout << "Starting training...\n";

    network.train(20000000, dataset);

    std::cout << "IN:  [ 1, 0 ]\n";
    network.train(1, dataset);
    network.copy_outputs(host);
    std::cout << "OUT: [ " << (float) host[0] << " " << (float) host[1] << " ]\n";

    std::cout << "IN:  [ 0, 1 ]\n";
    network.train(2, dataset);
    network.copy_outputs(host);
    std::cout << "OUT: [ " << (float) host[0] << " " << (float) host[1] << " ]\n";

    std::cout << "IN:  [ 0, 0 ]\n";
    network.train(3, dataset);
    network.copy_outputs(host);
    std::cout << "OUT: [ " << (float) host[0] << " " << (float) host[1] << " ]\n";

    std::cout << "IN:  [ 1, 1 ]\n";
    network.train(4, dataset);
    network.copy_outputs(host);
    std::cout << "OUT: [ " << (float) host[0] << " " << (float) host[1] << " ]\n";
}