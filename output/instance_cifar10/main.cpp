#include <iostream>
#include <filesystem>

#include "output.hpp"
#include "cifar10.hpp"

int main() {
    InstanceCifar10::Network network;
    InstanceCifar10::Dataset dataset;

    Cifar10::load_datafile(dataset, "cifar10/data_batch_1.bin");
    Cifar10::load_datafile(dataset, "cifar10/data_batch_2.bin");
    Cifar10::load_datafile(dataset, "cifar10/data_batch_3.bin");
    Cifar10::load_datafile(dataset, "cifar10/data_batch_4.bin");
    Cifar10::load_datafile(dataset, "cifar10/data_batch_5.bin");

//    if (std::filesystem::exists("cifar10.net")) network.load("cifar10.net");

    std::ofstream csv;
    std::ofstream accuracy_csv;

    if (std::filesystem::exists("out.csv")) {
        csv.open("out.csv", std::ofstream::app);
    }
    else {
        csv.open("out.csv");
        csv << "Total Cost";
    }
    if (std::filesystem::exists("accuracy.csv")) {
        accuracy_csv.open("accuracy.csv", std::ofstream::app);
    }
    else {
        accuracy_csv.open("accuracy.csv");
        accuracy_csv << "Accuracy";
    }

    for (std::size_t i = 0; i < 1000; i++) {
        float cost = 0;
        float accuracy = 0;

        for (std::size_t j = 0; j < 250; j++) {
            InstanceCifar10::Dataset::TrainingSet set;
            dataset.pull_set(rand() % dataset.size(), set);

            PKML::float_t temp[10];

            network.copy_inputs(set.inputs);
            network.forward();
            network.copy_outputs(temp);

            for (std::size_t k = 0; k < 2; k++) {
                PKML::float_t test = (temp[k] - set.outputs[k]);
                cost += (test * test) / 250;
            }

            int correct_index = 0;
            int output_index = 0;

            for (std::size_t k = 1; k < 10; k++) {
                if (set.outputs[k] > set.outputs[correct_index]) correct_index = k;
                if (temp[k] > temp[output_index]) output_index = k;
            }

            if (correct_index == output_index) accuracy += 1 / 250.0f;
        }

        std::cout << "Total cost: " << cost << "\n";
        std::cout << "Accuracy: " << (accuracy * 100) << "%\n";

        csv << ",\n" << cost;
        csv.flush();

        accuracy_csv << ",\n" << (accuracy * 100);
        accuracy_csv.flush();

        network.train(50000, 1, dataset);

        network.save("cifar10.net");
    }
}