#pragma once

#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

#include <logger.hpp>

class Parser {
protected:
    static constexpr auto logger = Logger("File Parser");

    struct module_t {
        std::string folder_path;
        std::string class_path;
        std::string class_name;
        bool gated;
    };

    struct layer_t {
        module_t & module;
        std::string module_name;
        std::vector<std::string> input_dimension;
        std::vector<std::string> output_dimension;
        std::vector<std::size_t> input_dimension_nums;
        std::vector<std::size_t> output_dimension_nums;
        nlohmann::json params;
    };

    std::string file_path;
    std::ifstream fs;

    nlohmann::json data;

    std::vector<module_t> modules;
    std::vector<layer_t> layers;

    std::string class_name;
    std::string float_type;
    std::string grid_dim, block_dim;

public:
    explicit inline Parser(const char * _file_path): file_path(_file_path), fs(file_path) {
        data = nlohmann::json::parse(fs);

        if (!data["modules"].is_array()) {
            logger << "Unable to locate modules definition";

            throw std::exception();
        }

        for (std::string module_path : data["modules"]) {
            logger << "Opening module '" << module_path << "'";

            module_t & module = modules.emplace_back();

            nlohmann::json module_data = nlohmann::json::parse(std::ifstream(module_path + std::string("/manifest.json")));

            module.folder_path = module_path;

            if (!module_data.contains("class_file") || !module_data.at("class_file").is_string()) {
                logger.error() << "Module manifest does not contain 'class_file' field";

                throw std::exception();
            }
            module.class_path = module_path + '/' + std::string(module_data.at("class_file"));

            if (!module_data.contains("class_name") || !module_data.at("class_name").is_string()) {
                logger.error() << "Module manifest does not contain 'class_name' field";

                throw std::exception();
            }
            module.class_name = std::string(module_data.at("class_name"));

            if (!module_data.contains("gated") || !module_data.at("gated").is_boolean()) {
                logger.error() << "Module manifest does not contain 'gated' field";

                throw std::exception();
            }
            module.gated = bool(module_data.at("gated"));
        }

        auto & network_data = data.at("network");

        class_name = std::string(network_data.at("class_name"));
        float_type = std::string(network_data.at("float_type"));
        grid_dim = std::to_string((int) network_data.at("grid_dim"));
        block_dim = std::to_string((int) network_data.at("block_dim"));

        for (auto & layer_data : network_data.at("layers")) {
            std::string module_name = std::string(layer_data.at("module_name"));
            module_t * module = nullptr;
            for (auto & mod : modules) {
                if (mod.class_name == module_name) {
                    module = &mod;
                    break;
                }
            }
            if (!module) {
                logger.error() << "Unable to find module with name '" << module_name << "'";

                throw std::exception();
            }

            layer_t layer {
                .module = *module,
                .module_name = module_name,
            };

            for (auto & dimension_ele : layer_data.at("input_dimension")) {
                layer.input_dimension.push_back(std::to_string((int) dimension_ele));
                layer.input_dimension_nums.push_back((std::size_t) dimension_ele);
            }

            for (auto & dimension_ele : layer_data.at("output_dimension")) {
                layer.output_dimension.push_back(std::to_string((int) dimension_ele));
                layer.output_dimension_nums.push_back((std::size_t) dimension_ele);
            }

            if (layer_data.contains("params")) layer.params = layer_data.at("params");

            layers.push_back(std::move(layer));
        }
    }

    inline void write_output(const char * source_file, const char * header_file) {
        std::ofstream sfs(source_file);
        std::ofstream hfs(header_file);

        auto print_buf_name = [this](auto & stream, std::size_t layer) {
            if (layer == 0) stream << "buf0";
            else if (layer == layers.size()) stream << "buf" << (layer - 1);
            else {
                stream << "buf" << (layer - 1) << '_' << layer;
            }
        };
        auto print_layer_size_name = [this](auto & stream, std::size_t layer) {
            if (layer == 0) stream << "l0_size";
            else if (layer == layers.size())  stream << "l" << (layer - 1) << "_size";
            else {
                stream << "l" << (layer - 1) << '_' << layer << "_size";
            }
        };
        auto print_layer_alloc_name = [this](auto & stream, std::size_t layer) {
            stream << "l" << layer << "_alloc";
        };
        auto print_layer_type = [this](auto & stream, std::size_t layer) {
            stream << "l" << layer << "_t";
        };

        auto print_thread_gate_open = [this, print_buf_name, print_layer_alloc_name, print_layer_size_name](auto & stream, std::size_t size) {
            auto print_lambda_capture = [this, print_buf_name, print_layer_alloc_name](auto & stream) {
                stream << "thread_index, ";
                for (std::size_t j = 0; j < layers.size() + 1; j++) {
                    stream << "_";
                    print_buf_name(stream, j);
                    stream << ", ";
                }
                for (std::size_t j = 0; j < layers.size() - 1; j++) {
                    stream << "_";
                    print_layer_alloc_name(stream, j);
                    stream << ", ";
                }
                stream << "_";
                print_layer_alloc_name(stream, layers.size() - 1);
            };

            stream << "PKML::thread_gate<total_threads, ";
            print_layer_size_name(stream, size + 1);
            stream << "::element_product>(thread_index, [";
            print_lambda_capture(stream);
            stream << "]() {\n";
        };

        auto compare_dimensions = [](std::vector<std::size_t> & a, std::vector<std::size_t> & b) {
            if (a.size() != b.size()) return false;

            for (std::size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) return false;

            return true;
        };

        hfs << "#pragma once\n"
               "#include <cstdint>\n"
               "namespace PKML{using float_t="
            << float_type
            << ";}\n"
               "#include <pkml.hpp>\n"
               "namespace "
            << class_name
            << "{"
               "namespace _LayerSizes {";
        for (std::size_t i = 0; i < layers.size(); i++) {
            hfs << "using ";
            print_layer_size_name(hfs, i);
            hfs << " = PKML::Dimension<";
            for (std::size_t j = 0; j < layers[i].input_dimension.size() - 1; j++) {
                hfs << layers[i].input_dimension[j];
                hfs << ", ";
            }
            hfs << layers[i].input_dimension[layers[i].input_dimension.size() - 1];
            hfs << ">;";
        }
        hfs << "using ";
        print_layer_size_name(hfs, layers.size());
        hfs << " = PKML::Dimension<";
        for (std::size_t i = 0; i < layers[layers.size() - 1].input_dimension.size() - 1; i++) {
            hfs << layers[layers.size() - 1].input_dimension[i];
            hfs << ", ";
        }
        hfs << layers[layers.size() - 1].input_dimension[layers[layers.size() - 1].input_dimension.size() - 1];
        hfs << ">;";
        hfs << "}"
              "class Dataset;"
              "class Network{"
              "public:"
              "Network();"
              "~Network();"
              "void forward();"
              "void train(std::size_t iterations, std::size_t mult, "
            << class_name
            << "::Dataset & dataset);"
               "PKML::float_t evaluate("
            << class_name
            << "::Dataset & dataset);"
               "void save(const char * path);"
               "void load(const char * path);"
               "void copy_outputs(PKML::float_t * dst);"
               "void copy_inputs(PKML::float_t * src);"
               "};"
               "class Dataset {"
               "friend class " << class_name << "::Network;"
               "public:"
               "struct TrainingSet {"
               "PKML::float_t inputs[" << class_name << "::_LayerSizes::";
        print_layer_size_name(hfs, 0);
        hfs << "::element_product];"
               "PKML::float_t outputs[" << class_name << "::_LayerSizes::";
        print_layer_size_name(hfs, layers.size());
        hfs << "::element_product];"
               "};"
               "protected:"
               "std::size_t _size, _capacity;"
               "TrainingSet * _data;"
               "public:"
               "Dataset();"
               "~Dataset();"

               "void push_back(const TrainingSet & value);"
               "inline void push_back(const TrainingSet && value) { push_back((TrainingSet &) value); }"
               "inline void copy_back(PKML::float_t * inputs, PKML::float_t * outputs) {"
               "TrainingSet temp;"
               "for (std::size_t i = 0; i < " << class_name << "::_LayerSizes::";
        print_layer_size_name(hfs, 0);
        hfs << "::element_product; i++) temp.inputs[i] = inputs[i];"
               "for (std::size_t i = 0; i < " << class_name << "::_LayerSizes::";
        print_layer_size_name(hfs, layers.size());
        hfs << "::element_product; i++) temp.outputs[i] = outputs[i];"
               "push_back(temp);"
               "}"
               "void pull_set(std::size_t index, TrainingSet & set) const;"
               "[[nodiscard]] inline std::size_t size() const noexcept { return _size; }"
               "[[nodiscard]] inline std::size_t capacity() const noexcept { return _capacity; }"
               "};"
               "}";

        sfs << "#include <fstream>\n";
        sfs << "#include \"output.hpp\"\n";
        for (module_t & module : modules) {
            sfs << "#include \"" << module.class_path << "\"\n";
        }

        sfs << '\n';

        {
            auto print_csv = [](auto & stream, std::vector<std::string> & dim) {
                for (std::size_t i = 0; i < dim.size() - 1; i++) stream << dim[i] << ',';
                stream << dim[dim.size() - 1];
            };

            sfs << "using l0_size=PKML::Dimension<";
            print_csv(sfs, layers[0].input_dimension);
            sfs << ">;\n";

            for (std::size_t i = 0; i < layers.size() - 1; i++) {
                sfs << "using l";
                sfs << i << '_' << (i + 1);
                sfs << "_size=PKML::Dimension<";
                print_csv(sfs, layers[i].output_dimension);
                sfs << ">;\n";

                if (layers[i].output_dimension != layers[i + 1].input_dimension) throw std::runtime_error("Incompatible widths between layers");
            }

            sfs << "using l" << layers.size() - 1 << "_size=PKML::Dimension<";
            print_csv(sfs, layers[layers.size() - 1].output_dimension);
            sfs << ">;\n";
        }

        sfs << '\n';

        {
            for (int i = 0; i < layers.size(); i++) {
                sfs << "using l" << i << "_t=" << layers[i].module_name << '<';
                print_layer_size_name(sfs, i);
                sfs << ",";
                print_layer_size_name(sfs, i + 1);
                sfs << "," << layers[i].module_name << "_params{";
                for (auto & ele : layers[i].params.items()) {
                    sfs << "." << ele.key() << "=" << ele.value() << ",";
                }
                sfs << "}>;\n";
            }
        }

        sfs << '\n';

        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "PKML::float_t * ";
            print_buf_name(sfs, i);
            sfs << ";\n";
        }

        sfs << '\n';

        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "PKML::float_t * l"
                << i
                << "_alloc;\n";
        }

        sfs << '\n';

        sfs << "static constexpr PKML::float_t *& input_buffer = ";
        print_buf_name(sfs, 0);
        sfs << ";\n";
        sfs << "static constexpr PKML::float_t *& output_buffer = ";
        print_buf_name(sfs, layers.size());
        sfs << ";\n";

        sfs << '\n';

        sfs << class_name << "::Network::Network() {\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "cudaMalloc((void **) &";
            print_buf_name(sfs, i);
            sfs << ", ";
            print_layer_size_name(sfs, i);
            sfs << "::element_product * sizeof(PKML::float_t));\n";
        }
        sfs << '\n';
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "if constexpr (l"
                << i
                << "_t::memory_requirement > 0) cudaMalloc((void **) &l"
                << i
                << "_alloc, l"
                << i
                << "_t::memory_requirement * sizeof(PKML::float_t));\n";
        }
        sfs << '\n';
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "l"
                << i
                << "_t::init(l"
                << i
                << "_alloc);\n";
        }
        sfs << "}\n\n";

        sfs << class_name << "::Network::~Network() {\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "cudaFree(";
            print_buf_name(sfs, i);
            sfs << ");\n";
        }
        sfs << '\n';
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "if constexpr (l"
                << i
                << "_t::memory_requirement > 0) cudaFree(l"
                << i
                << "_alloc);\n";
        }
        sfs << "}\n\n";

        // forward_d start
        sfs << "template<std::size_t total_threads>\n"
            << "__device__ void forward_d(\n"
            << "const std::size_t thread_index,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "PKML::float_t * const _";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size() - 1; i++) {
            sfs << "PKML::float_t * const _";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "PKML::float_t * const _";
        print_layer_alloc_name(sfs, layers.size() - 1);
        sfs << ") {\n";

        {
            bool is_first_layer = true;
            std::vector<std::size_t> prev_layer_size;
            std::fill(prev_layer_size.begin(), prev_layer_size.end(), 0);
            for (std::size_t i = 0; i < layers.size(); i++) {
                if (!compare_dimensions(layers[i].output_dimension_nums, prev_layer_size) || layers[i].module.gated) {
                    if (!is_first_layer) {
                        sfs << "});\n";
                        if (layers[i].module.gated) sfs << "__syncthreads();\n";
                    }
                    else is_first_layer = false;

                    prev_layer_size = layers[i].output_dimension_nums;
                    print_thread_gate_open(sfs, i + 1);

                    sfs << "_";
                    print_buf_name(sfs, i + 1);
                    sfs << "[thread_index] = ";
                    print_layer_type(sfs, i);
                    sfs << "::forward_gated(thread_index, _";
                    print_buf_name(sfs, i);
                    sfs << ", _";
                    print_layer_alloc_name(sfs, i);
                    sfs << ");\n";
                }
                else {
                    sfs << "_";
                    print_buf_name(sfs, i + 1);
                    sfs << "[thread_index] = ";
                    print_layer_type(sfs, i);
                    sfs << "::forward_ungated(_";
                    print_buf_name(sfs, i);
                    sfs << "[thread_index], _";
                    print_layer_alloc_name(sfs, i);
                    sfs << ");\n";
                }
            }

            sfs << "});\n"
                << "__syncthreads();\n";
        }
        sfs << "}\n\n";
        // forward_d end

        // propogate_d start
        sfs << "template<std::size_t total_threads>\n"
            << "__device__ void propogate_d(\n"
            << "const std::size_t thread_index,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "PKML::float_t * const _";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "PKML::float_t * const _";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "const PKML::float_t * const correct\n"
            << ") {\n"
            << "forward_d<total_threads>(\n"
            << "thread_index,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "_";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size() - 1; i++) {
            sfs << "_";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "_";
        print_layer_alloc_name(sfs, layers.size() - 1);
        sfs << ");\n";

        {
            auto print_backward_thread_gate_open = [this, print_buf_name, print_layer_alloc_name, print_layer_size_name](auto & stream, std::size_t size) {
                auto print_lambda_capture = [this, print_buf_name, print_layer_alloc_name](auto & stream) {
                    stream << "thread_index, intermediate_costs_ptr, ";
                    for (std::size_t j = 0; j < layers.size() + 1; j++) {
                        stream << "_";
                        print_buf_name(stream, j);
                        stream << ", ";
                    }
                    for (std::size_t j = 0; j < layers.size() - 1; j++) {
                        stream << "_";
                        print_layer_alloc_name(stream, j);
                        stream << ", ";
                    }
                    stream << "_";
                    print_layer_alloc_name(stream, layers.size() - 1);
                };

                stream << "PKML::thread_gate<total_threads, ";
                print_layer_size_name(stream, size + 1);
                stream << "::element_product>(thread_index, [";
                print_lambda_capture(stream);
                stream << "]() {\n";
            };

            sfs << "__shared__ PKML::float_t intermediate_costs[total_threads];\n"
                << "PKML::float_t * intermediate_costs_ptr = intermediate_costs;\n";

            sfs << "PKML::thread_gate<total_threads, ";
            print_layer_size_name(sfs, layers.size());
            sfs << "::element_product>(thread_index, [thread_index, intermediate_costs_ptr, correct, _";
            print_buf_name(sfs, layers.size());
            sfs << "]() {\n"
                << "intermediate_costs_ptr[thread_index] = PKML::Math::sub(_";
            print_buf_name(sfs, layers.size());
            sfs << "[thread_index], correct[thread_index]);\n"
                << "});\n";

            print_backward_thread_gate_open(sfs, layers.size() - 1);
            sfs << "PKML::float_t intermediate = intermediate_costs_ptr[thread_index];\n";
            for (long i = (long) layers.size() - 1; i >= 0; i--) {
                if (layers[i].module.gated) {
                    print_layer_type(sfs, i);
                    sfs << "::backward_gated(thread_index, intermediate_costs_ptr, _";
                    print_buf_name(sfs, i);
                    sfs << ", intermediate, _";
                    print_layer_alloc_name(sfs, i);
                    sfs << ");\n";

                    sfs << "});\n"
                        << "__syncthreads();\n";

                    if (i != 0) {
                        print_backward_thread_gate_open(sfs, i - 1);
                        sfs << "PKML::float_t intermediate = intermediate_costs_ptr[thread_index];\n";
                    }
                }
                else {
                    sfs << "intermediate = PKML::Math::mul(intermediate, ";
                    print_layer_type(sfs, i);
                    sfs << "::backward_ungated(_";
                    print_buf_name(sfs, i);
                    sfs << "[thread_index], _";
                    print_buf_name(sfs, i + 1);
                    sfs << "[thread_index], _";
                    print_layer_alloc_name(sfs, i);
                    sfs << "));\n";
                }
            }
        }

        sfs << "}\n\n";
        // propogate_d end

        // forward_k start
        sfs << "template<std::size_t total_threads>\n"
            << "__global__ void forward_k(\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "PKML::float_t * const _";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size() - 1; i++) {
            sfs << "PKML::float_t * const _";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "PKML::float_t * const _";
        print_layer_alloc_name(sfs, layers.size() - 1);
        sfs << ") {\n"
            << "const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
            << "forward_d<total_threads>(\n"
            << "thread_index,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "_";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size() - 1; i++) {
            sfs << "_";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "_";
        print_layer_alloc_name(sfs, layers.size() - 1);
        sfs << ");\n"
            << "}\n\n";
        // forward_k end

        // forward start
        sfs << "void "
            << class_name
            << "::Network::forward() {\n"
            << "forward_k<"
            << grid_dim
            << " * "
            << block_dim
            << "><<<"
            << grid_dim
            << ", "
            << block_dim
            << ">>>(\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size() - 1; i++) {
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }

        print_layer_alloc_name(sfs, layers.size() - 1);
        sfs << ");\n"
            << "}\n\n";
        // forward end

        sfs << "template<std::size_t total_threads>\n"
            << "__global__ void train_k(\n"
            << "std::size_t iterations,\n"
            << "std::size_t mult,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "PKML::float_t * const _";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "PKML::float_t * const _";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "const " << class_name << "::Dataset::TrainingSet * const dataset,\n"
            << "const std::size_t dataset_size\n"
            << ") {\n"
            << "const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
            << "for (std::size_t i = 0; i < iterations; i++) {\n"
            << "const " << class_name << "::Dataset::TrainingSet & training_set = dataset[(i * mult) % dataset_size];\n"
            << "PKML::thread_gate<total_threads, l0_size::element_product>(thread_index, [thread_index, training_set, _buf0]() {\n"
            << "_buf0[thread_index] = training_set.inputs[thread_index]; // possible optimization by moving _buf0 pointer\n"
            << "});\n"
            << "__syncthreads();\n"
            << "propogate_d<total_threads>(\n"
            << "thread_index,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "_";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "_";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "training_set.outputs\n"
            << ");\n"
            << "}\n"
            << "}\n\n";

        sfs << "void "
            << class_name
            << "::Network::train(std::size_t iterations, std::size_t mult, "
            << class_name
            << "::Dataset & dataset) {\n"
            << "train_k<"
            << grid_dim
            << " * "
            << block_dim
            << "><<<"
            << grid_dim
            << ", "
            << block_dim
            << ">>>(\n"
            << "iterations,\n"
            << "mult,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "dataset._data,\n"
            << "dataset.size()\n"
            << ");\n"
            << "cudaDeviceSynchronize();\n"
            << "}\n\n";

        // evaluate_k start
        sfs << "template<std::size_t total_threads>\n"
            << "__global__ void evaluate_k(\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "PKML::float_t * const _";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "PKML::float_t * const _";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "const " << class_name << "::Dataset::TrainingSet * const dataset,\n"
            << "const std::size_t dataset_size,\n"
            << "PKML::float_t * cost_ptr\n"
            << ") {\n"
            << "const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
            << "PKML::float_t cost = 0;\n"
            << "for (std::size_t i = 0; i < dataset_size; i++) {\n"
            << "const " << class_name << "::Dataset::TrainingSet & training_set = dataset[i];\n"
            << "PKML::thread_gate<total_threads, l0_size::element_product>(thread_index, [thread_index, training_set, _buf0]() {\n"
            << "_buf0[thread_index] = training_set.inputs[thread_index]; // possible optimization by moving _buf0 pointer\n"
            << "});\n"
            << "__syncthreads();\n"
            << "forward_d<total_threads>(\n"
            << "thread_index,\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            sfs << "_";
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size() - 1; i++) {
            sfs << "_";
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "_";
        print_layer_alloc_name(sfs, layers.size() - 1);
        sfs << ");\n"
            << "PKML::thread_gate<total_threads, ";
        print_layer_size_name(sfs, layers.size() - 1);
        sfs << "::element_product>(thread_index, [thread_index, &cost, dataset_size, training_set, _";
        print_buf_name(sfs, layers.size());
        sfs << "]() {\n"
            << "cost += (training_set.outputs[thread_index] - _";
        print_buf_name(sfs, layers.size());
        sfs << "[thread_index]) / dataset_size;\n"
            << "});\n"
            << "}\n"
            << "*cost_ptr = cost;\n"
            << "}\n\n";
        // evaluate_k end

        // evaluate start
        sfs << "PKML::float_t "
            << class_name
            << "::Network::evaluate("
            << class_name
            << "::Dataset & dataset) {\n"
            << "PKML::float_t * dev_cost;\n"
            << "cudaMalloc((void **) &dev_cost, sizeof(PKML::float_t));\n"
            << "evaluate_k<"
            << grid_dim
            << " * "
            << block_dim
            << "><<<"
            << grid_dim
            << ", "
            << block_dim
            << ">>>(\n";
        for (std::size_t i = 0; i < layers.size() + 1; i++) {
            print_buf_name(sfs, i);
            sfs << ",\n";
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            print_layer_alloc_name(sfs, i);
            sfs << ",\n";
        }
        sfs << "dataset._data,\n"
            << "dataset.size(),\n"
            << "dev_cost\n"
            << ");\n"
            << "PKML::float_t cost;\n"
            << "cudaMemcpy(&cost, dev_cost, sizeof(float), cudaMemcpyDeviceToHost);\n"
            << "return cost;\n"
            << "}\n\n";
        // evaluate end

        // save start
        sfs << "void "
            << class_name
            << "::Network::save(const char * path) {\n"
            << "std::ofstream fs(path);\n";
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "if constexpr (";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement != 0) {\n";

            sfs << "float * temp = new float[";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement"
                << "];\n"
                << "cudaMemcpy(temp, ";
            print_layer_alloc_name(sfs, i);
            sfs << ", ";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement * sizeof(float), cudaMemcpyDeviceToHost);\n"
                << "fs.write((const char *) temp, ";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement * sizeof(float));\n"
                << "delete[] temp;\n";

            sfs << "}\n";
        }
        sfs << "}\n\n";
        // save end

        // load start
        sfs << "void "
            << class_name
            << "::Network::load(const char * path) {\n"
            << "std::ifstream fs(path);\n";
        for (std::size_t i = 0; i < layers.size(); i++) {
            sfs << "if constexpr (";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement != 0) {\n";

            sfs << "float * temp = new float[";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement"
                << "];\n"
                << "fs.read((char *) temp, ";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement * sizeof(float));\n";

            sfs << "cudaMemcpy(";
            print_layer_alloc_name(sfs, i);
            sfs << ", temp, ";
            print_layer_type(sfs, i);
            sfs << "::memory_requirement * sizeof(float), cudaMemcpyHostToDevice);\n"
                << "delete[] temp;\n";

            sfs << "}\n";
        }
        sfs << "}\n\n";
        // load end

        sfs << "void "
            << class_name
            << "::Network::copy_outputs(PKML::float_t * dst) {\n"
               "cudaMemcpy(dst, output_buffer, ";
        print_layer_size_name(sfs, layers.size());
        sfs << "::element_product * sizeof(PKML::float_t), cudaMemcpyDeviceToHost);\n"
               "}\n\n";
        sfs << "void "
            << class_name
            << "::Network::copy_inputs(PKML::float_t * src) {\n"
            << "cudaMemcpy(input_buffer, src, ";
        print_layer_size_name(sfs, 0);
        sfs << "::element_product * sizeof(PKML::float_t), cudaMemcpyHostToDevice);\n"
               "}\n\n";

        sfs << class_name << "::Dataset::Dataset(): _size(0), _capacity(1) {\n";
        sfs << "if (cudaMalloc((void **) &_data, _capacity * sizeof(TrainingSet)) == cudaErrorMemoryAllocation) throw std::bad_alloc();\n";
        sfs << "}\n\n";
        sfs << class_name << "::Dataset::~Dataset() {\n";
        sfs << "cudaFree(_data);\n";
        sfs << "}\n\n";
        sfs << "void " << class_name << "::Dataset::push_back(const TrainingSet & value) {\n";
        sfs << "if (_size == _capacity) {\n";
        sfs << "_capacity *= 2;\n";
        sfs << "TrainingSet * temp_mem;\n";
        sfs << "if (cudaMalloc((void **) &temp_mem, _capacity * sizeof(TrainingSet)) == cudaErrorMemoryAllocation) throw std::bad_alloc();\n";
        sfs << "cudaMemcpy(temp_mem, _data, _size * sizeof(TrainingSet), cudaMemcpyDeviceToDevice);\n";
        sfs << "cudaFree(_data);\n";
        sfs << "_data = temp_mem;\n";
        sfs << "}\n";
        sfs << "cudaMemcpy(&_data[_size++], &value, sizeof(TrainingSet), cudaMemcpyHostToDevice);\n";
        sfs << "}\n\n";

        sfs << "void " << class_name << "::Dataset::pull_set(std::size_t index, TrainingSet & set) const {\n";
        sfs << "cudaMemcpy(&set, &_data[index], sizeof(TrainingSet), cudaMemcpyDeviceToHost);\n";
        sfs << "}\n\n";
    }
};