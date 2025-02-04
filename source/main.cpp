#include <iostream>

#include <cli/parser.hpp>

#include <parser.hpp>

int main(int argc, char ** argv) {
    CLI::Parser::parse(argc, argv);

    CLI::Options::input_file = argv[1];

    Parser input_parser(CLI::Options::input_file.c_str());

    input_parser.write_output("output.cu", "output.hpp");

    return 0;
}
