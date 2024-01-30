#include <iostream>

#include <cli/parser.hpp>

#include <parser.hpp>

int main(int argc, char ** argv) {
    CLI::Parser::parse(argc, argv);

    CLI::Options::input_file = "xor.pkml";

    Parser input_parser(CLI::Options::input_file.c_str());



    return 0;
}
