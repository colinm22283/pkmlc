#pragma once

#include <string>
#include <fstream>

class Parser {
protected:
    std::string file_path;
    std::ifstream fs;

public:
    explicit inline Parser(const char * _file_path): file_path(_file_path), fs(file_path) { }


};