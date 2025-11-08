#include "file_reader.h"

#include "csv_reader.h"
#include "json_reader.h"

#include <filesystem>
using namespace duckies;

FileReader::~FileReader() = default;

std::unordered_map<std::string, std::unique_ptr<FileReader> > duckies::get_readers() {
    std::unordered_map<std::string, std::unique_ptr<FileReader> > readers;

    auto csv_reader = std::make_unique<CSVReader>();
    std::string csv_ext(csv_reader->supported_file_extension());
    readers[csv_ext] = std::move(csv_reader);

    auto json_reader = std::make_unique<JSONReader>();
    std::string json_ext(json_reader->supported_file_extension());
    readers[json_ext] = std::move(json_reader);

    return readers;
}