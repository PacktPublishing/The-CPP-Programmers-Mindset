#ifndef DUCKIES_JSON_READER_H
#define DUCKIES_JSON_READER_H

#include "file_reader.h"

#include <iosfwd>

namespace duckies {
class JSONReader : public FileReader {
public:
    using FileReader::Data;
    using FileReader::FSPath;

    void read_stream(Data &data, std::istream &stream) const;

    void read_file(Data &data, const FSPath &path) const override;

    std::string_view supported_file_extension() const noexcept override;
};
} // namespace duckies

#endif // DUCKIES_JSON_READER_H