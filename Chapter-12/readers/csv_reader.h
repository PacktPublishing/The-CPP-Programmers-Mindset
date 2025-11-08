#ifndef DUCKIES_CSV_READER_H
#define DUCKIES_CSV_READER_H

#include "file_reader.h"

namespace duckies {
class CSVReader : public FileReader {
public:
    using typename FileReader::Data;
    using typename FileReader::FSPath;

    void read_view(Data &data, std::string_view view) const;

    void read_file(Data &data, const FSPath &path) const override;

    std::string_view supported_file_extension() const noexcept override;
};
} // namespace duckies

#endif // DUCKIES_CSV_READER_H