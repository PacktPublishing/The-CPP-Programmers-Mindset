#ifndef DUCKIES_FILE_READER_H
#define DUCKIES_FILE_READER_H

#include <filesystem>
#include <memory>
#include <string_view>
#include <unordered_map>

#include "duckies_types.h"

namespace duckies {
class FileReader {
public:
    using Data = RubberDuckData;
    using FSPath = std::filesystem::path;

    virtual ~FileReader();

    /**
	 * @brief Read the data from file into data
	 *
	 */
    virtual void read_file(Data &data, const FSPath &path) const = 0;

    /**
	 * @brief Get the file extension that this reader can parse.
	 */
    virtual std::string_view supported_file_extension() const noexcept = 0;
};

std::unordered_map<std::string, std::unique_ptr<FileReader> > get_readers();
} // namespace duckies

#endif // DUCKIES_FILE_READER_H