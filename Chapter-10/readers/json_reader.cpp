#include "json_reader.h"

#include <chrono>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <string_view>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <spdlog/spdlog.h>

using namespace duckies;

void JSONReader::read_stream(Data &data, std::istream &stream) const {
    rapidjson::IStreamWrapper json_stream(stream);
    rapidjson::Document doc;
    doc.ParseStream(json_stream);

    if (!doc.IsArray()) {
        spdlog::warn("could not parse as JSON"
            ", top level object is not an array");
        return;
    }

    auto num_records = doc.Size();
    spdlog::info("file contains {} entries", num_records);

    data.reserve_additional(num_records);

    size_t index = 0;
    for (const auto &entry: doc.GetArray()) {
        // checks for well-formedness
        if (!entry.IsObject() || !entry.HasMember("longitude")
            || !entry.HasMember("latitude") || !entry.HasMember("date")) {
            spdlog::warn("entry {} is malformed", index);
            continue;
        }

        // parse coordinate
        float latitude = entry["latitude"].GetDouble();
        float longitude = entry["longitude"].GetDouble();

        // parse date
        std::chrono::year_month_day date;
        std::istringstream stream(entry["date"].GetString());
        std::chrono::from_stream(stream, "%F", date);

        // optionally get description
        std::string descr;
        if (entry.HasMember("description")) {
            descr = entry["description"].GetString();
        }

        data.insert(latitude, longitude, date, std::move(descr));
        ++index;
    }
}

void JSONReader::read_file(Data &data, const FSPath &path) const {
    std::ifstream input_file(path);
    read_stream(data, input_file);
}

std::string_view JSONReader::supported_file_extension() const noexcept {
    return "json";
}