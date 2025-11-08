#include "csv_reader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>

#include <csv2/reader.hpp>
#include <spdlog/spdlog.h>

using namespace duckies;

using Reader = csv2::Reader<csv2::delimiter<','>, csv2::quote_character<'"'>,
	csv2::first_row_is_header<true>, csv2::trim_policy::trim_whitespace>;
using Row = typename Reader::Row;

static constexpr std::array<std::string_view, 4> col_headings
        = {
            std::string_view("date"), std::string_view("latitude"),
            std::string_view("longitude"), std::string_view("description")
        };

static std::array<int, 4> parse_headers(const Row &header) {
    const auto begin = col_headings.begin();
    const auto end = col_headings.end();

    std::array < int, 4 > interests = {-1, -1, -1, -1};

    int col = 0;
    for (auto cell: header) {
        auto cell_view = cell.read_view();
        auto it = std::find(begin, end, cell_view);

        if (it != end) {
            auto value_index = static_cast<int>(it - begin);
            interests[value_index] = col;
        }

        ++col;
    }

    for (const auto &check: interests) {
        if (check == -1) {
            throw std::invalid_argument("not all required headings are provided");
        }
    }

    return interests;
}

static void read_csv(typename FileReader::Data &data, const Reader &csv) {
    auto interests = parse_headers(csv.header());

    data.reserve_additional(csv.rows());

    const auto ind_begin = interests.begin();
    const auto ind_end = interests.end();

    for (auto row: csv) {
        if (row.length() == 0) { continue; }

        std::array<std::string_view, 4> values;
        int col = 0;
        for (auto cell: row) {
            auto pos = std::find(ind_begin, ind_end, col);
            if (pos != ind_end) {
                auto index = static_cast<int>(pos - ind_begin);
                values[index] = cell.read_view();
            }

            ++col;
        }

        // parse string values to actual values
        std::string tmp(values[0]);
        std::istringstream ss(tmp);
        std::chrono::year_month_day date;
        std::chrono::from_stream(ss, "%F", date);

        tmp = values[1];
        auto latitude = std::stof(tmp);

        tmp = values[2];
        auto longitude = std::stof(tmp);

        tmp = values[3];

        // insert into data container
        data.insert(latitude, longitude, std::move(date), std::move(tmp));
    }
}

void CSVReader::read_view(Data &data, std::string_view view) const {
    Reader csv;
    if (!csv.parse_view(view)) {
        spdlog::warn("failed to parse csv from string");
        return;
    }

    read_csv(data, csv);
}

void CSVReader::read_file(Data &data, const FSPath &path) const {
    Reader csv;
    if (!csv.mmap(path.c_str())) {
        spdlog::warn("failed to parse csv from file \"{}\"", path.c_str());
        return;
    }

    read_csv(data, csv);
}

std::string_view CSVReader::supported_file_extension() const noexcept {
    return "csv";
}