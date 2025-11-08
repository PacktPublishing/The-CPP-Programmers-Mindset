#include "free_reader.h"

#include <charconv>
#include <chrono>
#include <fstream>
#include <istream>
#include <optional>
#include <string>
#include <string_view>

#include <ctre-unicode.hpp>
#include <spdlog/spdlog.h>

#include "duckies_types.h"

using namespace duckies;
namespace chrono = std::chrono;

template<typename T>
bool str_to_num(std::u8string_view sv, T &result) {
    auto *begin = reinterpret_cast<const char *>(sv.data());
    auto *end = begin + sv.size();
    std::from_chars(begin, end, result);
    return result;
}

template<typename Match>
static std::optional<chrono::year> get_year(const Match &match) {
    std::optional<chrono::year> result;

    std::u8string_view view;
    if (auto group = match.template get<iso_date_year_group>()) {
        view = group.to_view();
    } else if (auto group = match.template get<dmy_date_year_group>()) {
        view = group.to_view();
    } else if (auto group = match.template get<mdy_date_year_group>()) {
        view = group.to_view();
    } else {
        spdlog::warn("no matching day pattern");
        return result;
    }

    int num = 0;
    if (str_to_num(view, num)) {
        result = chrono::year{num};
    }
    return result;
}

template<typename Match>
static std::optional<chrono::day> get_day(const Match &match) {
    std::optional<chrono::day> result;

    std::u8string_view view;
    if (auto iso = match.template get<iso_date_day_group>()) {
        view = iso.to_view();
    } else if (auto dmy = match.template get<dmy_date_day_group>()) {
        view = dmy.to_view();
    } else if (auto mdy = match.template get<mdy_date_day_group>()) {
        view = mdy.to_view();
    } else {
        spdlog::warn("no matching day pattern");
        return result;
    }

    unsigned num = 0;
    if (str_to_num(view, num)) {
        result = chrono::day{num};
    }
    return result;
}

template<typename Match>
static std::optional<chrono::month> get_month(const Match &match) {
    std::optional<chrono::month> result;

    if (auto month = match.template get<iso_date_month_group>()) {
        unsigned month_num = 0;
        if (str_to_num(month.to_view(), month_num)) {
            result = chrono::month{month_num};
        }
    } else {
        std::u8string_view view;
        if (auto dmy = match.template get<dmy_date_month_group>()) {
            view = dmy.to_view();
        } else if (auto mdy = match.template get<mdy_date_month_group>()) {
            view = mdy.to_view();
        } else {
            spdlog::error("no matching month");
            return result;
        }

        switch (view[0]) {
            case 'J':
                if (view[1] == 'a') {
                    result = chrono::January;
                } else if (view[2] == 'n') {
                    result = chrono::June;
                } else {
                    result = chrono::July;
                }
                break;
            case 'A':
                if (view[1] == 'p') {
                    result = chrono::April;
                } else {
                    result = chrono::August;
                }
                break;
            case 'F':
                result = chrono::February;
                break;
            case 'M':
                result = chrono::March;
                break;
            case 'S':
                result = chrono::September;
                break;
            case 'O':
                result = chrono::October;
                break;
            case 'N':
                result = chrono::November;
                break;
            case 'D':
                result = chrono::December;
                break;
        }
    }
    return result;
}

static std::optional<chrono::year_month_day> parse_date(std::u8string_view text) {
    // Pattern and names of groups as fixed_strings
    constexpr ctll::fixed_string date_re{RE_DATE};

    std::optional<chrono::year_month_day> result;
    auto match = ctre::search < date_re > (text);
    if (!match) {
        spdlog::warn("no date string match found");
        return result;
    }

    auto year = get_year(match);
    auto month = get_month(match);
    auto day = get_day(match);
    if (!year || !month || !day) {
        spdlog::warn("missing year, month, or day");
        return result;
    }

    result = chrono::year_month_day{*year, *month, *day};
    if (!result->ok()) {
        spdlog::warn("parsed an invalid date");
        result = std::nullopt;
    }

    return result;
}

template<typename Match>
static std::optional<Coordinate> get_dms(const Match &match) {
    constexpr ctll::fixed_string lat_d_name{"lat_deg"};
    constexpr ctll::fixed_string lat_m_name{"lat_min"};
    constexpr ctll::fixed_string lat_s_name{"lat_sec"};
    constexpr ctll::fixed_string lat_cd_name{"lat_card"};
    constexpr ctll::fixed_string lon_d_name{"lon_deg"};
    constexpr ctll::fixed_string lon_m_name{"lon_min"};
    constexpr ctll::fixed_string lon_s_name{"lon_sec"};
    constexpr ctll::fixed_string lon_cd_name{"lon_card"};

    std::optional<Coordinate> result;

    auto lat_d = match.template get<lat_d_name>();
    auto lat_m = match.template get<lat_m_name>();
    auto lat_s = match.template get<lat_s_name>();
    auto lat_cd = match.template get<lat_cd_name>();

    auto lon_d = match.template get<lon_d_name>();
    auto lon_m = match.template get<lon_m_name>();
    auto lon_s = match.template get<lon_s_name>();
    auto lon_cd = match.template get<lon_cd_name>();

    float longitude = 0.0f, latitude = 0.0f, tmp = 0.0f;
    if (lat_d && str_to_num(lat_d.to_view(), tmp)) {
        latitude += tmp;
    }
    if (lat_m && str_to_num(lat_m.to_view(), tmp)) {
        latitude += tmp / 60;
    }
    if (str_to_num(lat_s.to_view(), tmp)) {
        latitude += tmp / 3600;
    }
    if (lat_cd.to_view()[0] == 'S') {
        latitude = -latitude;
    }

    if (lon_d && str_to_num(lon_d.to_view(), tmp)) {
        longitude += tmp;
    }
    if (lon_m && str_to_num(lon_m.to_view(), tmp)) {
        longitude += tmp / 60;
    }
    if (str_to_num(lon_s.to_view(), tmp)) {
        longitude += tmp / 3600;
    }
    if (lon_cd.to_view()[0] == 'W') {
        longitude = -longitude;
    }

    if (-90.0f <= latitude && latitude <= 90.0f && -180.0f < longitude
        && longitude <= 180.0f) {
        result = Coordinate{latitude, longitude};
    } else {
        spdlog::warn("parsed invalid coordinates {}, {}", latitude, longitude);
    }

    return result;
}

static std::optional<Coordinate> parse_coord(std::u8string_view text) {
    constexpr ctll::fixed_string coord_re{RE_COORD};

    std::optional<Coordinate> result;
    auto match = ctre::search < coord_re > (text);

    if (!match) {
        spdlog::warn("no coordinates found in text");
        return result;
    }

    constexpr ctll::fixed_string dd_lat_name{"dd_lat"};
    constexpr ctll::fixed_string dd_lon_name{"dd_lon"};

    if (auto dd_lat = match.get<dd_lat_name>()) {
        auto dd_lon = match.get<dd_lon_name>();
        float lat, lon;
        if (str_to_num(dd_lat.to_view(), lat)
            && str_to_num(dd_lon.to_view(), lon)) {
            if (-90.0f <= lat && lat <= 90.0f && -180.0f < lon
                && lon <= 180.0f) {
                result = Coordinate{lat, lon};
            } else {
                spdlog::warn("parsed invalid coordinates {}, {}", lat, lon);
            }
        } else {
            spdlog::warn("unable to parse decimal degree lat or long");
        }
    } else {
        result = get_dms(match);
    }

    return result;
}

static void parse_entry(
    typename FileReader::Data &data, std::u8string_view text) {
    auto date = parse_date(text);
    auto coords = parse_coord(text);

    if (date && coords) {
        data.insert(coords->latitude, coords->longitude, *date,
                    std::string(text.begin(), text.end()));
    }
}

static bool get_entry(std::istream &stream, std::u8string &buffer) {
    using traits = typename std::istream::traits_type;
    const auto endl = traits::to_int_type(stream.widen('\n'));
    buffer.clear();
    for (auto next_char = stream.peek();
         traits::not_eof(next_char) && next_char != endl;
         next_char = stream.peek()) {
        for (auto c = stream.get(); traits::not_eof(c) && c != endl;
             c = stream.get()) {
            buffer.push_back(traits::to_char_type(c));
        }

        if (!stream) {
            spdlog::warn("an error occurred when parsing the stream");
            return false;
        }
        buffer.push_back(' '); // replace the \n with a space
    }
    return true;
}

void FreeTextReader::read_stream(Data &data, std::istream &stream) const {
    using traits = typename std::istream::traits_type;
    std::u8string buffer;

    for (; get_entry(stream, buffer);) {
        auto next_char = stream.get(); // consume the endl we just saw

        if (buffer.empty()) {
            continue;
        }

        parse_entry(data, buffer);
        if (!traits::not_eof(next_char)) {
            break;
        }
    }
}

void FreeTextReader::read_file(Data &data, const FSPath &path) const {
    std::ifstream stream(path);
    read_stream(data, stream);
}

std::string_view FreeTextReader::supported_file_extension() const noexcept {
    return "txt";
}