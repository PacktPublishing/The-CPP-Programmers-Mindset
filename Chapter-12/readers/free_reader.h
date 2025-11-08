#ifndef DUCKIES_FREE_READER_H
#define DUCKIES_FREE_READER_H

#include "file_reader.h"
#include <iosfwd>

#define RE_MONTHS                                                              \
	"(Jan(?:uary)?"                                                             \
	"|Feb(?:ruary)?"                                                           \
	"|Mar(?:ch)?"                                                              \
	"|Apr(?:il)?"                                                              \
	"|May|June?"                                                               \
	"|July?"                                                                   \
	"|Aug(?:ust)?"                                                             \
	"|Sep(?:tember)?"                                                          \
	"|Oct(?:ober)?"                                                            \
	"|Nov(?:ember)?"                                                           \
	"|Dec(?:ember)?)"

#define RE_OR "|"
#define RE_YEAR "(\\d{4})"
#define RE_DAY "(\\d{2})"
#define RE_MONTH "(\\d{2})"
#define RE_DATE_SEP "(?:-|\\/)"
#define RE_WS "\\s"

#define RE_ISO RE_YEAR RE_DATE_SEP RE_MONTH RE_DATE_SEP RE_DAY
#define RE_DMY RE_DAY RE_WS RE_MONTHS RE_WS RE_YEAR
#define RE_MDY RE_MONTHS RE_WS RE_DAY RE_WS RE_YEAR

#define RE_DATE RE_ISO RE_OR RE_DMY RE_OR RE_MDY

#define RE_LAT_DEG "(?:(?<lat_deg>\\d{1,2})(?:\u00b0)?\\s*)?"
#define RE_LAT_MIN "(?:(?<lat_min>\\d{1,2})'\\s*)"
#define RE_LAT_SEC "(?<lat_sec>\\d{1,2}(?:\\.\\d+)?)(?:\"|'')\\s*"
#define RE_LAT_NS "(?<lat_card>[NS])"

#define RE_DMS_LAT RE_LAT_DEG RE_LAT_MIN RE_LAT_SEC RE_LAT_NS

#define RE_LON_DEG "(?:(?<lon_deg>\\d{1,3})(?:\u00b0)?\\s*)?"
#define RE_LON_MIN "(?:(?<lon_min>\\d{1,2})'\\s*)?"
#define RE_LON_SEC "(?<lon_sec>\\d{1,2}(?:\\.\\d+)?)(?:\"|'')\\s*"
#define RE_LON_EW "(?<lon_card>[EW])"

#define RE_DMS_LON RE_LON_DEG RE_LON_MIN RE_LON_SEC RE_LON_EW

#define RE_DD_LAT "(?<dd_lat>[+\\-]?\\d{1,2}\\.\\d+)(?:\u00b0)?"
#define RE_DD_LON "(?<dd_lon>[+\\-]?\\d{1,3}\\.\\d+)(?:\u00b0)?"

#define RE_CO_SEP "(?:,|\\s)\\s*"
#define RE_DMS RE_DMS_LAT RE_CO_SEP RE_DMS_LON
#define RE_DD RE_DD_LAT RE_CO_SEP RE_DD_LON
#define RE_COORD RE_DMS RE_OR RE_DD

namespace duckies {
// Groups for date pattern
inline constexpr int iso_date_year_group = 1;
inline constexpr int iso_date_month_group = 2;
inline constexpr int iso_date_day_group = 3;
inline constexpr int dmy_date_day_group = 4;
inline constexpr int dmy_date_month_group = 5;
inline constexpr int dmy_date_year_group = 6;
inline constexpr int mdy_date_month_group = 7;
inline constexpr int mdy_date_day_group = 8;
inline constexpr int mdy_date_year_group = 9;

class FreeTextReader : public FileReader {
public:
    using typename FileReader::Data;
    using typename FileReader::FSPath;

    void read_stream(Data &data, std::istream &view) const;

    void read_file(Data &data, const FSPath &path) const override;

    std::string_view supported_file_extension() const noexcept override;
};
} // namespace duckies

#endif // DUCKIES_FREE_READER_H