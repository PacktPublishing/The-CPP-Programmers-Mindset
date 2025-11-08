#ifndef CHAPTER_02_TEMPERATURE_SENSORS_HPP
#define CHAPTER_02_TEMPERATURE_SENSORS_HPP


#include <string_view>
#include <span>

namespace ct {
class TempSensor {
public:
    // this was defaulted in the text, this is moved to the
    // corresponding cpp file.
    virtual ~TempSensor();

    [[nodiscard]]
    virtual std::string_view id() const noexcept = 0;

    [[nodiscard]]
    virtual float temperature_kelvin() const noexcept = 0;
};


void check_sensors(std::span<const TempSensor *> sensors, float threshold);
}

#endif //CHAPTER_02_TEMPERATURE_SENSORS_HPP