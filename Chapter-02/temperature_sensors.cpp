#include "temperature_sensors.hpp"

#include <format>
#include <stdexcept>


ct::TempSensor::~TempSensor() = default;

void ct::check_sensors(std::span<const TempSensor *> sensors, float threshold) {
    for (const auto &sensor: sensors) {
        auto temp = sensor->temperature_kelvin();
        if (temp > threshold) {
            throw std::runtime_error(
                std::format("Sensor {} reports temperature {}", sensor->id(), temp)
            );
        }
    }
}