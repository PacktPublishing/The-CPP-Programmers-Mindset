#include "temperature_sensors.hpp"

#include <ranges>

#include <gtest/gtest.h>

#include "concrete_temperature_sensor.hpp"


namespace {
class TemperatureSensors : public ::testing::Test {
    std::vector<ct::ConcreteTemperatureSensor> sensors_;
    std::vector<const ct::TempSensor *> sensor_pointers_;

protected:
    std::span<const ct::TempSensor *> get_sensors(std::initializer_list<float> temperatures) noexcept {
        sensors_.reserve(temperatures.size());

        for (auto &&[i, temp]: std::views::enumerate(temperatures)) {
            sensors_.emplace_back(std::to_string(i), temp);
        }

        sensor_pointers_.reserve(sensors_.size());
        for (auto &sensor: sensors_) {
            sensor_pointers_.push_back(&sensor);
        }

        return sensor_pointers_;
    }
};
}


TEST_F(TemperatureSensors, AllGoodSensors) {
    auto sensors = get_sensors({293.15, 0.0, 150.0, 299.15});

    EXPECT_NO_THROW(ct::check_sensors(sensors, 300.00));
}

TEST_F(TemperatureSensors, BadSensorThrow) {
    auto sensors = get_sensors({293.15, 0.0, 150.0, 299.15});

    EXPECT_THROW(ct::check_sensors(sensors, 295.0), std::runtime_error);
}