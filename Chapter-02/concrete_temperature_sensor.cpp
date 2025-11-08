#include "concrete_temperature_sensor.hpp"


ct::ConcreteTemperatureSensor::~ConcreteTemperatureSensor() = default;


std::string_view ct::ConcreteTemperatureSensor::id() const noexcept {
    return m_id;
}

float ct::ConcreteTemperatureSensor::temperature_kelvin() const noexcept {
    return m_temperature;
}