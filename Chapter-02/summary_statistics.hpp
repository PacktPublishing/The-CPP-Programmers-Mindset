#ifndef CHAPTER_02_SUMMARY_STATISTICS_HPP
#define CHAPTER_02_SUMMARY_STATISTICS_HPP


#include <span>
#include <string>
#include <vector>


namespace ct {
struct SummaryStatistics {
    // for example
    double mean;
    double st_dev;
    double min;
    double max;
};


class Configuration {
    bool b_include_mean = true;
    bool b_include_std = true;
    bool b_include_min = true;
    bool b_include_max = true;

public:
    [[nodiscard]]
    bool include_mean() const noexcept { return b_include_mean; }

    void include_mean(bool setting) noexcept {
        b_include_mean = setting;
    }

    [[nodiscard]]
    bool include_std() const noexcept { return b_include_std; }

    void include_std(bool setting) noexcept {
        b_include_std = setting;
        // it doesn't make sense to include std deviation without
        // the mean too
        if (setting) {
            b_include_mean = setting;
        }
    }

    [[nodiscard]]
    bool include_min() const noexcept { return b_include_min; }

    void include_min(bool setting) noexcept {
        b_include_min = setting;
    }

    [[nodiscard]]
    bool include_max() const noexcept { return b_include_max; }

    void include_max(bool setting) noexcept {
        b_include_max = setting;
    }
};


std::vector<SummaryStatistics>
compute_summary_statistics(const Configuration &config, std::span<std::string> sources);
} // namespace ct

#endif //CHAPTER_02_SUMMARY_STATISTICS_HPP