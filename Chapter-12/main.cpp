#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>

#include <boost/program_options.hpp>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

// headers for our logic
#include "clustering/clustering.h"
#include "readers/file_reader.h"

using namespace duckies;
namespace po = boost::program_options;

static po::variables_map parse_config(int argc, const char *const*argv);

static void setup_signals();

static void setup_logging(const po::variables_map &args);

static void setup_threading(const po::variables_map &args);

static void run_and_report(const po::variables_map &args);

int main(int argc, const char *const*argv) {
    setup_signals();

    auto args = parse_config(argc, argv);

    setup_logging(args);
    setup_threading(args);

    run_and_report(args);

    return EXIT_SUCCESS;
}

static const char version_string[] = "duckies Version 1.0.0\n";

static const char program_synopsis_string[]
        = "duckies [options] file1 file2 ... \n";
// clang-format off
static const char program_usage_string[] = 
  R"doc(Parse rubber ducky observation files and perform a clustering analysis to
find out where are very polite dubugging assistants are coming from.

)doc";
// clang-format on

static constexpr int unlimited_arguments = -1;

po::variables_map parse_config(int argc, const char *const*argv) {
    using PathVec = std::vector<std::filesystem::path>;

	// clang-format off
  po::options_description options { "Options" };
  options.add_options()
      ("help,h", po::bool_switch(), "produce_help_message")
      ("version", po::bool_switch(), "print the version and exit")
      ("verbose,v", po::bool_switch(), "print logging information to the terminal")
      ("jobs,j", po::value<int>(), "set the maximum number of jobs")
      ("kmin", po::value<size_t>()->default_value(2), 
       "minimum number of clusters to find")
      ("kmax", po::value<size_t>()->default_value(256), 
       "maximum number of clusters to find")
      ("max_iter", po::value<size_t>()->default_value(100), 
       "maximum number of iterations to find each clustering")
      ("repetitions", po::value<size_t>()->default_value(5),
       "number of trial clusterings for each value of k")
      ("paths", po::value<PathVec>(), "paths to process");

  po::positional_options_description positional;
  positional.add("paths", unlimited_arguments);
    // clang-format on

    po::variables_map cli_args;
    try {
        // parse the command line arguments
        po::command_line_parser parser(argc, argv);
        parser.options(options);
        parser.positional(positional);
        po::store(parser.run(), cli_args);

        // parse environment variables
        po::store(po::parse_environment(options, "DUCKIES_"), cli_args);

        po::notify(cli_args);
    } catch (std::exception &exc) {
        std::cerr << exc.what() << "\n\n"
                << program_synopsis_string << options << '\n';
        std::exit(EXIT_FAILURE);
    }

    if (cli_args["version"].as<bool>()) {
        std::cerr << version_string;
        std::exit(EXIT_SUCCESS);
    }

    if (cli_args["help"].as<bool>()) {
        std::cerr << program_synopsis_string << program_usage_string << options
                << '\n';
        std::exit(EXIT_SUCCESS);
    }

    if (cli_args["paths"].empty()) {
        std::cerr << program_synopsis_string << options << '\n';
        std::exit(EXIT_FAILURE);
    }

    return cli_args;
}

void setup_logging(const po::variables_map &args) {
    auto console_logger = spdlog::stderr_logger_mt("console");
    console_logger->set_pattern("[%Y-%m-%d %T] [%L] %v");
    spdlog::set_default_logger(console_logger);

    console_logger->set_level(spdlog::level::warn);
    if (args["verbose"].as<bool>()) {
        console_logger->set_level(spdlog::level::info);
    }

    spdlog::info("Fininshed setting up logger");
}

void setup_threading(const po::variables_map &args) {
    auto num_threads = omp_get_max_threads();

    if (args.count("jobs")) {
        auto requested_threads = args["jobs"].as<int>();
        omp_set_num_threads(std::min(num_threads, requested_threads));
    } else {
        // set a very conservative default
        omp_set_num_threads(2);
    }

    spdlog::info("We will use {} threads", num_threads);
}

static std::atomic_int signal_count = 0;

extern "C" void sigint_handler(int signum) {
    auto current = signal_count.fetch_add(1);
    if (current > 0) {
        std::exit(1);
    }
}

bool check_interrupts() noexcept {
    auto interrupt = signal_count.load() > 0;
    if (interrupt) {
        std::cerr << "Received Ctrl-C event, terminating gracefully\n"
                << "Press Ctrl-C again to exit immediately" << std::endl;
    }
    return interrupt;
}

void setup_signals() { std::signal(SIGINT, &sigint_handler); }

template<typename F>
void handle_errors(F &&func) noexcept {
    try {
        func();
    } catch (const std::exception &exc) {
        spdlog::critical(exc.what());
        std::exit(EXIT_FAILURE);
    }
}

RubberDuckData get_data(const po::variables_map &args) {
    using PathVec = std::vector<std::filesystem::path>;
    const auto readers = get_readers();

    RubberDuckData data;

    for (const auto &path: args["paths"].as<PathVec>()) {
        spdlog::info("processing file {}", path.c_str());
        if (!std::filesystem::exists(path)) {
            spdlog::warn(
                "path argument {} does not exist, skipping", path.c_str());
            continue;
        }

        auto ext = path.extension().string();
        if (ext.empty()) {
            spdlog::warn("path {} does not have an extension", path.c_str());
            continue;
        }

        auto search = ext.substr(1);
        auto reader = readers.find(search);
        if (reader == readers.end()) {
            spdlog::warn(
                "no file reader for files with \"{}\" extension", search);
            continue;
        }

        reader->second->read_file(data, path);
    }

    return data;
}

void run_and_report(const po::variables_map &args) {
    handle_errors([&] {
        auto data = get_data(args);

        auto k_min = args["kmin"].as<size_t>();
        auto k_max = args["kmax"].as<size_t>();
        auto num_repetitions = args["repetitions"].as<size_t>();
        auto max_iterations = args["max_iter"].as<size_t>();

        auto results = compute_clusters(
            data, k_min, k_max, num_repetitions, max_iterations);

        for (const auto &cluster: results) {
            std::cout << std::format("{: 7.3f} {: 8.3f} {}\n",
                                     cluster.position.latitude, cluster.position.longitude,
                                     cluster.num_points);
        }
    });
}