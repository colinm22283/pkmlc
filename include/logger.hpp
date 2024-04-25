#pragma once

namespace LoggerConfig {
    static constexpr bool enable_debug   = true;
    static constexpr bool enable_status  = true;
    static constexpr bool enable_warning = true;
    static constexpr bool enable_error   = true;
}

namespace _LoggerPrivate {
    enum logger_level_t { DEBUG, STATUS, WARNING, ERROR };

    template<logger_level_t level>
    static constexpr const char * level_to_string = (level == DEBUG ? "DEBUG" : (level == STATUS ? "STATUS" : (level == WARNING ? "WARNING" : (level == ERROR ? "ERROR" : "INVALID"))));

    template<logger_level_t level>
    static constexpr bool level_enabled = (level == DEBUG && LoggerConfig::enable_debug) || (level == STATUS && LoggerConfig::enable_status) || (level == WARNING && LoggerConfig::enable_warning) || (level == ERROR && LoggerConfig::enable_error);

    template<logger_level_t level>
    static constexpr const char * level_prefix = (level == DEBUG ? "\033[32m" : (level == STATUS ? "\033[34m" : (level == WARNING ? "\033[93m" : (level == ERROR ? "\033[91m" : "INVALID"))));

    template<logger_level_t level>
    static constexpr const char * level_postfix = "\033[0m\n";

    template<logger_level_t level>
    class Logger {
    protected:
        template<bool output_newline>
        struct SubLogger {
            inline ~SubLogger() noexcept { if constexpr (output_newline) std::cout << level_postfix<level>; }

            template<typename T>
            inline SubLogger<false> operator<<(T & v) const noexcept {
                if constexpr (level_enabled<level>) {
                    std::cout << v; return SubLogger<false>();
                }
            }
            template<typename T>
            inline SubLogger<false> operator<<(T && v) const noexcept {
                if constexpr (level_enabled<level>) {
                    std::cout << v; return SubLogger<false>();
                }
            }
        };

        const char * name;

    public:
        constexpr Logger(const char * _name) noexcept: name(_name) { }

        template<typename T>
        inline SubLogger<true> operator<<(T & v) const noexcept {
            if constexpr (level_enabled<level>) {
                std::cout << level_prefix<level> << "[ " << level_to_string<level> << "  '" << name << "' ] " << v;
                return SubLogger<true>();
            }
        }
        template<typename T>
        inline SubLogger<true> operator<<(T && v) const noexcept {
            if constexpr (level_enabled<level>) {
                std::cout << level_prefix<level> << "[ " << level_to_string<level> << "  '" << name << "' ] " << v;
                return SubLogger<true>();
            }
        }

        consteval auto debug() const noexcept { return _LoggerPrivate::Logger<_LoggerPrivate::DEBUG>(name); }
        consteval auto status() const noexcept { return _LoggerPrivate::Logger<_LoggerPrivate::STATUS>(name); }
        consteval auto warning() const noexcept { return _LoggerPrivate::Logger<_LoggerPrivate::WARNING>(name); }
        consteval auto error() const noexcept { return _LoggerPrivate::Logger<_LoggerPrivate::ERROR>(name); }
    };
}

using Logger = _LoggerPrivate::Logger<_LoggerPrivate::STATUS>;