#ifndef STROBEALIGN_TIMER_HPP
#define STROBEALIGN_TIMER_HPP

#include <sys/time.h>
inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

#include <chrono>

// A timer that automatically starts on construction
class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {
    }

    // Return time elapsed since construction
    std::chrono::duration<double> duration() const {
        return std::chrono::high_resolution_clock::now() - start_time;
    }

    std::chrono::duration<double>::rep elapsed() const {
        return duration().count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

#endif
