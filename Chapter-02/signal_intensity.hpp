#ifndef CHAPTER_02_SIGNAL_INTENSITY_HPP
#define CHAPTER_02_SIGNAL_INTENSITY_HPP


namespace ct {
struct Pos {
    int x;
    int y;

    constexpr Pos(int x_, int y_) : x(x_), y(y_) {
    }
};


double compute_signal_intensity(int x, int y);
} // namespace ct


#endif //CHAPTER_02_SIGNAL_INTENSITY_HPP