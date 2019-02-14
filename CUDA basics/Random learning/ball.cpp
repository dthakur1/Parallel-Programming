#include <random>
#include <cmath>
#include <iostream>
#include <array>

class RandCoord {
    public:
        RandCoord() : eng(1234) {}
        double operator()() { return dist(eng); }
    private:
        std::minstd_rand eng;
        std::uniform_real_distribution<double> dist{-1, 1};
};

class Point {
    public:
        Point(int d, RandCoord &rand);
        double distance_from_origin() const;
    private:
        const int dim;
        std::vector<double> coords;
};

Point::Point(int d, RandCoord &rand) : dim(d), coords(d) {
    for (int i = 0; i < dim; i++) {
        coords.at(i) = rand();
    }
}

double
Point::distance_from_origin() const {
    double s = 0;
    for (int i = 0; i < dim; i++) {
        s += coords.at(i)*coords.at(i);
    }
    return std::sqrt(s);
}

int
main() {

    RandCoord rand;

    for (int d = 16; d <= 16; d += 1) {
        std::array<int, 100> counts{};
        double max_dist = 0;
        for (int i = 0; i < 1'000; ) {
            Point p(d, rand);
            double d = p.distance_from_origin();
            max_dist = std::max(d, max_dist);
            if (d < 1) {
                i++;
                int ind = d/.01;
                counts.at(ind)++;
            }
        }
        std::cout << "Dimension: " << d << std::endl;
        std::cout << "    max dist: " << max_dist << std::endl;
        for (int i = 0; i < 100; i++) {
            std::cout << "    " << i << ": " << counts.at(i) << std::endl;
        }
        /*
        if (d == 2) {
            d += 8;
        } else {
            d += 5;
        }
        */
    }
}
