#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

std::vector<double> xs;
std::vector<double> ys;
std::vector<double> dists;

double get_closest_distance(double x, double y) {
	double min_dist = 1e50;

	for (int i = 0; i < xs.size(); i++) {
		double dist = dists[i] + sqrt(pow(xs[i] - x, 2) + pow(ys[i] - y, 2)) / 12000.0;
		if (dist < min_dist) min_dist = dist;
	}

	return min_dist;
}

int main() {

	std::ifstream source("locations.txt");

	double x, y, dist;
	while (source >> x >> y >> dist) {
		xs.push_back(x);
		ys.push_back(y);
		dists.push_back(dist);
	}

	std::ifstream sought("sought.txt");
	std::ofstream result("result.txt");
	while (sought >> x >> y) {
		result << get_closest_distance(x, y) << std::endl;
	}

	return 0;
}