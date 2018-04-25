#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
	std::vector<std::vector<double> > dist;
	std::ifstream source("dist.txt");
	std::string input;
	double x;
	
	while (std::getline(source, input)) {
		std::stringstream stream(input);
		std::vector<double> newl;
		while (stream >> x) newl.push_back(x);
		dist.push_back(newl);
	}
	
	std::vector<std::vector<int> > prev;
	for (int i = 0; i < dist.size(); i++) {
		prev.push_back(std::vector<int>());
		for (int j = 0; j < dist.size(); j++) {
			prev[i].push_back(j);
		}
	}
	
	for (int k = 0; k < dist.size(); k++) {
		for (int i = 0; i < dist.size(); i++) {
			for (int j = 0; j < dist.size(); j++) {
				if (dist[i][j] > dist[i][k] + dist[k][j]) {
					dist[i][j] = dist[i][k] + dist[k][j];
					prev[i][j] = prev[i][k];
				}
			}
		}
		std::cout << k << "/" << dist.size() << std::endl;
	}
	
	std::ofstream output("dist_res.txt");
	for (int i = 0; i < dist.size(); i++) {
		for (int j = 0; j < dist.size(); j++) {
			output << dist[i][j] << " ";
		}
		output << std::endl;
	}
	output.close();
	
	output.open("prev_res.txt");
	for (int i = 0; i < prev.size(); i++) {
		for (int j = 0; j < prev.size(); j++) {
			output << prev[i][j] << " ";
		}
		output << std::endl;
	}
	output.close();
	
	return 0;
}