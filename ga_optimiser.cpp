#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>


std::random_device rd;
std::mt19937 eng(rd());

bool validate_deps(const std::vector<int> &route, const std::vector<std::vector<int> > &deps) {
	std::vector<bool> done(route.size());
	
	for (int i = 0; i < route.size(); i++) {
		for (int j = 0; j < deps[route[i]].size(); j++) {
			if (!done[deps[route[i]][j]]) return false;
		}
		done[route[i]] = true;
	}
	
	return true;
}

std::vector<int> mutate_route(std::vector<int> base, const std::vector<std::vector<int> > &deps, int start_hold_out, int no_mutations) {
	int successful = 0;
	
	std::vector<int> old_base = base;
	while (successful < no_mutations) {
		std::uniform_int_distribution<> rnd_size(start_hold_out, base.size() - 1);

		int from = rnd_size(eng);
	
		int to = 0;
		do to = rnd_size(eng); while (to == from);

		if (from < to) {
			std::rotate(base.begin() + from, base.begin() + from + 1, base.begin() + to);
		}
		else {
			std::rotate(base.begin() + to, base.begin() + from, base.begin() + from + 1);
		}
		
		if (validate_deps(base, deps)) {
			successful++;
		} else {
			base = old_base;
		}
	}
	return base;
}

double evaluate_route(const std::vector<int> &route, const std::vector<std::vector<double> > &dist) {
	double total = 0.0;
	for (int i = 0; i < route.size() - 1; i++) {
		total += dist[route[i]][route[i+1]];
	}
	return total;
}

void topological_sort_visit(int node, const std::vector<std::vector<int> > &deps, std::vector<bool> &visited, std::vector<int> &result) {
	if (visited[node]) return;
	visited[node] = true;
	for (int i = 0; i < deps[node].size(); i++) {
		topological_sort_visit(deps[node][i], deps, visited, result);
	}
	result.push_back(node);
}

std::vector<int> topological_sort(const std::vector<std::vector<int> > &deps, const std::vector<int> &start) {
	std::vector<bool> visited(deps.size());
	std::vector<int> result;

	for (int i = 0; i < start.size(); i++) {
		visited[start[i]] = true;
		result.push_back(start[i]);
	}
	
	for (int i = 0; i < visited.size(); i++) {
		if (!visited[i])
			topological_sort_visit(i, deps, visited, result);
	}
	
	return result;
}

int main(int argc, char* argv[]) {
	std::vector<std::vector<double> > dist;
	std::ifstream source("dist_graph.txt");
	std::string input;
	double x;
	
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " pool_size no_iterations branching no_mutations [forced start vertices]" << std::endl;
		return -1;
	}
	
	std::cout << "Loading the node distance matrix..." << std::endl;
	// NxN node distance matrix
	while (std::getline(source, input)) {
		std::stringstream stream(input);
		std::vector<double> newl;
		while (stream >> x) newl.push_back(x);
		dist.push_back(newl);
	}
	source.close();
	source.open("dep_graph.txt");
	std::vector<std::vector<int> > deps;
	
	std::cout << "Loading the dependency graph..." << std::endl;
	// Dependency graph: N lines, up to N dependencies in each line
	for (int i = 0; i < dist.size(); i++) {
		std::getline(source, input);
		std::stringstream stream(input);
		std::vector<int> newdep;
		int dep;
		while (stream >> dep) newdep.push_back(dep);
		deps.push_back(newdep);
	}
	
	int pool_size = std::stoi(argv[1]);
	int no_iterations = std::stoi(argv[2]);
	int branching = std::stoi(argv[3]);
	int no_mutations = std::stoi(argv[4]);

	std::vector<int> start;
	for (int i = 0; i < argc - 5; i++) {
		start.push_back(std::stoi(argv[i + 5]));
	}
	
	// Initialize the solution pool
	std::vector<std::vector<int> > solution_pool;
	std::vector<int> initial = topological_sort(deps, start);

	for (int i = 0; i < pool_size; i++) {
		solution_pool.push_back(initial);
	}
	
	std::cout << "Optimising..." << std::endl;
	
	for (int i = 0; i < no_iterations; i++) {
		std::sort(solution_pool.begin(), solution_pool.end(),
			[&dist](std::vector<int> l, std::vector<int> r) {return evaluate_route(l, dist) < evaluate_route(r, dist);});
		double best = evaluate_route(solution_pool[0], dist);
		std::cout << "Iteration " << i << "; best " << best << std::endl;
		
		std::vector<std::vector<int> > new_solution_pool;
		for (int i = 0; i < pool_size / branching; i++) {
			for (int j = 0; j < branching; j++) {
				new_solution_pool.push_back(mutate_route(solution_pool[i], deps, start.size(), no_mutations));
			}
		}
		solution_pool = new_solution_pool;
	}
	
	std::cout << "Optimisation done, exporting..." << std::endl;
	std::ofstream output("optimiser_result.txt");
	
	for (int i = 0; i < solution_pool.size(); i++) {
		for (int j = 0; j < solution_pool[i].size(); j++) {
			output << solution_pool[i][j] << " ";
		}
		output << std::endl;
	}
	output.close();

	return 0;
}