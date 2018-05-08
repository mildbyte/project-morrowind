#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>


std::random_device rd;
std::mt19937 eng{ 34567 };

bool validate_deps(const std::vector<int> &route, const std::vector<std::vector<int> > &deps) {
	// Check that the dependency graph is satisfied: maintain a mask of nodes that we've visited
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
	// Random generator for the old and new position of a random point in the route (we're leaving first start_hold_out items alone)

	std::uniform_int_distribution<> rnd_segment_length(1, 30);
	while (successful < no_mutations) {
		std::vector<int> old_base = base;
		int segment_length = rnd_segment_length(eng);
		std::uniform_int_distribution<> rnd_size(start_hold_out, base.size() - segment_length);
		int from = rnd_size(eng);

		int to = 0;
		do to = rnd_size(eng); while (to == from);

		// Move (not swap!) the item(s) at "from" up/down towards "to"
		if (from < to) {
			std::rotate(base.begin() + from, base.begin() + from + segment_length, base.begin() + to + segment_length);
		}
		else {
			std::rotate(base.begin() + to, base.begin() + from, base.begin() + from + segment_length);
		}

		if (validate_deps(base, deps)) {
			successful++;
		}
		else {
			// If dependencies not satisfied, throw this attempt away
			base = old_base;
		}
	}
	return base;
}

double evaluate_route(const std::vector<int> &route, const std::vector<std::vector<double> > &dist) {
	double total = 0.0;
	for (int i = 0; i < route.size() - 1; i++) {
		total += dist[route[i]][route[i + 1]];
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

template <class T> void read_matrix(std::vector<std::vector<T> > &matrix, std::ifstream &source) {
	std::string input;
	while (std::getline(source, input)) {
		std::stringstream stream(input);
		std::vector<T> newl;
		T x;
		while (stream >> x) newl.push_back(x);
		matrix.push_back(newl);
	}
}

int main(int argc, char* argv[]) {

	if (argc < 10) {
		std::cerr << "Usage: " << argv[0] << " dependency_graph_file distance_graph_file pool_size branching start_no_mutations end_no_mutations mutations_switch start_node_1 start_node_2 [initial route]" << std::endl;
		return -1;
	}

	std::vector<std::vector<double> > dist;
	std::ifstream source(argv[2]);
	std::cout << "Loading the recall-allowed distance matrix..." << std::endl;
	read_matrix<double>(dist, source);
	source.close();

	source.open(argv[1]);
	std::vector<std::vector<int> > deps;

	std::cout << "Loading the dependency graph..." << std::endl;
	// Dependency graph: N lines, up to N dependencies in each line
	read_matrix<int>(deps, source);
	source.close();

	int pool_size = std::stoi(argv[3]);
	int branching = std::stoi(argv[4]);

	int no_mutations_start = std::stoi(argv[5]);
	int no_mutations_end = std::stoi(argv[6]);
	int mutations_switch = std::stoi(argv[7]);

	std::vector<int> start;
	start.push_back(std::stoi(argv[8]));
	start.push_back(std::stoi(argv[9]));

	// Initialize the solution pool
	std::vector<std::vector<int> > solution_pool;

	std::vector<int> initial;

	if (argc == 11) {
		std::cout << "Loading the initial route..." << std::endl;
		source.open(argv[10]);
		int n;
		while (source >> n) initial.push_back(n);
	}
	else {
		initial = topological_sort(deps, start);
	}

	for (int i = 0; i < pool_size; i++) {
		solution_pool.push_back(initial);
	}

	std::cout << "Optimising..." << std::endl;

	double prev_best = 1e10;
	for (int no_mutations = no_mutations_start; no_mutations >= no_mutations_end; no_mutations--) {	
		int times_prev_matched = 0;
		int iteration = 0;

		std::cout << "Number of mutations " << no_mutations << std::endl;

		while (times_prev_matched < mutations_switch) {
			std::vector<std::vector<int> > new_solution_pool;
			for (int i = 0; i < pool_size / branching; i++) {
				for (int j = 0; j < branching; j++) {
					new_solution_pool.push_back(mutate_route(solution_pool[i], deps, start.size(), no_mutations));
				}
			}

			std::sort(new_solution_pool.begin(), new_solution_pool.end(),
				[&, dist](std::vector<int> l, std::vector<int> r) {
				return evaluate_route(l, dist)
					< evaluate_route(r, dist);});
			double best = evaluate_route(solution_pool[0], dist);

			if (best < prev_best) {
				solution_pool = new_solution_pool;
				prev_best = best;
			}
			else {
				times_prev_matched++;
			}

			std::cout << "Iteration " << iteration++ << "; best " << prev_best << std::endl;
		}
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