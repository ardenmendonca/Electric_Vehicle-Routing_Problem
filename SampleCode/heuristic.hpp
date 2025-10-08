#ifndef HEURISTIC_HPP
#define HEURISTIC_HPP
#include <vector>

struct solution
{
  std::vector<int> tour;
  int steps = 0;
  double tour_length = 0.0;
  int id = 0;
};

extern solution *best_sol;

void initialize_heuristic();
void run_heuristic();
void free_heuristic();

#endif // HEURISTIC_HPP
