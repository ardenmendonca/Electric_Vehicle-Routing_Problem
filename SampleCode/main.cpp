/********************   main.cpp   *********************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <iomanip>

#include "EVRP.hpp"
#include "heuristic.hpp"
#include "stats.hpp"
#include <cmath>
#include <iterator>
using namespace std;

double mean(double *values, int n); //  ←  add this line
double best_of_vector(double *values, int n);
double worst_of_vector(double *values, int n);

extern double *perf_of_trials; // stats.cpp
extern solution *best_sol;     // heuristic.cpp

static std::vector<std::vector<int>> all_tours; // every run's best tour

// Performance logging variables
static std::ofstream performance_log;
static int current_run = 0;

// Function to open performance log file
void open_performance_log()
{
  performance_log.open("performance_log.csv");
  if (performance_log.is_open())
  {
    performance_log << "run,evaluations,best_fitness,current_fitness,temperature,accepted" << std::endl;
  }
  else
  {
    std::cerr << "Warning: Could not open performance_log.csv for writing" << std::endl;
  }
}

// Function to log performance data
void log_performance(int run, double evaluations, double best_fitness,
                     double current_fitness, double temperature, bool accepted)
{
  if (performance_log.is_open())
  {
    performance_log << run << ","
                    << std::fixed << std::setprecision(1) << evaluations << ","
                    << std::setprecision(6) << best_fitness << ","
                    << current_fitness << ","
                    << temperature << ","
                    << (accepted ? 1 : 0) << std::endl;
  }
}

// Function to close performance log
void close_performance_log()
{
  if (performance_log.is_open())
  {
    performance_log.close();
  }
}

static void save_current_tour()
{
  std::vector<int> t(best_sol->tour.begin(), // first
                     best_sol->tour.end());  // last  (exclusive)
  all_tours.push_back(std::move(t));
}

static void write_tour(const std::vector<int> &t,
                       const std::string &name)
{
  std::ofstream f(name);
  for (std::size_t i = 0; i < t.size(); ++i)
    f << t[i] << (i + 1 == t.size() ? '\n' : ' ');
}

void start_run(int r)
{
  current_run = r;
  srand(r);
  init_evals();
  init_current_best();
  initialize_heuristic();
}

void end_run(int r)
{
  // VALIDATE THE BEST SOLUTION FOUND IN THIS RUN
  std::cout << "Validating solution for run " << r << "..." << std::endl;
  check_solution(best_sol->tour.data(), best_sol->steps);
  std::cout << "Run " << r << " solution is valid!" << std::endl;

  get_mean(r - 1, get_current_best());
  save_current_tour();
}

bool termination_condition()
{
  return get_evals() >= TERMINATION;
}

// Getter function for current run (to be used by heuristic)
int get_current_run()
{
  return current_run;
}

int main(int argc, char *argv[])
{

  problem_instance = argv[1];
  read_problem(problem_instance);
  open_stats();

  // Open performance logging
  open_performance_log();

  for (int run = 1; run <= MAX_TRIALS; ++run)
  {
    start_run(run);
    while (!termination_condition())
      run_heuristic();
    end_run(run);
    free_heuristic(); // ready for next run
  }

  close_stats();           // perf_of_trials now filled
  close_performance_log(); // Close performance log

  double best_val = perf_of_trials[0];
  double worst_val = perf_of_trials[0];
  double mean_val = mean(perf_of_trials, MAX_TRIALS);

  int best_idx = 0;
  int worst_idx = 0;
  int mean_idx = 0;
  double closest = std::numeric_limits<double>::max();

  for (int i = 1; i < MAX_TRIALS; ++i)
  {
    if (perf_of_trials[i] < best_val)
    {
      best_val = perf_of_trials[i];
      best_idx = i;
    }
    if (perf_of_trials[i] > worst_val)
    {
      worst_val = perf_of_trials[i];
      worst_idx = i;
    }
    double d = std::fabs(perf_of_trials[i] - mean_val);
    if (d < closest)
    {
      closest = d;
      mean_idx = i;
    }
  }

  write_tour(all_tours[best_idx], "tour_best.txt");
  write_tour(all_tours[worst_idx], "tour_worst.txt");
  write_tour(all_tours[mean_idx], "tour_mean.txt");

  // FINAL VALIDATION OF THE THREE KEY SOLUTIONS
  std::cout << "\n=== FINAL VALIDATION ===" << std::endl;

  std::cout << "\n=== RESULTS ===" << std::endl;
  std::cout << " Best  : " << best_val << '\n';
  std::cout << " Worst : " << worst_val << '\n';
  std::cout << " Mean  : " << mean_val << '\n';
  std::cout << "\nPerformance data saved to: performance_log.csv" << std::endl;

  free_stats();
  free_EVRP();
  return 0;
}