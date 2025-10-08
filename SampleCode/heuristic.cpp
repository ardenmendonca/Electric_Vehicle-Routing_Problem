#include "heuristic.hpp"
#include "EVRP.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <thread>
#include <unordered_set>
#include <queue>
#include <map>
#include <set>

/* =========================================================
Enhanced Hybrid EVRP Algorithm with SA Framework
Combines: VNS + Adaptive Large Neighborhood Search +
Tabu Search + Path Relinking + Enhanced Repair Mechanisms +
Multi-Criteria Charging Station Selection + Chain Exchange
All within Simulated Annealing acceptance framework
=========================================================*/

// Forward declarations
static void make_initial(solution &);
void repair_in_place(solution &);
extern void log_performance(int run, double evaluations, double best_fitness,
                            double current_fitness, double temperature, bool accepted);
extern int get_current_run();

// Global helpers from EVRP.cpp
extern struct node *node_list;
extern bool *charging_station;
extern double **distances;
extern bool termination_condition(void);
extern double get_distance(int, int);
extern int get_customer_demand(int);
extern double get_energy_consumption(int, int);

// SA Parameters
static constexpr double T_INIT = 5.0;   // Initial temperature
static constexpr double T_END = 0.1;    // Final temperature
static constexpr double ALPHA = 0.9;    // Cooling rate
static constexpr double PROGRESS = 2.0; // Progress report interval (seconds)

// Enhanced parameters with better scaling
static std::mt19937 rng;
static std::uniform_real_distribution<double> URD(0.0, 1.0);
static constexpr int BASE_TABU_SIZE = 10;
static constexpr int BASE_ITERATIONS_WITHOUT_IMPROVEMENT = 10;
static constexpr int VNS_MAX_NEIGHBORHOODS = 6;
static constexpr double ALNS_DESTROY_RATE = 0.25;
static constexpr int ELITE_SOLUTIONS = 10;
static constexpr double BASE_ENERGY_SAFETY_MARGIN = 0.05;

// Global variables
solution *best_sol = nullptr;
static solution cur_sol, cand_sol;
static std::vector<solution> elite_solutions;
static std::queue<std::pair<int, int>> tabu_list;
static std::unordered_set<std::string> tabu_set;
static int iterations_without_improvement = 0;

// Enhanced variables for variations
static int current_tabu_size;
static int current_max_iterations_without_improvement;
static double current_energy_safety_margin;
static std::uniform_real_distribution<double> tie_breaker_dist(-0.001, 0.001);
static std::uniform_int_distribution<int> tabu_variation(-5, 5);
static std::uniform_int_distribution<int> iteration_variation(-5, 5);
static std::uniform_real_distribution<double> margin_variation(-0.005, 0.005);

// SA acceptance function
static bool accept(double current_cost, double candidate_cost, double temperature)
{
  if (candidate_cost <= current_cost)
    return true;

  if (temperature <= 1e-10)
    return false;

  double delta = candidate_cost - current_cost;
  double probability = std::exp(-delta / temperature);
  return URD(rng) < probability;
}

/* =========================================================
Adaptive Parameter Management System
=========================================================*/
class AdaptiveParameterManager
{
private:
  struct OperatorStats
  {
    int applications = 0;
    int improvements = 0;
    double total_improvement = 0.0;
    double success_rate = 0.0;
    double avg_improvement = 0.0;
    double weight = 1.0;
  };

  std::map<std::string, OperatorStats> operator_stats;
  int update_frequency = 50;
  int iterations_since_update = 0;

public:
  void record_application(const std::string &op_name, double improvement)
  {
    auto &stats = operator_stats[op_name];
    stats.applications++;
    if (improvement > 1e-9)
    {
      stats.improvements++;
      stats.total_improvement += improvement;
    }
  }

  void update_parameters()
  {
    for (auto &[name, stats] : operator_stats)
    {
      if (stats.applications > 0)
      {
        stats.success_rate = (double)stats.improvements / stats.applications;
        stats.avg_improvement = stats.total_improvement /
                                std::max(1, stats.improvements);
        // Update weight based on performance
        stats.weight = stats.success_rate * 0.7 +
                       std::min(1.0, stats.avg_improvement / 100.0) * 0.3;
      }
    }
  }

  std::vector<std::string> get_operator_ranking()
  {
    std::vector<std::pair<std::string, double>> rankings;
    for (const auto &[name, stats] : operator_stats)
    {
      rankings.push_back({name, stats.weight});
    }

    std::sort(rankings.begin(), rankings.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    std::vector<std::string> result;
    for (const auto &[name, weight] : rankings)
    {
      result.push_back(name);
    }
    return result;
  }

  bool should_update()
  {
    return ++iterations_since_update >= update_frequency;
  }

  void reset_update_counter()
  {
    iterations_since_update = 0;
  }

  double get_operator_weight(const std::string &op_name)
  {
    return operator_stats[op_name].weight;
  }
};

static AdaptiveParameterManager param_manager;

// Utility functions
static inline bool is_station(int n) { return charging_station[n]; }
static void copy(const solution &src, solution &dst) { dst = src; }
static inline bool is_customer(int n) { return n > 0 && n <= NUM_OF_CUSTOMERS && !is_station(n); }

// Enhanced energy checking with dynamic safety margin
static bool has_sufficient_energy(double current_battery, int from, int to)
{
  double required = get_energy_consumption(from, to);
  return current_battery >= required * (1.0 + current_energy_safety_margin);
}

// Modified distance function with tiny random perturbation for tie-breaking
static double get_distance_with_variation(int from, int to)
{
  double base_distance = get_distance(from, to);
  if (URD(rng) < 0.1)
  {
    base_distance += tie_breaker_dist(rng);
  }
  return std::max(0.0, base_distance);
}

// Get route segments
static std::vector<std::pair<int, int>> get_route_segments(const solution &s)
{
  std::vector<std::pair<int, int>> segments;
  int start = 0;
  for (int i = 1; i < s.steps; ++i)
  {
    if (s.tour[i] == DEPOT)
    {
      if (i - start > 1)
      {
        segments.push_back({start, i});
      }
      start = i;
    }
  }
  return segments;
}

// Enhanced tabu management
static bool is_tabu_move(int i, int j)
{
  std::string move = std::to_string(i) + "-" + std::to_string(j);
  return tabu_set.find(move) != tabu_set.end();
}

static void add_tabu_move(int i, int j)
{
  std::string move = std::to_string(i) + "-" + std::to_string(j);
  tabu_list.push({i, j});
  tabu_set.insert(move);

  if (tabu_list.size() > static_cast<size_t>(current_tabu_size))
  {
    auto old_move = tabu_list.front();
    tabu_list.pop();
    std::string old_move_str = std::to_string(old_move.first) + "-" + std::to_string(old_move.second);
    tabu_set.erase(old_move_str);
  }
}

// Update dynamic parameters
static void update_dynamic_parameters()
{
  current_tabu_size = BASE_TABU_SIZE + tabu_variation(rng);
  current_max_iterations_without_improvement = BASE_ITERATIONS_WITHOUT_IMPROVEMENT + iteration_variation(rng);
  current_energy_safety_margin = BASE_ENERGY_SAFETY_MARGIN + margin_variation(rng);

  current_tabu_size = std::max(80, std::min(120, current_tabu_size));
  current_max_iterations_without_improvement = std::max(140, std::min(160, current_max_iterations_without_improvement));
  current_energy_safety_margin = std::max(0.04, std::min(0.06, current_energy_safety_margin));
}

/* =========================================================
Multi-Criteria Charging Station Selection
=========================================================*/
struct ChargingStationCandidate
{
  int station_id;
  double detour_cost;
  double remaining_battery_after;
  double future_accessibility_score;

  double composite_score() const
  {
    return detour_cost * 0.6 +
           (1.0 - remaining_battery_after / BATTERY_CAPACITY) * 0.3 +
           (1.0 - future_accessibility_score) * 0.1;
  }
};

static int find_smart_charging_station(int from, int to,
                                       double current_battery,
                                       const std::vector<int> &remaining_customers)
{
  std::vector<ChargingStationCandidate> candidates;

  for (int station = 0; station < ACTUAL_PROBLEM_SIZE; ++station)
  {
    if (is_station(station) && station != from && station != to)
    {
      double energy_to_station = get_energy_consumption(from, station);
      double energy_from_station = get_energy_consumption(station, to);

      if (current_battery >= energy_to_station * (1.0 + current_energy_safety_margin) &&
          BATTERY_CAPACITY >= energy_from_station * (1.0 + current_energy_safety_margin))
      {

        ChargingStationCandidate candidate;
        candidate.station_id = station;
        candidate.detour_cost = get_distance(from, station) +
                                get_distance(station, to) -
                                get_distance(from, to);
        candidate.remaining_battery_after = BATTERY_CAPACITY - energy_from_station;

        // Calculate future accessibility score
        candidate.future_accessibility_score = 0.0;
        for (int customer : remaining_customers)
        {
          double energy_to_customer = get_energy_consumption(station, customer);
          if (BATTERY_CAPACITY >= energy_to_customer)
          {
            candidate.future_accessibility_score += 1.0;
          }
        }
        if (!remaining_customers.empty())
        {
          candidate.future_accessibility_score /= remaining_customers.size();
        }

        candidates.push_back(candidate);
      }
    }
  }

  if (candidates.empty())
    return -1;

  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b)
            {
              return a.composite_score() < b.composite_score();
            });

  // Randomly select from top 3 candidates if available
  int num_candidates = std::min(3, static_cast<int>(candidates.size()));
  int selected_idx = rng() % num_candidates;
  return candidates[selected_idx].station_id;
}

// Fallback to original simple charging station finder
static int find_best_charging_station(int from, int to, double current_battery)
{
  std::vector<int> empty_customers;
  int result = find_smart_charging_station(from, to, current_battery, empty_customers);
  if (result != -1)
    return result;

  // Original fallback logic
  std::vector<std::pair<int, double>> candidates;
  for (int station = 0; station < ACTUAL_PROBLEM_SIZE; ++station)
  {
    if (is_station(station) && station != from && station != to)
    {
      double energy_to_station = get_energy_consumption(from, station);
      double energy_from_station = get_energy_consumption(station, to);

      if (current_battery >= energy_to_station * (1.0 + current_energy_safety_margin) &&
          BATTERY_CAPACITY >= energy_from_station * (1.0 + current_energy_safety_margin))
      {
        double detour_cost = get_distance(from, station) + get_distance(station, to) - get_distance(from, to);
        candidates.push_back({station, detour_cost});
      }
    }
  }

  if (candidates.empty())
    return -1;

  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b)
            { return a.second < b.second; });

  double best_cost = candidates[0].second;
  std::vector<int> best_candidates;

  for (const auto &candidate : candidates)
  {
    if (candidate.second <= best_cost * 1.01)
    {
      best_candidates.push_back(candidate.first);
    }
    else
    {
      break;
    }
  }

  return best_candidates[rng() % best_candidates.size()];
}

/* =========================================================
Enhanced Feasibility and Repair Functions
=========================================================*/
static bool is_feasible_detailed(const std::vector<int> &tour, bool verbose = false)
{
  double load = MAX_CAPACITY;
  double battery = BATTERY_CAPACITY;

  for (size_t i = 0; i + 1 < tour.size(); ++i)
  {
    int from = tour[i], to = tour[i + 1];

    if (is_customer(to))
    {
      load -= get_customer_demand(to);
      if (load < -1e-6)
      {
        if (verbose)
          std::cout << "Capacity violation at customer " << to << std::endl;
        return false;
      }
    }

    double consumption = get_energy_consumption(from, to);
    if (battery < consumption - 1e-6)
    {
      if (verbose)
        std::cout << "Energy violation from " << from << " to " << to << std::endl;
      return false;
    }

    battery -= consumption;

    if (to == DEPOT)
    {
      load = MAX_CAPACITY;
      battery = BATTERY_CAPACITY;
    }
    else if (is_station(to))
    {
      battery = BATTERY_CAPACITY;
    }
  }
  return true;
}

void repair_in_place(solution &s)
{
  if (s.tour.empty() || s.tour[0] != DEPOT)
  {
    s.tour.clear();
    s.tour.push_back(DEPOT);
    s.steps = 1;
    s.tour_length = 0.0;
    return;
  }

  if (s.tour.back() != DEPOT)
  {
    s.tour.push_back(DEPOT);
  }

  std::vector<int> repaired_tour;
  repaired_tour.push_back(DEPOT);

  double current_load = MAX_CAPACITY;
  double current_battery = BATTERY_CAPACITY;

  // Collect remaining customers for smart charging decisions
  std::vector<int> remaining_customers;
  for (size_t i = 1; i < s.tour.size(); ++i)
  {
    if (is_customer(s.tour[i]))
    {
      remaining_customers.push_back(s.tour[i]);
    }
  }

  for (size_t i = 1; i < s.tour.size(); ++i)
  {
    int next_node = s.tour[i];
    int current_node = repaired_tour.back();

    if (next_node == DEPOT && current_node == DEPOT)
    {
      continue;
    }

    if (is_customer(next_node))
    {
      double demand = get_customer_demand(next_node);
      if (current_load < demand)
      {
        repaired_tour.push_back(DEPOT);
        current_load = MAX_CAPACITY;
        current_battery = BATTERY_CAPACITY;
        current_node = DEPOT;
      }

      // Remove this customer from remaining list
      remaining_customers.erase(
          std::remove(remaining_customers.begin(), remaining_customers.end(), next_node),
          remaining_customers.end());
    }

    if (!has_sufficient_energy(current_battery, current_node, next_node))
    {
      int charging_station;
      if (next_node == DEPOT)
      {
        charging_station = find_best_charging_station(current_node, DEPOT, current_battery);
      }
      else
      {
        charging_station = find_smart_charging_station(current_node, next_node, current_battery, remaining_customers);
      }

      if (charging_station != -1)
      {
        repaired_tour.push_back(charging_station);
        current_battery = BATTERY_CAPACITY;
        current_node = charging_station;
      }
      else
      {
        repaired_tour.push_back(DEPOT);
        current_load = MAX_CAPACITY;
        current_battery = BATTERY_CAPACITY;
        continue;
      }
    }

    repaired_tour.push_back(next_node);

    if (is_customer(next_node))
    {
      current_load -= get_customer_demand(next_node);
    }

    current_battery -= get_energy_consumption(current_node, next_node);

    if (next_node == DEPOT)
    {
      current_load = MAX_CAPACITY;
      current_battery = BATTERY_CAPACITY;
    }
    else if (is_station(next_node))
    {
      current_battery = BATTERY_CAPACITY;
    }
  }

  // Remove consecutive depots
  std::vector<int> final_tour;
  for (size_t i = 0; i < repaired_tour.size(); ++i)
  {
    if (i == 0 || repaired_tour[i] != DEPOT || repaired_tour[i - 1] != DEPOT)
    {
      final_tour.push_back(repaired_tour[i]);
    }
  }

  if (final_tour.empty() || final_tour.back() != DEPOT)
  {
    final_tour.push_back(DEPOT);
  }

  s.tour = final_tour;
  s.steps = static_cast<int>(s.tour.size());
  s.tour_length = fitness_evaluation(s.tour.data(), s.steps);

  if (!is_feasible_detailed(s.tour))
  {
    s.tour.clear();
    s.tour.push_back(DEPOT);
    for (int i = 1; i <= NUM_OF_CUSTOMERS; ++i)
    {
      s.tour.push_back(i);
      s.tour.push_back(DEPOT);
    }
    s.steps = static_cast<int>(s.tour.size());
    repair_in_place(s);
  }
}

/* =========================================================
Enhanced Local Search Operators
=========================================================*/
static bool two_opt_enhanced(solution &s)
{
  auto segments = get_route_segments(s);
  bool improved = false;
  double improvement_threshold = 1e-6 + tie_breaker_dist(rng) * 1e-8;
  double old_fitness = s.tour_length;

  for (auto seg : segments)
  {
    if (seg.second - seg.first <= 3)
      continue;

    for (int i = seg.first + 1; i < seg.second - 2; ++i)
    {
      for (int k = i + 1; k < seg.second - 1; ++k)
      {
        if (is_tabu_move(i, k))
          continue;

        solution temp = s;
        std::reverse(temp.tour.begin() + i, temp.tour.begin() + k + 1);

        if (is_feasible_detailed(temp.tour))
        {
          temp.tour_length = fitness_evaluation(temp.tour.data(), temp.steps);
          if (temp.tour_length < s.tour_length - improvement_threshold)
          {
            s = temp;
            add_tabu_move(i, k);
            improved = true;
            break;
          }
        }
      }
      if (improved)
        break;
    }
    if (improved)
      break;
  }

  param_manager.record_application("2-opt", old_fitness - s.tour_length);
  return improved;
}

static bool or_opt(solution &s)
{
  bool improved = false;
  auto segments = get_route_segments(s);
  double improvement_threshold = 1e-6 + tie_breaker_dist(rng) * 1e-8;
  double old_fitness = s.tour_length;

  for (int seq_len = 1; seq_len <= 2; ++seq_len)
  {
    for (auto seg : segments)
    {
      for (int i = seg.first + 1; i <= seg.second - seq_len - 1; ++i)
      {
        for (int j = seg.first + 1; j < seg.second; ++j)
        {
          if (j >= i && j <= i + seq_len)
            continue;
          if (is_tabu_move(i, j))
            continue;

          solution temp = s;
          std::vector<int> sequence(temp.tour.begin() + i, temp.tour.begin() + i + seq_len);
          temp.tour.erase(temp.tour.begin() + i, temp.tour.begin() + i + seq_len);

          int insert_pos = j;
          if (j > i)
            insert_pos -= seq_len;

          temp.tour.insert(temp.tour.begin() + insert_pos, sequence.begin(), sequence.end());
          temp.steps = static_cast<int>(temp.tour.size());

          if (is_feasible_detailed(temp.tour))
          {
            temp.tour_length = fitness_evaluation(temp.tour.data(), temp.steps);
            if (temp.tour_length < s.tour_length - improvement_threshold)
            {
              s = temp;
              add_tabu_move(i, j);
              param_manager.record_application("or-opt", old_fitness - s.tour_length);
              return true;
            }
          }
        }
      }
    }
  }

  param_manager.record_application("or-opt", old_fitness - s.tour_length);
  return improved;
}

/* =========================================================
New Chain Exchange Operator
=========================================================*/
static bool chain_exchange(solution &s)
{
  auto segments = get_route_segments(s);
  if (segments.size() < 2)
    return false;

  bool improved = false;
  double old_fitness = s.tour_length;

  for (size_t r1 = 0; r1 < segments.size() && !improved; ++r1)
  {
    for (size_t r2 = r1 + 1; r2 < segments.size() && !improved; ++r2)
    {
      auto seg1 = segments[r1];
      auto seg2 = segments[r2];

      for (int len1 = 1; len1 <= 2; ++len1)
      {
        for (int len2 = 1; len2 <= 2; ++len2)
        {
          if (seg1.second - seg1.first <= len1 + 1 ||
              seg2.second - seg2.first <= len2 + 1)
            continue;

          for (int start1 = seg1.first + 1; start1 <= seg1.second - len1 - 1; ++start1)
          {
            for (int start2 = seg2.first + 1; start2 <= seg2.second - len2 - 1; ++start2)
            {

              solution temp = s;

              std::vector<int> chain1(temp.tour.begin() + start1,
                                      temp.tour.begin() + start1 + len1);
              std::vector<int> chain2(temp.tour.begin() + start2,
                                      temp.tour.begin() + start2 + len2);

              if (start2 > start1)
              {
                temp.tour.erase(temp.tour.begin() + start2,
                                temp.tour.begin() + start2 + len2);
                temp.tour.erase(temp.tour.begin() + start1,
                                temp.tour.begin() + start1 + len1);

                temp.tour.insert(temp.tour.begin() + start1,
                                 chain2.begin(), chain2.end());
                temp.tour.insert(temp.tour.begin() + start2 - len1 + len2,
                                 chain1.begin(), chain1.end());
              }
              else
              {
                temp.tour.erase(temp.tour.begin() + start1,
                                temp.tour.begin() + start1 + len1);
                temp.tour.erase(temp.tour.begin() + start2 - len1,
                                temp.tour.begin() + start2 - len1 + len2);

                temp.tour.insert(temp.tour.begin() + start1,
                                 chain2.begin(), chain2.end());
                temp.tour.insert(temp.tour.begin() + start2 - len1 + len2,
                                 chain1.begin(), chain1.end());
              }

              temp.steps = static_cast<int>(temp.tour.size());

              if (is_feasible_detailed(temp.tour))
              {
                temp.tour_length = fitness_evaluation(temp.tour.data(), temp.steps);
                if (temp.tour_length < s.tour_length - 1e-6)
                {
                  s = temp;
                  improved = true;
                  break;
                }
              }
            }
            if (improved)
              break;
          }
          if (improved)
            break;
        }
        if (improved)
          break;
      }
    }
  }

  param_manager.record_application("chain-exchange", old_fitness - s.tour_length);
  return improved;
}

/* =========================================================
Enhanced ALNS
=========================================================*/
static void simplified_alns(solution &s)
{
  double old_fitness = s.tour_length;

  std::vector<int> customers;
  for (int i = 1; i < s.steps - 1; ++i)
  {
    if (is_customer(s.tour[i]))
    {
      customers.push_back(i);
    }
  }

  if (customers.size() < 2)
    return;

  int num_remove = std::max(1, std::min(3, static_cast<int>(customers.size() * 0.2)));

  std::shuffle(customers.begin(), customers.end(), rng);
  std::vector<int> removed_customers;
  std::vector<int> removal_positions;

  for (int i = 0; i < num_remove; ++i)
  {
    removed_customers.push_back(s.tour[customers[i]]);
    removal_positions.push_back(customers[i]);
  }

  std::sort(removal_positions.rbegin(), removal_positions.rend());
  for (int pos : removal_positions)
  {
    s.tour.erase(s.tour.begin() + pos);
  }
  s.steps = static_cast<int>(s.tour.size());

  for (int customer : removed_customers)
  {
    std::vector<std::pair<int, double>> insertion_costs;

    for (int pos = 1; pos < s.steps; ++pos)
    {
      double cost = get_distance_with_variation(s.tour[pos - 1], customer) +
                    get_distance_with_variation(customer, s.tour[pos]) -
                    get_distance_with_variation(s.tour[pos - 1], s.tour[pos]);

      solution temp = s;
      temp.tour.insert(temp.tour.begin() + pos, customer);
      temp.steps++;

      if (is_feasible_detailed(temp.tour))
      {
        insertion_costs.push_back({pos, cost});
      }
    }

    if (!insertion_costs.empty())
    {
      std::sort(insertion_costs.begin(), insertion_costs.end(),
                [](const auto &a, const auto &b)
                { return a.second < b.second; });

      int num_candidates = std::min(3, static_cast<int>(insertion_costs.size()));
      int selected_idx = rng() % num_candidates;
      int best_pos = insertion_costs[selected_idx].first;

      s.tour.insert(s.tour.begin() + best_pos, customer);
      s.steps++;
    }
    else
    {
      s.tour.insert(s.tour.end() - 1, customer);
      s.steps++;
    }
  }

  repair_in_place(s);
  param_manager.record_application("ALNS", old_fitness - s.tour_length);
}

/* =========================================================
SA Framework Supporting Functions
=========================================================*/

// Modified VNS for single step (instead of full VNS loop)
static void vns_search_single_step(solution &s)
{
  // Use adaptive operator ranking if available
  auto operator_ranking = param_manager.get_operator_ranking();
  std::vector<int> neighborhood_order = {1, 2, 3, 4, 5, 6};

  if (!operator_ranking.empty())
  {
    std::shuffle(neighborhood_order.begin(), neighborhood_order.end(), rng);
  }

  // Pick a random neighborhood (1 to VNS_MAX_NEIGHBORHOODS)
  int neighborhood = neighborhood_order[rng() % VNS_MAX_NEIGHBORHOODS];
  double old_fitness = s.tour_length;

  // Apply shaking based on selected neighborhood
  switch (neighborhood)
  {
  case 1:
  {
    // Simple 2-opt move
    auto segments = get_route_segments(s);
    if (!segments.empty())
    {
      auto seg = segments[rng() % segments.size()];
      if (seg.second - seg.first > 3)
      {
        int i = seg.first + 1 + rng() % (seg.second - seg.first - 3);
        int j = i + 2 + rng() % (seg.second - i - 2);
        std::reverse(s.tour.begin() + i, s.tour.begin() + j + 1);
      }
    }
    break;
  }
  case 2:
    simplified_alns(s);
    break;
  case 3:
  {
    // Random customer swap
    std::vector<int> customers;
    for (int i = 1; i < s.steps - 1; ++i)
    {
      if (is_customer(s.tour[i]))
        customers.push_back(i);
    }
    if (customers.size() >= 2)
    {
      int i = customers[rng() % customers.size()];
      int j = customers[rng() % customers.size()];
      if (i != j)
        std::swap(s.tour[i], s.tour[j]);
    }
    break;
  }
  case 4:
    or_opt(s);
    break;
  case 5:
  {
    // Route perturbation
    auto segments = get_route_segments(s);
    if (segments.size() > 1)
    {
      auto seg = segments[rng() % segments.size()];
      if (seg.second - seg.first > 2)
      {
        std::vector<int> route_customers;
        for (int i = seg.first + 1; i < seg.second; ++i)
        {
          if (is_customer(s.tour[i]))
          {
            route_customers.push_back(s.tour[i]);
          }
        }
        std::shuffle(route_customers.begin(), route_customers.end(), rng);

        int cust_idx = 0;
        for (int i = seg.first + 1; i < seg.second; ++i)
        {
          if (is_customer(s.tour[i]))
          {
            s.tour[i] = route_customers[cust_idx++];
          }
        }
      }
    }
    break;
  }
  case 6:
    // New chain exchange neighborhood
    chain_exchange(s);
    break;
  }

  repair_in_place(s);

  // Single local search step with adaptive selection
  if (param_manager.get_operator_weight("2-opt") > 0.5 && URD(rng) < 0.4)
  {
    two_opt_enhanced(s);
  }
  else if (param_manager.get_operator_weight("or-opt") > 0.5 && URD(rng) < 0.4)
  {
    or_opt(s);
  }
  else if (param_manager.get_operator_weight("chain-exchange") > 0.3 && URD(rng) < 0.2)
  {
    chain_exchange(s);
  }

  param_manager.record_application("VNS-neighborhood-" + std::to_string(neighborhood),
                                   old_fitness - s.tour_length);
}

// Route perturbation function
static void route_perturbation(solution &s)
{
  double old_fitness = s.tour_length;

  auto segments = get_route_segments(s);
  if (segments.size() > 1)
  {
    auto seg = segments[rng() % segments.size()];
    if (seg.second - seg.first > 2)
    {
      std::vector<int> route_customers;
      for (int i = seg.first + 1; i < seg.second; ++i)
      {
        if (is_customer(s.tour[i]))
        {
          route_customers.push_back(s.tour[i]);
        }
      }
      std::shuffle(route_customers.begin(), route_customers.end(), rng);

      int cust_idx = 0;
      for (int i = seg.first + 1; i < seg.second; ++i)
      {
        if (is_customer(s.tour[i]))
        {
          s.tour[i] = route_customers[cust_idx++];
        }
      }
    }
  }

  repair_in_place(s);
  param_manager.record_application("route-perturbation", old_fitness - s.tour_length);
}

// Elite solution guided search
static void elite_guided_search(solution &s)
{
  if (elite_solutions.empty())
    return;

  double old_fitness = s.tour_length;

  // Pick a random elite solution
  const solution &elite = elite_solutions[rng() % elite_solutions.size()];

  std::set<std::pair<int, int>> elite_edges;
  for (int i = 0; i < elite.steps - 1; ++i)
  {
    int from = elite.tour[i], to = elite.tour[i + 1];
    if (is_customer(from) && is_customer(to))
    {
      elite_edges.insert({std::min(from, to), std::max(from, to)});
    }
  }

  // Try to apply a move that creates an edge present in elite solution
  for (int attempts = 0; attempts < 10; ++attempts)
  {
    solution temp = s;
    simplified_alns(temp);

    // Check if this move created any elite edges
    bool has_elite_edge = false;
    for (int i = 0; i < temp.steps - 1; ++i)
    {
      int from = temp.tour[i], to = temp.tour[i + 1];
      if (is_customer(from) && is_customer(to))
      {
        std::pair<int, int> edge = {std::min(from, to), std::max(from, to)};
        if (elite_edges.count(edge))
        {
          has_elite_edge = true;
          break;
        }
      }
    }

    if (has_elite_edge)
    {
      s = temp;
      break;
    }
  }

  param_manager.record_application("elite-guidance", old_fitness - s.tour_length);
}

/* =========================================================
Enhanced Initialization with Savings Algorithm
=========================================================*/
static void make_initial_savings(solution &s)
{
  struct SavingsPair
  {
    int i, j;
    double savings;
    double energy_feasibility_score;

    double composite_score() const
    {
      return savings * 0.8 + energy_feasibility_score * 0.2;
    }
  };

  std::vector<SavingsPair> savings_list;

  for (int i = 1; i <= NUM_OF_CUSTOMERS; ++i)
  {
    for (int j = i + 1; j <= NUM_OF_CUSTOMERS; ++j)
    {
      double savings = get_distance(DEPOT, i) + get_distance(DEPOT, j) -
                       get_distance(i, j);

      double energy_ij = get_energy_consumption(i, j);
      double energy_ji = get_energy_consumption(j, i);
      double energy_score = 1.0 / (1.0 + (energy_ij + energy_ji) / BATTERY_CAPACITY);

      savings_list.push_back({i, j, savings, energy_score});
    }
  }

  std::sort(savings_list.begin(), savings_list.end(),
            [](const auto &a, const auto &b)
            {
              return a.composite_score() > b.composite_score();
            });

  std::vector<std::vector<int>> routes;
  std::vector<bool> used(NUM_OF_CUSTOMERS + 1, false);

  for (const auto &pair : savings_list)
  {
    if (!used[pair.i] && !used[pair.j])
    {
      std::vector<int> route = {DEPOT, pair.i, pair.j, DEPOT};

      if (is_feasible_detailed(route))
      {
        routes.push_back(route);
        used[pair.i] = used[pair.j] = true;
      }
    }
  }

  // Add remaining customers as individual routes
  for (int i = 1; i <= NUM_OF_CUSTOMERS; ++i)
  {
    if (!used[i])
    {
      routes.push_back({DEPOT, i, DEPOT});
    }
  }

  // Combine routes into single tour
  s.tour.clear();
  s.tour.push_back(DEPOT);

  for (const auto &route : routes)
  {
    for (size_t i = 1; i < route.size() - 1; ++i)
    {
      s.tour.push_back(route[i]);
    }
    if (&route != &routes.back())
    {
      s.tour.push_back(DEPOT);
    }
  }

  s.tour.push_back(DEPOT);
  s.steps = static_cast<int>(s.tour.size());
  repair_in_place(s);
}

static void make_initial_nearest_neighbor(solution &s)
{
  std::vector<bool> visited(NUM_OF_CUSTOMERS + 1, false);
  s.tour.clear();
  s.tour.push_back(DEPOT);

  while (true)
  {
    std::vector<int> unvisited;
    for (int i = 1; i <= NUM_OF_CUSTOMERS; ++i)
    {
      if (!visited[i])
        unvisited.push_back(i);
    }

    if (unvisited.empty())
      break;

    int current = s.tour.back();
    std::vector<std::pair<int, double>> candidates;

    for (int customer : unvisited)
    {
      if (get_customer_demand(customer) <= MAX_CAPACITY)
      {
        double dist = get_distance_with_variation(current, customer);
        candidates.push_back({customer, dist});
      }
    }

    if (candidates.empty())
    {
      s.tour.push_back(DEPOT);
      continue;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b)
              { return a.second < b.second; });

    int num_candidates = std::min(3, static_cast<int>(candidates.size()));
    int selected_idx = rng() % num_candidates;
    int nearest = candidates[selected_idx].first;

    s.tour.push_back(nearest);
    visited[nearest] = true;
  }

  s.tour.push_back(DEPOT);
  s.steps = static_cast<int>(s.tour.size());
}

static void make_initial(solution &s)
{
  // Randomly choose initialization method
  if (URD(rng) < 0.3)
  {
    make_initial_savings(s);
  }
  else
  {
    make_initial_nearest_neighbor(s);
  }
  repair_in_place(s);
}

/* =========================================================
Path Relinking Implementation
=========================================================*/
static solution path_relink(const solution &source, const solution &target)
{
  solution current = source;
  solution best_intermediate = source;

  std::vector<int> source_customers, target_customers;

  for (int i = 1; i < source.steps - 1; ++i)
  {
    if (is_customer(source.tour[i]))
    {
      source_customers.push_back(source.tour[i]);
    }
  }

  for (int i = 1; i < target.steps - 1; ++i)
  {
    if (is_customer(target.tour[i]))
    {
      target_customers.push_back(target.tour[i]);
    }
  }

  // Simple path relinking by trying to match customer sequences
  int max_moves = std::min(5, static_cast<int>(source_customers.size() / 4));

  for (int move = 0; move < max_moves; ++move)
  {
    solution candidate = current;

    // Try a random local change towards target structure
    if (!source_customers.empty() && !target_customers.empty())
    {
      // Random customer swap towards target order
      int src_customer = source_customers[rng() % source_customers.size()];
      int tgt_customer = target_customers[rng() % target_customers.size()];

      // Find positions and swap
      for (int i = 1; i < candidate.steps - 1; ++i)
      {
        if (candidate.tour[i] == src_customer)
        {
          for (int j = 1; j < candidate.steps - 1; ++j)
          {
            if (candidate.tour[j] == tgt_customer)
            {
              std::swap(candidate.tour[i], candidate.tour[j]);
              break;
            }
          }
          break;
        }
      }
    }

    repair_in_place(candidate);

    if (candidate.tour_length < best_intermediate.tour_length)
    {
      best_intermediate = candidate;
    }

    current = candidate;
  }

  return best_intermediate;
}

/* =========================================================
Elite Solution Management
=========================================================*/
static void update_elite_solutions(const solution &s)
{
  // Check if solution is already in elite set
  for (const auto &elite : elite_solutions)
  {
    if (std::abs(elite.tour_length - s.tour_length) < 1e-9)
    {
      return; // Already have this solution
    }
  }

  elite_solutions.push_back(s);

  // Sort by fitness
  std::sort(elite_solutions.begin(), elite_solutions.end(),
            [](const solution &a, const solution &b)
            {
              return a.tour_length < b.tour_length;
            });

  // Keep only the best solutions
  if (elite_solutions.size() > ELITE_SOLUTIONS)
  {
    elite_solutions.resize(ELITE_SOLUTIONS);
  }
}

static void path_relinking_phase()
{
  if (elite_solutions.size() < 2)
    return;

  // Try path relinking between random elite pairs
  int num_pairs = std::min(3, static_cast<int>(elite_solutions.size() / 2));

  for (int pair = 0; pair < num_pairs; ++pair)
  {
    int idx1 = rng() % elite_solutions.size();
    int idx2 = rng() % elite_solutions.size();

    if (idx1 != idx2)
    {
      solution relinked = path_relink(elite_solutions[idx1], elite_solutions[idx2]);

      // Apply local search to the relinked solution
      for (int ls = 0; ls < 2; ++ls)
      {
        two_opt_enhanced(relinked);
        or_opt(relinked);
      }

      if (relinked.tour_length < cur_sol.tour_length - 1e-6)
      {
        cur_sol = relinked;
        if (cur_sol.tour_length < best_sol->tour_length - 1e-6)
        {
          *best_sol = cur_sol;
        }
      }

      update_elite_solutions(relinked);
    }
  }
}

/* =========================================================
Main SA Algorithm with Enhanced Hybrid EVRP Logic
=========================================================*/
void initialize_heuristic()
{
  auto now = std::chrono::high_resolution_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  unsigned seed = static_cast<unsigned>(std::time(nullptr)) ^ static_cast<unsigned>(nanos);
  rng.seed(seed);

  update_dynamic_parameters();

  best_sol = new solution;
  make_initial(cur_sol);
  *best_sol = cur_sol;

  elite_solutions.clear();
  elite_solutions.push_back(cur_sol);

  std::cout << "Enhanced Hybrid EVRP Algorithm initialized with seed: " << seed << std::endl;
  std::cout << "Initial solution fitness: " << best_sol->tour_length << std::endl;
  std::cout << "Dynamic parameters - Tabu size: " << current_tabu_size
            << ", Max iterations: " << current_max_iterations_without_improvement
            << ", Energy margin: " << current_energy_safety_margin << std::endl;
}

void run_heuristic()
{
  double T = T_INIT;
  auto last_print = std::chrono::high_resolution_clock::now();
  auto start_time = std::chrono::high_resolution_clock::now();
  long iteration = 0;

  std::cout << "Starting SA-Enhanced Hybrid EVRP with T_init=" << T_INIT << std::endl;

  while (get_evals() < TERMINATION && T > T_END && !termination_condition())
  {
    iteration++;

    /* --- Update dynamic parameters periodically --- */
    if (iteration % 100 == 0)
    {
      update_dynamic_parameters();
    }

    /* --- Update adaptive parameter manager --- */
    if (param_manager.should_update())
    {
      param_manager.update_parameters();
      param_manager.reset_update_counter();
    }

    /* --- Create candidate solution using enhanced hybrid methods --- */
    cand_sol = cur_sol;

    // Apply one of your enhanced hybrid operators based on iteration
    int operator_choice = iteration % 8;

    switch (operator_choice)
    {
    case 0:
    case 1:
      /* Primary: VNS search (most frequent) */
      vns_search_single_step(cand_sol);
      break;

    case 2:
      /* ALNS with local improvement */
      simplified_alns(cand_sol);
      // Enhanced local search polish with adaptive selection
      for (int ls = 0; ls < 2; ++ls)
      {
        bool improved = false;
        if (param_manager.get_operator_weight("2-opt") > 0.5)
        {
          if (two_opt_enhanced(cand_sol))
            improved = true;
        }
        if (param_manager.get_operator_weight("or-opt") > 0.5)
        {
          if (or_opt(cand_sol))
            improved = true;
        }
        if (!improved)
          break;
      }
      break;

    case 3:
      /* Pure local search operators */
      if (!two_opt_enhanced(cand_sol))
      {
        or_opt(cand_sol);
      }
      break;

    case 4:
      /* Route-level perturbation */
      route_perturbation(cand_sol);
      break;

    case 5:
      /* Elite solution guidance (if available) */
      if (!elite_solutions.empty())
      {
        elite_guided_search(cand_sol);
      }
      else
      {
        simplified_alns(cand_sol);
      }
      break;

    case 6:
      /* Chain exchange operator */
      chain_exchange(cand_sol);
      repair_in_place(cand_sol);
      break;

    case 7:
      /* Path relinking with elite solutions */
      if (elite_solutions.size() >= 2)
      {
        int elite_idx = rng() % elite_solutions.size();
        cand_sol = path_relink(cur_sol, elite_solutions[elite_idx]);
      }
      else
      {
        simplified_alns(cand_sol);
      }
      break;
    }

    /* --- Metropolis acceptance --- */
    bool accepted = accept(cur_sol.tour_length, cand_sol.tour_length, T);

    if (accepted)
    {
      cur_sol = cand_sol;
      iterations_without_improvement = 0;

      // Update global best
      if (cur_sol.tour_length < best_sol->tour_length)
      {
        *best_sol = cur_sol;

        // Update elite solutions when we find new best
        update_elite_solutions(cur_sol);

        std::cout << "New best found: " << best_sol->tour_length
                  << " at iteration " << iteration << " (T=" << T << ")" << std::endl;
      }
      else
      {
        // Update elite solutions for good solutions too
        update_elite_solutions(cur_sol);
      }
    }
    else
    {
      ++iterations_without_improvement;
    }

    /* --- Enhanced diversification when stuck --- */
    if (iterations_without_improvement > current_max_iterations_without_improvement)
    {
      std::cout << "Diversifying at iteration " << iteration
                << " (T=" << T << ")" << std::endl;

      // Multiple diversification strategies
      for (int i = 0; i < 3; ++i)
      {
        simplified_alns(cur_sol);
      }

      // Try elite solution injection
      if (!elite_solutions.empty() && URD(rng) < 0.3)
      {
        int elite_idx = rng() % elite_solutions.size();
        cur_sol = elite_solutions[elite_idx];

        // Perturb the elite solution
        for (int j = 0; j < 2; ++j)
        {
          simplified_alns(cur_sol);
        }
      }

      iterations_without_improvement = 0;

      // Clear tabu memory
      while (!tabu_list.empty())
        tabu_list.pop();
      tabu_set.clear();
    }

    /* --- Path relinking phase --- */
    if (iteration % 50 == 0 && elite_solutions.size() >= 2)
    {
      path_relinking_phase();
    }

    /* --- Optional console progress --- */
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration<double>(now - last_print).count() > PROGRESS)
    {
      double elapsed = std::chrono::duration<double>(now - start_time).count();
      std::cout << "evals " << get_evals()
                << "  best " << best_sol->tour_length
                << "  curr " << cur_sol.tour_length
                << "  T " << T
                << "  elite_size " << elite_solutions.size()
                << "  time " << elapsed << "s"
                << "  iter " << iteration
                << std::endl;
      last_print = now;
    }

    /* --- Performance log for analysis --- */
    log_performance(get_current_run(),
                    get_evals(),
                    best_sol->tour_length,
                    cur_sol.tour_length,
                    T,
                    accepted);

    /* ===== Geometric cooling ===== */
    T *= ALPHA;
    if (T < T_END)
      T = T_END;
  }

  std::cout << "SA-Enhanced Hybrid finished. Best solution = " << best_sol->tour_length << std::endl;
  std::cout << "Elite solutions found: " << elite_solutions.size() << std::endl;

  // Print operator performance statistics
  if (param_manager.should_update())
  {
    param_manager.update_parameters();
    auto ranking = param_manager.get_operator_ranking();
    std::cout << "Operator performance ranking:" << std::endl;
    for (const auto &op : ranking)
    {
      std::cout << "  " << op << ": weight = " << param_manager.get_operator_weight(op) << std::endl;
    }
  }
}

void free_heuristic()
{
  delete best_sol;
  elite_solutions.clear();
  while (!tabu_list.empty())
  {
    tabu_list.pop();
  }
  tabu_set.clear();
}