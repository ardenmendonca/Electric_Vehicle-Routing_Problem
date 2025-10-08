#define CHAR_LEN 100
#define TERMINATION 25000 // DO NOT CHANGE THE NUMBER

extern char *problem_instance;
void init_evals();
void init_current_best();

struct node
{
  int id;
  double x;
  double y;
};

extern int NUM_OF_CUSTOMERS;
extern int ACTUAL_PROBLEM_SIZE;
extern int NUM_OF_STATIONS;
extern int MAX_CAPACITY;
extern int DEPOT;
extern double OPTIMUM;
extern int BATTERY_CAPACITY;
extern int MIN_VEHICLES;

double fitness_evaluation(int *routes, int size);
void print_solution(int *routes, int size);
void check_solution(int *routes, int size);
void read_problem(char *filename);
double get_energy_consumption(int from, int to);
int get_customer_demand(int customer);
double get_distance(int from, int to);
bool is_charging_station(int node);
double get_current_best();
double get_evals();
void free_EVRP();
