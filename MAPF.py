import itertools
from collections import deque
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import time
import random

# ================= CLASES BÁSICAS =================

class Agent:
    def __init__(self, id, start: tuple[int, int], goal: tuple[int, int]):
        self.id = id
        self.start = start
        self.goal = goal
        self.path = []

    def restart_path(self):
        self.path = []

    def __repr__(self):
        return f"A{self.id}"


class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Environment:
    def __init__(self, width: int, height: int, obstacles: list[Obstacle]):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.table = self.create_table()

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)
        self.table[obs.x, obs.y] = 1

    def remove_obstacle(self, obs: Obstacle):
        if obs in self.obstacles:
            self.obstacles.remove(obs)
        self.table[obs.x, obs.y] = 0

    def restart_table(self):
        self.table = np.zeros((self.height, self.width))
        for obs in self.obstacles:
            self.table[obs.x, obs.y] = 1

    def create_table(self):
        table = np.zeros((self.height, self.width))
        for obs in self.obstacles:
            table[obs.x, obs.y] = 1
        return table

    def update_table(self, path):
        for x, y in path:
            self.table[x, y] = 2

    def is_free(self, position):
        x, y = position
        if (0 <= x) and (x < self.width) and (0 <= y) and (y < self.height):
            return self.table[x, y] == 0
        return False

    def get_neighbours(self, position):
        x, y = position
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        n = []
        for pos in neighbors:
            if self.is_free(pos):
                n.append(pos)
        return n

    def get_cost(self, agents):
        return sum(len(agent.path) for agent in agents)


class MultiAgentPathfinding:
    def __init__(self, environment, agents):
        self.environment = environment
        self.agents = agents

        for agent in self.agents:
            self.environment.add_obstacle(Obstacle(agent.goal[0], agent.goal[1]))
            self.environment.add_obstacle(Obstacle(agent.start[0], agent.start[1]))

    def plot_paths(self, paths, title="Paths"):
        canvas = np.zeros((self.environment.height, self.environment.width))

        for obs in self.environment.obstacles:
            canvas[obs.x, obs.y] = 100

        for i, path in enumerate(paths):
            for x, y in path:
                canvas[x, y] = 30 + (i * 15)

        plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.imshow(canvas, cmap='viridis', origin='upper')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def find_path(self, start, goal):
        visited = np.zeros((self.environment.height, self.environment.width))
        queue = [[start]]

        while len(queue) > 0:
            cur = queue.pop(0)
            last = cur[-1]

            if last == goal:
                return cur

            if visited[last[0], last[1]]:
                continue

            visited[last[0], last[1]] = 1

            for neigh in self.environment.get_neighbours(last):
                if visited[neigh[0], neigh[1]] == 0:
                    new_cur = list(cur)
                    new_cur.append(neigh)
                    queue.append(new_cur)

        return []

    def evaluate_permutation(self, perm_indices):
        self.environment.restart_table()

        for agent in self.agents:
            agent.restart_path()

        paths_in_order = []

        for idx in perm_indices:
            agent = self.agents[idx]

            self.environment.remove_obstacle(Obstacle(agent.goal[0], agent.goal[1]))
            self.environment.remove_obstacle(Obstacle(agent.start[0], agent.start[1]))

            path = self.find_path(agent.start, agent.goal)

            self.environment.add_obstacle(Obstacle(agent.goal[0], agent.goal[1]))
            self.environment.add_obstacle(Obstacle(agent.start[0], agent.start[1]))

            if not path:
                return [], float("inf")

            paths_in_order.append(path)
            self.agents[idx].path = path
            self.environment.update_table(path)

        return paths_in_order, self.environment.get_cost(self.agents)

    def all_paths(self, max_limit=3000):
        best_paths = []
        best_cost = float("inf")

        n = len(self.agents)
        permutations = list(itertools.permutations(list(range(n))))

        steps = 0
        limit = min(len(permutations), max_limit)

        for i in range(limit):
            steps += 1

            paths, cost = self.evaluate_permutation(permutations[i])

            if cost < best_cost:
                best_cost = cost
                best_paths = paths

            self.environment.restart_table()
            for agent in self.agents:
                agent.restart_path()

        status = "Exact" if steps == len(permutations) else "Cutoff"
        return best_paths, best_cost, steps, status

    def solve_random_search(self, iterations=20):
        best_paths = []
        best_cost = float("inf")

        n = len(self.agents)
        base_perm = list(range(n))

        steps = 0

        for _ in range(iterations):
            steps += 1

            current_perm = base_perm[:]
            random.shuffle(current_perm)

            paths, cost = self.evaluate_permutation(current_perm)

            if cost < best_cost:
                best_cost = cost
                best_paths = paths

            self.environment.restart_table()
            for agent in self.agents:
                agent.restart_path()

        return best_paths, best_cost, steps

    def solve_hill_climbing_first_improvement(self):
        n = len(self.agents)
        current_perm = list(range(n))

        steps = 0

        best_paths, best_cost = self.evaluate_permutation(current_perm)
        steps += 1

        while True:
            improved = False

            for i in range(n):
                if improved:
                    break

                for j in range(i + 1, n):
                    neighbor = current_perm[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                    paths, cost = self.evaluate_permutation(neighbor)
                    steps += 1

                    if cost < best_cost:
                        best_cost = cost
                        best_paths = paths
                        current_perm = neighbor
                        improved = True
                        break

            if not improved:
                break

        self.environment.restart_table()
        for agent in self.agents:
            agent.restart_path()

        return best_paths, best_cost, steps

    def solve_aco(self, num_ants=5, iterations=4, alpha=1.0, rho=0.2):
        n_agents = len(self.agents)
        pheromones = np.ones((n_agents, n_agents)) * 0.1

        best_global_paths = []
        best_global_cost = float("inf")

        steps = 0

        for it in range(iterations):
            ants_solutions = []

            for _ in range(num_ants):
                current_perm = []
                available_agents = list(range(n_agents))

                for step in range(n_agents):
                    probs = []
                    for agent_id in available_agents:
                        p = pheromones[step][agent_id] ** alpha
                        probs.append(p)

                    total_prob = sum(probs)
                    if total_prob == 0:
                        probs = [1 / len(probs)] * len(probs)
                    else:
                        probs = [p / total_prob for p in probs]

                    chosen_agent = np.random.choice(available_agents, p=probs)
                    current_perm.append(chosen_agent)
                    available_agents.remove(chosen_agent)

                paths, cost = self.evaluate_permutation(current_perm)
                steps += 1

                ants_solutions.append((current_perm, cost))

                if cost < best_global_cost:
                    best_global_cost = cost
                    best_global_paths = paths

            pheromones *= (1 - rho)

            for perm, cost in ants_solutions:
                if cost != float("inf"):
                    deposit = 100.0 / cost
                    for step, agent_id in enumerate(perm):
                        pheromones[step][agent_id] += deposit

            self.environment.restart_table()
            for agent in self.agents:
                agent.restart_path()

        return best_global_paths, best_global_cost, steps


def plot_benchmark_results(results):
    scenarios = list(results.keys())
    algorithms = list(results[scenarios[0]].keys())

    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.2

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, algo in enumerate(algorithms):
        costs = []
        for scen in scenarios:
            c = results[scen][algo][0]
            costs.append(c if c != float('inf') else 0)

        offset = width * i
        rects = ax1.bar(x + offset, costs, width, label=algo, alpha=0.9, edgecolor='white')

        for j, rect in enumerate(rects):
            height = rect.get_height()
            orig = results[scenarios[j]][algo][0]

            if orig == float('inf'):
                label = "FAIL"
                color = 'red'
                pos_y = 5
            else:
                label = f"{int(height)}"
                color = 'black'
                pos_y = height + 1

            if j < 2 or orig == float('inf') or height > 0:
                ax1.annotate(label,
                             (rect.get_x() + rect.get_width() / 2, pos_y),
                             ha='center', va='bottom',
                             color=color, fontsize=8,
                             fontweight='bold')

    ax1.set_ylabel('Coste total')
    ax1.set_title('Calidad de la solución (menor es mejor)')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(scenarios, fontweight='bold')
    ax1.legend(loc='upper left')

    for i, algo in enumerate(algorithms):
        steps_list = [results[scen][algo][1] for scen in scenarios]
        rects = ax2.bar(x + width * i, steps_list, width, label=algo, alpha=0.9, edgecolor='white')
        ax2.bar_label(rects, padding=3, fontsize=7, rotation=90)

    ax2.set_ylabel('Evaluaciones')
    ax2.set_title('Coste computacional (menor es más rápido)')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(scenarios, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    print("\n[INFO] Gráfico guardado como 'benchmark_results.png'")
    plt.show()


def run_performance_benchmark():
    print("\n" + "=" * 95)
    print(f"{'PERFORMANCE BENCHMARK (Low Budget)':^95}")
    print("=" * 95)

    scenarios = []

    env1 = Environment(10, 10, [])
    ags1 = [
        Agent(0, (1, 5), (8, 5)),
        Agent(1, (5, 1), (5, 8))
    ]
    scenarios.append(("EASY (2 Ags)", env1, ags1))

    env2 = Environment(15, 15, [])
    ags2 = [
        Agent(0, (7, 1), (7, 13)),
        Agent(1, (7, 13), (7, 1)),
        Agent(2, (1, 7), (13, 7)),
        Agent(3, (13, 7), (1, 7))
    ]
    scenarios.append(("MED (4 Ags)", env2, ags2))

    obs3 = [Obstacle(10, y) for y in range(20) if y not in [4, 5, 6, 14, 15, 16]]
    env3 = Environment(20, 20, obs3)
    ags3 = [
        Agent(0, (2, 5), (18, 5)),
        Agent(1, (2, 6), (18, 6)),
        Agent(2, (18, 5), (2, 5)),
        Agent(3, (18, 6), (2, 6)),
        Agent(4, (2, 15), (18, 15)),
        Agent(5, (18, 15), (2, 15))
    ]
    scenarios.append(("HARD (6 Ags)", env3, ags3))

    obs4 = [Obstacle(10, y) for y in range(20) if y not in [3, 4, 10, 11, 16, 17]]
    env4 = Environment(20, 20, obs4)
    ags4 = [
        Agent(0, (5, 10), (15, 10)),
        Agent(1, (15, 10), (5, 10)),
        Agent(2, (5, 3), (15, 16)),
        Agent(3, (5, 16), (15, 3)),
        Agent(4, (8, 8), (12, 12)),
        Agent(5, (12, 12), (8, 8)),
        Agent(6, (2, 2), (2, 18))
    ]
    scenarios.append(("INSANE (7 Ags)", env4, ags4))

    results = {}

    best_insane_paths_aco = None
    insane_env = None

    print(f"{'SCENARIO':<15} | {'ALGORITHM':<15} | {'COST':<8} | {'STEPS':<8} | {'COMMENT'}")
    print("-" * 95)

    for name, env, agents in scenarios:
        mapf = MultiAgentPathfinding(env, agents)
        results[name] = {}

        _, cost_bf, steps_bf, status_bf = mapf.all_paths(max_limit=3000)
        results[name]['Brute Force'] = (cost_bf, steps_bf)
        print(f"{name:<15} | {'Brute Force':<15} | {str(cost_bf):<8} | {str(steps_bf):<8} | {status_bf}")

        _, cost_rs, steps_rs = mapf.solve_random_search(iterations=15)
        results[name]['Random Search'] = (cost_rs, steps_rs)
        print(f"{'':<15} | {'Random Search':<15} | {str(cost_rs):<8} | {str(steps_rs):<8} | Baseline")

        _, cost_hc, steps_hc = mapf.solve_hill_climbing_first_improvement()
        results[name]['Hill Climbing'] = (cost_hc, steps_hc)
        print(f"{'':<15} | {'Hill Climbing':<15} | {str(cost_hc):<8} | {str(steps_hc):<8} | Greedy")

        paths_aco, cost_aco, steps_aco = mapf.solve_aco(num_ants=5, iterations=4, rho=0.3)
        results[name]['ACO'] = (cost_aco, steps_aco)

        if "INSANE" in name and cost_aco != float('inf'):
            best_insane_paths_aco = paths_aco
            insane_env = mapf

        comment_aco = "Optimal" if cost_aco <= cost_hc and cost_aco != float('inf') else "Winner"
        if cost_hc == float('inf') and cost_aco != float('inf'):
            comment_aco = "SURVIVOR"

        print(f"{'':<15} | {'ACO':<15} | {str(cost_aco):<8} | {str(steps_aco):<8} | {comment_aco}")
        print("-" * 95)

    plot_benchmark_results(results)

    if best_insane_paths_aco and insane_env:
        print("\n[INFO] Visualizando solución ACO para el escenario INSANE...")
        insane_env.plot_paths(best_insane_paths_aco, title="INSANE Scenario - ACO Solution (Survivor)")


if __name__ == "__main__":
    run_performance_benchmark()
