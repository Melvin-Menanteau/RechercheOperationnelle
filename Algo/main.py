import numpy as np
import igraph
import matplotlib.pyplot as plt
import pulp
import time

# Parameters for the simulation
NB_VERTICES = 10             # Number of cities
PROBABILITY_EDGE = 0.5         # Probability of an edge between two vertices
MIN_WEIGHT = 10                # Minimum weight of an edge
MAX_WEIGHT = 500               # Maximum weight of an edge
INF = 1e6                      # Infinite value to represent impossible connections

# Create a random graph with nb_vertices vertices
def create_graph(nb_vertices):
    graph = igraph.Graph.Erdos_Renyi(nb_vertices, PROBABILITY_EDGE, directed=False)
    graph["title"] = "City Graph"
    graph.vs["name"] = ["City " + str(i) for i in range(nb_vertices)]
    graph.simplify()
    set_random_weights(graph)
    return graph

def get_adjacency_matrix(graph):
    adjacency_matrix = np.full((graph.vcount(), graph.vcount()), INF)
    for edge in graph.es:
        u, v = edge.source, edge.target
        weight = edge['weight']
        adjacency_matrix[u, v] = weight
        adjacency_matrix[v, u] = weight  # Because the graph is undirected
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix

def set_random_weights(graph):
    for edge in graph.es:
        edge["weight"] = np.random.randint(MIN_WEIGHT, MAX_WEIGHT)
        edge["label"] = edge["weight"]

def solve_tsp_with_pulp(adjacency_matrix, result):

    # Create the problem
    prob = pulp.LpProblem("Traveling_Salesman", pulp.LpMinimize)
    nb_vertices = adjacency_matrix.shape[0]
    cities = list(range(nb_vertices))

    # Decision variables
    x = pulp.LpVariable.dicts("x", (cities, cities), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", cities, 0, nb_vertices, pulp.LpContinuous)

    # Objective function
    prob += pulp.lpSum(adjacency_matrix[i][j] * x[i][j] for i in cities for j in cities), "Total_Cost"

    # Constraints: each city must be visited exactly once
    for i in cities:
        prob += pulp.lpSum(x[i][j] for j in cities if j != i) == 1, f"Out_{i}"
    for j in cities:
        prob += pulp.lpSum(x[i][j] for i in cities if i != j) == 1, f"In_{j}"

    # Subtour elimination constraints (Miller-Tucker-Zemlin)
    for i in cities[1:]:
        for j in cities[1:]:
            if i != j:
                prob += u[i] - u[j] + nb_vertices * x[i][j] <= nb_vertices - 1

    # Constraints to ensure return to the starting city
    prob += pulp.lpSum(x[0][j] for j in cities if j != 0) == 1, "Out_Start"
    prob += pulp.lpSum(x[i][0] for i in cities if i != 0) == 1, "In_Start"

    # Solver initialization with a time limit
    solver = pulp.PULP_CBC_CMD(timeLimit=60)

    prob.solve(solver)  # Solve the problem
    print(f"Status after solving: {pulp.LpStatus[prob.status]}")

    # Check the status of the solution
    if pulp.LpStatus[prob.status] == 'Optimal':
        result.append("Status: Optimal")
    else:
        result.append("Status: Non-optimal")

    # Extract the solution
    solution = [(i, j) for i in cities for j in cities if x[i][j].varValue == 1]

    # Construct the complete cycle starting from city 0
    cycle = [0]
    total_cost = 0
    while len(cycle) < nb_vertices + 1:
        for (i, j) in solution:
            if i == cycle[-1] and j not in cycle:
                cycle.append(j)
                total_cost += adjacency_matrix[i][j]
                break
        if len(cycle) == nb_vertices:  # Append the start city to complete the cycle
            cycle.append(0)
            total_cost += adjacency_matrix[cycle[-2]][0]  # Add cost of returning to the start

    # Display the complete cycle and the total cost
    result.append(f"\nComplete cycle with a total cost of {total_cost}:")
    for i in range(len(cycle) - 1):
        result.append(f"From city {cycle[i]} to city {cycle[i+1]} with a cost of {adjacency_matrix[cycle[i]][cycle[i+1]]}")

def main():
    graph = create_graph(nb_vertices=NB_VERTICES)
    adjacency_matrix = get_adjacency_matrix(graph)
    print("Adjacency Matrix:")
    print(adjacency_matrix)

    # Verify the weights of the edges
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] < INF:
                print(f"Weight of the edge between {i} and {j}: {adjacency_matrix[i][j]}")

    # Container for the result
    result = []

    # Solve the Traveling Salesman Problem with a time limit
    solve_tsp_with_pulp(adjacency_matrix, result)

    # Display the result
    for line in result:
        print(line)

    # Display the graph
    figures, axes = plt.subplots()
    igraph.plot(
        graph,
        target=axes,
        layout=graph.layout("kk"),
        vertex_size=30,
        vertex_label=graph.vs["name"],
        vertex_label_size=5,
        vertex_color=["red" if v.degree() > 0 else "black" for v in graph.vs],
        edge_width=1,
        edge_color="#e8e8e8",
    )

    plt.show()

if __name__ == "__main__":
    main()
