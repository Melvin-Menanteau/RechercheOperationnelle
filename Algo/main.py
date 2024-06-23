import numpy as np
import igraph
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pulp
import math
from gurobipy import *

# ------------------------------------------------- #
# Paramètres de la simulation
# ------------------------------------------------- #
# Limite de temps pour la résolution du problème (en secondes)
TIME_LIMIT = 300
# Nombre de camions
NB_TRUCKS = 3
# Nombre de villes
NB_VERTICES = 20
# Ville de départ (doit être < NB_VERTICES)
START_CITY = round(NB_VERTICES / 2)
# Probabilité d'avoir une arête entre deux sommets (chemin entre deux villes)
PROBABILITY_EDGE = 0.4
# Poids (distance) minimum d'une arête
MIN_WEIGHT = 10
# Poids (distance) maximum d'une arête
MAX_WEIGHT = 500
# Valeur infinie pour représenter une connexion impossible
INF = 1e100
# Couleurs pour les arêtes parcourues par les camions
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "gray"]

# Créer un graphe aléatoire avec NB_VERTICES sommets
def create_graph(nb_vertices):
    graph = igraph.Graph.Erdos_Renyi(nb_vertices, PROBABILITY_EDGE, directed=False)

    graph["title"] = "Graphe des villes"
    graph.vs["name"] = ["Ville " + str(i) for i in range(nb_vertices)]

    # La fonction simplify() supprime les arêtes en double
    # Permet d'alléger le graphe et la recherche de solutions
    graph.simplify()

    set_random_weights(graph)

    return graph

# Définir des poids aléatoires pour les arêtes du graphe
def set_random_weights(graph):
    matrix = [(0, 0) for _ in range(NB_VERTICES)]

    for i in range(NB_VERTICES):
        # matrix[i] = (np.random.randint(MIN_WEIGHT, MAX_WEIGHT), np.random.randint(MIN_WEIGHT, MAX_WEIGHT))
        matrix[i] = (MIN_WEIGHT * i, MIN_WEIGHT * i)

    print(matrix)

    for edge in graph.es:
        # edge["weight"] = np.random.randint(MIN_WEIGHT, MAX_WEIGHT)
        edge["weight"] = round(math.sqrt((matrix[edge.source][0] - matrix[edge.target][0]) ** 2 + (matrix[edge.source][1] - matrix[edge.target][1]) ** 2))
        edge["label"] = edge["weight"]

def get_adjacency_matrix(graph):
    matrice = np.full((graph.vcount(), graph.vcount()), INF)

    for edge in graph.es:
        u, v = edge.source, edge.target
        weight = edge['weight']
        matrice[u, v] = weight
        matrice[v, u] = weight  # Car le graphe est non orienté

    np.fill_diagonal(matrice, 0)

    return matrice

def solve(matrix):
    # Créer le problème PuLP
    prob = pulp.LpProblem("VRP", pulp.LpMinimize)

    # ------------------------------------------------- #
    # Variables de décision
    # ------------------------------------------------- #
    # Indique si l'arête (i, j) est empruntée par le camion k
    c = pulp.LpVariable.dicts("c", [(i, j, k) for i in range(NB_VERTICES) for j in range(NB_VERTICES) for k in range(NB_TRUCKS)], lowBound=0, upBound=1, cat=pulp.LpBinary)
    # Indique la position du camion k
    u = pulp.LpVariable.dicts("u", (i for i in range(1, NB_VERTICES)), lowBound=1, upBound=NB_VERTICES-1, cat=pulp.LpInteger)

    # Fonction objectif
    prob += pulp.lpSum(matrix[i][j] * c[(i, j, k)] for i in range(NB_VERTICES) for j in range(NB_VERTICES) for k in range(NB_TRUCKS))

    # ------------------------------------------------- #
    # Contraintes
    # ------------------------------------------------- #
    # Chaque ville doit être visitée au moins une fois
    for i in range(NB_VERTICES):
        prob += pulp.lpSum(c[(i, j, k)] for j in range(NB_VERTICES) for k in range(NB_TRUCKS)) >= 1

    # L'ensemble des chemins parcourus forment un cycle
    for k in range(NB_TRUCKS):
        for i in range(1, NB_VERTICES):
            prob += pulp.lpSum(c[(i, j, k)] for j in range(NB_VERTICES)) - pulp.lpSum(c[(j, i, k)] for j in range(NB_VERTICES)) == 0

    # Tous les camions doivent partir de la ville START_CITY
    for k in range(NB_TRUCKS):
        prob += pulp.lpSum(c[START_CITY, j, k] for j in [x for x in range(NB_VERTICES) if x != START_CITY]) == 1

    # Tous les camions doivent revenir à la ville START_CITY
    for k in range(NB_TRUCKS):
        prob += pulp.lpSum(c[i, START_CITY, k] for i in [x for x in range(NB_VERTICES) if x != START_CITY]) == 1

    # Ne pas boucler sur la même ville
    for k in range(NB_TRUCKS):
        prob += pulp.lpSum(c[(i, i, k)] for i in range(NB_VERTICES)) == 0

    for i in range(1, NB_VERTICES):
        for j in range(1, NB_VERTICES):
            if i != j:
                for k in range(NB_TRUCKS):
                    prob += u[i] - u[j] + (NB_VERTICES - 1) * c[(i, j, k)] <= NB_VERTICES - 2

    # La répartition entre les camions doit être équitable
    # for i in range(1, NB_VERTICES):
    #     prob += pulp.lpSum(c[(i, j, k)] for j in range(NB_VERTICES) for k in range(NB_TRUCKS)) == 1

    # Paramètres du solveur pour limiter le temps de résolution (en secondes)
    # solver = pulp.PULP_CBC_CMD(timeLimit=TIME_LIMIT)
    # solver = pulp.getSolver("GUROBI_CMD", timeLimit=TIME_LIMIT)

    prob.solve(pulp.GUROBI(Cuts=0, Presolve=0, Heuristics=0, OutputFlag=1, msg=1, TimeLimit=TIME_LIMIT))

    # Afficher les résultats
    print("Status:", pulp.LpStatus[prob.status])
    print("Objective:", pulp.value(prob.objective))

    for v in sorted(prob.variables(), key=lambda v: v.toDict()["name"].split("_")[-1][0:-1]):
        if v.varValue > 0 and v.name.startswith("c"):
            print(v.name, "=", v.varValue, v)

    return [(i, j, k) for i in range(NB_VERTICES) for j in range(NB_VERTICES) for k in range(NB_TRUCKS) if c[(i, j, k)].varValue > 0]

def main():
    graphe = create_graph(nb_vertices=NB_VERTICES)
    matrice = get_adjacency_matrix(graph=graphe)

    # print(matrice)

    solution = solve(matrice)

    print(solution)
    # solution = None

    # Colorier les arêtes parcourues par les camions
    BASE_COLOR = "#ececec"

    for edge in graphe.es:
        try:
            sol = list(filter(lambda s: (s[0] == edge.source) and (s[1] == edge.target) or (s[1] == edge.source) and (s[0] == edge.target), solution))[0]

            if sol is not None:
                edge["color"] = COLORS[sol[2]]
                continue
        except:
            pass

        edge["color"] = BASE_COLOR

    # Somme cumulée des distances parcourues pour chaque camion
    total_distance = [0] * NB_TRUCKS
    
    for edge in graphe.es:
        for i in range(NB_TRUCKS):
            if edge["color"] == COLORS[i]:
                total_distance[i] += edge["weight"]

    # Afficher le graphe
    figures, axes = plt.subplots()
    igraph.plot(
        graphe,
        target=axes,
        layout=graphe.layout("kk"),
        vertex_size=50,
        vertex_label=graphe.vs["name"],
        vertex_label_size=10,
        vertex_color=["yellow" if v.attributes()["name"] == f"Ville {START_CITY}" else "red" if v.degree() > 0 else "black" for v in graphe.vs], # Les villes n'ayant aucun lien avec d'autre villes sont en noir
        edge_width=3,
        edge_color=graphe.es["color"],
        edge_label=[e["weight"] if e.attributes()["color"] != BASE_COLOR else "" for e in graphe.es]
    )

    custom_legend = [
        mlines.Line2D([0], [0], color=COLORS[i], lw=4, label=f"Camion {i} - {total_distance[i]}") for i in range(NB_TRUCKS)
    ]

    axes.legend(custom_legend, [line.get_label() for line in custom_legend], loc="upper right")

    plt.show()

if __name__ == "__main__":
    main()