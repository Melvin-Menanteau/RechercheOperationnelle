import numpy
import igraph
import matplotlib as matplot

# Paramètres de la simulation
NB_TRUCKS = 5                   # Nombre de camions
NB_VERTICES = 15               # Nombre de villes
PROBABILITY_EDGE = 0.4          # Probabilité qu'il y ait une arête entre deux sommets
MIN_WEIGHT = 10                 # Poids minimal d'une arête
MAX_WEIGHT = 500                # Poids maximal d'une arête

# Créer un graphe aléatoire avec nb_vertices sommets
def create_graph(nb_vertices):
    graph = igraph.Graph.Erdos_Renyi(nb_vertices, PROBABILITY_EDGE, directed=False)

    graph["title"] = "Graphe de villes"
    graph.vs["name"] = ["Ville " + str(i + 1) for i in range(nb_vertices)]

    # La fonction simplify() permet de supprimer les arêtes en double et donc d'alléger la structure du graphe et la recherche de chemins
    graph.simplify()

    set_random_weights(graph)

    return graph

def get_adjacency_matrix(graph):
    adjacency_matrix = numpy.zeros((graph.vcount(), graph.vcount()))

    for edge in graph.es:
        u, v = edge.source, edge.target
        weight = edge['weight']
        adjacency_matrix[u, v] = weight
        adjacency_matrix[v, u] = weight  # Parce que le graphe est non directionnel

    return adjacency_matrix

# Permet de définir un poids pour chaque arête du graphe
# Permet aussi de simuler une variation dans le traffic, en changeant les poids lors de l'exécution
def set_random_weights(graph):
    for edge in graph.es:
        edge["weight"] = numpy.random.randint(MIN_WEIGHT, MAX_WEIGHT)
        edge["label"] = edge["weight"]

def main():
    graph = create_graph(nb_vertices=NB_VERTICES)
    print(get_adjacency_matrix(graph))

    # Afficher le graphe
    figures, axes = matplot.pyplot.subplots()
    igraph.plot(
        graph,
        target=axes,
        layout=graph.layout("kk"),
        vertex_size=30,
        vertex_label=graph.vs["name"],
        vertex_label_size=5,
        # Si la ville n'a pas de voisins, elle sera affichée en noir
        vertex_color=["red" if v.degree() > 0 else "black" for v in graph.vs],
        edge_width=1,
        edge_color="#e8e8e8",
        # edge_label=graph.es["label"],
        # edge_label_size=10
    )

    matplot.pyplot.show()

if __name__ == "__main__":
    main()