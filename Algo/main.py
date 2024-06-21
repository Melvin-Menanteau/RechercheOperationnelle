import numpy
import igraph
import matplotlib as matplot

NB_VERTICES = 50
PROBABILITY_EDGE = 0.2
MIN_WEIGHT = 10
MAX_WEIGHT = 500

# nb_vertices : Nombre de sommets (villes) dans le graphe
def create_graph(nb_vertices):
    # Créer un graphe aléatoire avec nb_vertices sommets
    # PROBABILITY_EDGE est la probabilité qu'il y ait une arête entre deux sommets
    graph = igraph.Graph.Erdos_Renyi(nb_vertices, PROBABILITY_EDGE, directed=False)

    graph["title"] = "Graphe de villes"
    graph.vs["name"] = ["Ville " + str(i) for i in range(nb_vertices)]

    # La fonction simplify() permet de supprimer les arêtes en double et donc d'alléger la structure du graphe et la recherche de chemins
    graph.simplify()

    for edge in graph.es:
        edge["weight"] = numpy.random.randint(MIN_WEIGHT, MAX_WEIGHT)
        edge["label"] = edge["weight"]

    return graph

def get_adjacency_matrix(graph):
    adjacency_matrix = numpy.zeros((graph.vcount(), graph.vcount()))

    for edge in graph.es:
        u, v = edge.source, edge.target
        weight = edge['weight']
        adjacency_matrix[u, v] = weight
        adjacency_matrix[v, u] = weight  # Parce que le graphe est non directionnel

    return adjacency_matrix

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