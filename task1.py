import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from collections import deque

def getDistanceBetweenPoints2(lat1, lon1, lat2, lon2, **kwarg):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = R * c
    return round(d, 4)

# Задаємо лівий нижній кут карти (довгота, широта)
lb = (44.605, 22.295)

# Перелік обласних центрів України
cities = {
    "Kyiv": (50.45, 30.5233),
    "Kharkiv": (49.9925, 36.2311),
    "Odesa": (46.4775, 30.7326),
    "Dnipro": (48.4675, 35.04),
    "Donetsk": (48.0028, 37.8053),
    "Zaporizhzhia": (47.85, 35.1175),
    "Lviv": (49.8425, 24.0322),
    "Sevastopol": (44.605, 33.5225),
    "Mykolaiv": (46.975, 31.995),
    "Luhansk": (48.5667, 39.3333),
    "Vinnytsia": (49.2333, 28.4833),
    "Simferopol": (44.9484, 34.1),
    "Poltava": (49.5894, 34.5514),
    "Chernihiv": (51.4939, 31.2947),
    "Kherson": (46.6425, 32.625),
    "Cherkasy": (49.4444, 32.0597),
    "Khmelnytskyi": (49.4167, 27),
    "Chernivtsi": (48.3, 25.9333),
    "Sumy": (50.9167, 34.75),
    "Zhytomyr": (50.25, 28.6667),
    "Rivne": (50.6192, 26.2519),
    "Ivano-Frankivsk": (48.9228, 24.7106),
    "Kropyvnytskyi": (48.5, 32.2667),
    "Ternopil": (49.5667, 25.6),
    "Lutsk": (50.75, 25.3358),
    "Uzhhorod": (48.6239, 22.295),
}

# Переводимо координати у кілометри
cities2 = {
    n: (
        getDistanceBetweenPoints2(lb[0], x[1], x[0], x[1]),
        getDistanceBetweenPoints2(x[0], lb[1], x[0], x[1]),
    )
    for n, x in cities.items()
}

G = nx.Graph()

# Додаємо усі вузли до графа
for c, pos in cities2.items():
    G.add_node(c, pos=(int(pos[1]), int(pos[0])))

# Додаємо усі ребра, призначивши значення відстаней в якості ваги
for i in range(len(cities)):
    s_city = list(cities.keys())[i]
    s_value = list(cities.values())[i]
    for j in range(i + 1, len(cities)):
        t_city = list(cities.keys())[j]
        t_value = list(cities.values())[j]
        calc_distance = getDistanceBetweenPoints2(
            s_value[0], s_value[1], t_value[0], t_value[1]
        )
        G.add_edge(s_city, t_city, weight=calc_distance)

pos = nx.get_node_attributes(G, "pos")

# Виводимо повний граф
plt.figure(figsize=(15, 15))
nx.draw_networkx(G, pos=pos)
plt.show()

# Виводимо статистику повного графа
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
is_connected = nx.is_connected(G)

print("Graph before optimization")
print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")
print(f"Graph is connected: {is_connected}")

print(f"Nodes degree for Graph")
print(G.degree)

#Видаляємо усі маршрути, які мають дублі довші на 5%
short_coef = 1.05

remove_list = []

for i in range(len(cities)):
    s_city = list(cities.keys())[i]
    for j in range(len(cities)):
        t_city = list(cities.keys())[j]
        if not G.has_edge(s_city, t_city):
            continue
        dir_dist = G.get_edge_data(s_city, t_city)["weight"]

        for k in range(len(cities)):
            m_city = list(cities.keys())[k]
            if not G.has_edge(s_city, m_city) or not G.has_edge(t_city, m_city):
                continue
            dist1 = G.get_edge_data(s_city, m_city)["weight"]
            dist2 = G.get_edge_data(t_city, m_city)["weight"]

            if dist1 + dist2 < short_coef * dir_dist:
                remove_list.append((s_city, t_city))
                break

G.remove_edges_from(remove_list)

# Виводимо оптимізовану карту маршрутів
plt.figure(figsize=(15, 15))
nx.draw_networkx(G, pos=pos)
plt.show()


# Виводимо статистику
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
is_connected = nx.is_connected(G)

print("Graph after optimization")
print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")
print(f"Graph is connected: {is_connected}")

print(f"Nodes degree for Graph")
print(G.degree)

d=list(G.degree)
v=[a[1] for a in d]


print(f"Sorted Nodes for Graph")
print(sorted(d,key=lambda a: a[1],reverse=True))

def dfs_recursive(graph, vertex, visited=None):
    if visited is None:
        visited = set()
    visited.add(vertex)
    print(vertex, end=' ')  # Відвідуємо вершину
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)


print("DFS search")
dfs_recursive(G, 'Kyiv')



def bfs_recursive(graph, queue, visited=None):
    # Перевіряємо, чи існує множина відвіданих вершин, якщо ні, то ініціалізуємо нову
    if visited is None:
        visited = set()
    # Якщо черга порожня, завершуємо рекурсію
    if not queue:
        return
    # Вилучаємо вершину з початку черги
    vertex = queue.popleft()
    # Перевіряємо, чи відвідували раніше дану вершину
    if vertex not in visited:
        # Якщо не відвідували, друкуємо вершину
        print(vertex, end=" ")
        # Додаємо вершину до множини відвіданих вершин.
        visited.add(vertex)
        # Додаємо невідвіданих сусідів даної вершини в кінець черги.
        queue.extend(set(graph[vertex]) - visited)
    # Рекурсивний виклик функції з тією ж чергою та множиною відвіданих вершин
    bfs_recursive(graph, queue, visited)

print()
print("BFS search")
bfs_recursive(G, deque(["Kyiv"]))


def dijkstra(graph, start):
    # Ініціалізація відстаней та множини невідвіданих вершин
    distances = {vertex: float('infinity') for vertex in graph}
    pathes = {vertex: [start] for vertex in graph}
    distances[start] = 0
    unvisited = list(graph.nodes())



    while unvisited:
        # Знаходження вершини з найменшою відстанню серед невідвіданих
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

        # Якщо поточна відстань є нескінченністю, то ми завершили роботу
        if distances[current_vertex] == float('infinity'):
            break
            
        for neighbor, weight in graph[current_vertex].items():
            # print(f"{current_vertex = } *** {neighbor = }")
            # print(f"{weight = }")
            distance = distances[current_vertex] + weight['weight']

            # Якщо нова відстань коротша, то оновлюємо найкоротший шлях
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                new_path=pathes[current_vertex][:]
                new_path.append(neighbor)
                pathes[neighbor]=new_path

        # Видаляємо поточну вершину з множини невідвіданих
        unvisited.remove(current_vertex)

    return distances,pathes

res=dijkstra(G, 'Kyiv')

print()

print(res[0])

print()

print(res[1]) 


