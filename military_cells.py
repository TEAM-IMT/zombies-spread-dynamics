from libraries.dynamics import spread_zombie_dynamics as szd
from libraries.dynamics import graph_by_default
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
import datetime as dt
import tqdm

G = graph_by_default(nodes = 10)
ini_date = dt.datetime(year = 2019, month = 8, day = 18)
dynamic = szd(graph = G, INTIAL_DATE = ini_date)


graph_2months = 0
for epoch in tqdm.tqdm(range(60)): # Just 20 epochs
    dynamic.step() # Run one step in dynamic procedure
    if epoch == 59 : graph_2months = dynamic.graph
    print(dynamic) # See basic statistics at each iteration


vertext_cover = list(min_weighted_vertex_cover(graph_2months))
nodes = sorted(graph_2months.degree, key=lambda x: x[1], reverse=True)
print(nodes, '\n', vertext_cover)