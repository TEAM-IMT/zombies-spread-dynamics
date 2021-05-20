## Libraries ###########################################################
import os, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorama, time
import datetime as dt
colorama.init()

import networkx as nx

## Functions and Class #################################################
class spread_zombie_dynamics:
    """
    Class with functions modeling the spread of a zombie epidemic

    Attributes
    ----------
    graph : networkx.classes.digraph.DiGraph
        Directional networkx library graph with {node_id (optional), human_pop, zombie_pop}
        as attributes for each node and {elev_factor} for each edge
    
    current_humans: int
        Current population of humans in the entire network

    current_zombies: int
        Current population of zombies in the entire network

    current_date: datetime.datetime
        Current process date

    evolution: pandas.DataFrame
        Datetime dataframe with the information by days of the populations of both communities

    INITIAL_DATE : datetime.datetime
        Day of onset of the epidemic spread

    MAX_ZOMBIE_AGE : int
        Number of days that a zombie will travel through the terrain before self-exploding

    TOTAL_POPULATION: int
        Total number of humans + zombies + zombies killed

    MILITARY_TROPS : dict
        Dictionary with date and cells in trop military process

    NUCLEAR_BOMBS : dict
        Dictionary with date and cells of nuclear bomb ignition
    
    Methods
    -------
    reset()
        Simulation restart, setting original values
    
    run()
        Execute one step in dynamic process
    
    plot_evolution(self, ax: plt.axes = None, **kwargs: dict)
        Dynamic function that draw the evolution of both populations

    plot_graph(self, ax: plt.axes = None, type: str = 'both', **kwargs: dict)
        Dynamic function that draw the evolution of populations on graph

    plot_all(self, axs: str = None, **kwargs: dict)
        Dynamic function that show a subplot with plot_evolution and plot_graph results

    plot_zombie_age(self, ax: plt.axes = None, **kwargs: dict)
        Dynamic function that draw the evolution of zombie subpopulation
    """
    
    ## Constructor
    def __init__(self, graph: nx.classes.digraph.DiGraph, INTIAL_DATE: dt.datetime, MAX_ZOMBIE_AGE: int = 15,
                MILITARY_TROPS: dict = None, NUCLEAR_BOMBS: dict = None):
        """
        Parameters
        ----------
        graph : networkx.classes.digraph.DiGraph
            Directional networkx library graph with the following attributes for each node (ni):
                - node_id, str (optional): Node identification 
                - human_pop, int: Initial population of humans
                - zombie_pop, int: Initial population of zombies
            And the following attributes for each edge (ni,nj):
                - elev_factor, float: Elevation factor in the direction ni -> nj between the range [0,1], 
                    with 0 no slope and 1 high slope. Influencing the spread of the epidemic.
        
        INITIAL_DATE : datetime.date
            Day of onset of the epidemic spread

        MAX_ZOMBIE_AGE : int, optional (default : 15 days)
            Number of days that a zombie will travel through the terrain before self-exploding.
        
        MILITARY_TROPS : dict, optional (default : None)
            Dictionary with information about military trops deployment, where keys are the date
            and values are nodes_id involved in the process

        NUCLEAR_BOMBS : dict, optional (default : None)
            Dictionary with information about nuclear bombs deployment, where keys are the date
            and values are nodes_id involved in the process
        """
        graph.name = 'Zombie epidemic spread dynamics graph'
        self.graph = graph
        self.INTIAL_DATE = INTIAL_DATE
        self.MAX_ZOMBIE_AGE = MAX_ZOMBIE_AGE
        self.MILITARY_TROPS = MILITARY_TROPS
        self.NUCLEAR_BOMBS = NUCLEAR_BOMBS
        self.reset()
    
    def __setattr__(self, name: any, value: any):
        """
        Implement setattr(self, name, value), with assert events if attributes are wrong
        """
        if name == 'graph':
            error = colorama.Fore.RED + "[ERROR] Graph hasn't {} attribute by each {}" + colorama.Fore.RESET
            assert len(nx.get_node_attributes(value, 'human_pop')) == len(value.nodes), error.format('human_pop','node')
            assert len(nx.get_node_attributes(value, 'zombie_pop')) == len(value.nodes), error.format('zombie_pop','node')
            assert len(nx.get_edge_attributes(value, 'elev_factor')) == len(value.edges), error.format('elev_factor','edge')
            if len(nx.get_node_attributes(value, 'node_id')) == len(value.nodes):
                value = nx.relabel_nodes(value, nx.get_node_attributes(value, 'node_id')) # Rename node_label
            self._ini_human_pop = nx.get_node_attributes(value, 'human_pop')
            self._ini_zombie_pop = nx.get_node_attributes(value, 'zombie_pop')
            print("[INFO] Graph was modified ...")
        if (name == 'MILITARY_TROPS' or name == 'NUCLEAR_BOMBS') and value is not None:
            error = colorama.Fore.RED + "[ERROR] {} of {} attribute must be {}." + colorama.Fore.RESET
            assert all(map(lambda x: type(x) == dt.date, value.keys())), error.format("Keys", name, "DATE")
            assert all(map(lambda x: type(x) == list, value.values())), error.format("Values", name, "LIST")
            cells = set(sum(value.values(), []))
            cells = cells - (self.graph.nodes & cells)
            if len(cells) > 0: 
                error = "[WARNING] {} has {} nodes that don't exist in graph. They will be ignored.".format(name, cells)
                print(colorama.Fore.YELLOW + error + colorama.Fore.RESET)
        self.__dict__[name] = value

    def __str__(self):
        s = "-"*30 + "\nINITIAL GRAPH DESCRIPTION:\n" + nx.info(self.graph) + "\n"
        human_pop, zombie_pop = self.evolution.iloc[0]['human_pop'], self.evolution.iloc[0]['zombie_pop']
        s += "Initial date of epidemic:\t{}\n".format(self.INTIAL_DATE)
        s += "Initial human population: \t{0} ({1:.2f}\% of all population)\n".format(human_pop, 100*human_pop/self.TOTAL_POPULATION)
        s += "Initial zombie population: \t{0} ({1:.2f}\% of all population)\n".format(zombie_pop, 100*zombie_pop/self.TOTAL_POPULATION)

        s += "\nCURRENT GRAPH DESCRIPTION\n"
        human_pop, zombie_pop = self.evolution.iloc[-1]['human_pop'], self.evolution.iloc[-1]['zombie_pop']
        s += "Date of epidemic:\t\t{}\n".format(self.current_date)
        s += "Total human population: \t{0} ({1:.2f}\% of all population)\n".format(human_pop, 100*human_pop/self.TOTAL_POPULATION)
        s += "Total zombie population: \t{0} ({1:.2f}\% of all population)\n".format(zombie_pop, 100*zombie_pop/self.TOTAL_POPULATION)
        s += "-"*30
        return s

    ## User methods
    def reset(self):
        """
        Simulation restart, setting original values
        """
        # Restart graph attributes
        nx.set_node_attributes(self.graph, self._ini_human_pop, name = 'human_pop')
        nx.set_node_attributes(self.graph, self._ini_zombie_pop, name = 'zombie_pop')

        # Internal subsets with age for each zombie-pop 
        self._subpop_zombies = pd.DataFrame(self._ini_zombie_pop, index = ['age_0']).T
        self._subpop_zombies[['age_' + str(x) for x in range(1, self.MAX_ZOMBIE_AGE)]] = 0

        # Restart control variables
        self._military_nodes, self._nuclear_nodes = set(), set()
        self.current_date = self.INTIAL_DATE - dt.timedelta(days = 1) # +1 day in update function
        self.evolution = pd.DataFrame() # Errase all evolution info
        self._update()
        self.TOTAL_POPULATION = self.total_humans + self.total_zombies
        
        # Graph controls
        self._fig_all, self._axs_all = None, None
        self._fig_evol, self._ax_evol = None, None
        self._fig_zombie, self._ax_zombie = None, None
        self._fig_graph, self._ax_graph, self._graph_pos, self._colorbar = None, None, None, None
        plt.close('all')

    def run(self):
        """
        Execute one step in dynamic process, which implies:
            - Step 0: Update zombies age and execute actions: (military trops and/or nuclear bombs)
            - Step 1: Epidemic spread by contribution of all neighbords
            - Step 2 : Estimates new populations after interacting on each node
            - Step 3: Update evolutional attributes
        """
        self._age_update() # Step 0
        self._zombies_propagation() # Step 1
        self._zombie_human_interactions() # Step 2
        self._update() # Step 3

    def plot_evolution(self, ax: plt.axes = None, **kwargs: dict):
        """
        Create or update a current plot to show the evolution of populations in experiment
        
        Parameters
        ----------
            ax: matplotlib.pyplot.axes, optional (default : None)
                Axes to draw the evolution plot. By default in inner axes

            **kwargs: dict
                Optional parameters to plt.subplots() function
        
        Returns
        -------
            ax_plot: matplotlib.pyplot.axes
                Axes where the evolution plot was drawed.
        """
        ax_plot = self.__preplot('_fig_evol', '_ax_evol', ax) # Preprocess
        ax_plot = self.evolution.plot(use_index = True, y = ['zombie_pop', 'human_pop'], kind = 'line', ax = ax_plot,
                                    xlabel = 'Date', ylabel = 'Population', marker = '.', cmap = plt.get_cmap('jet'))
        self.__postplot('_fig_evol') # Postprocess
        return ax_plot

    def plot_zombie_age(self, ax: plt.axes = None, **kwargs: dict):
        """
        Create or update a current plot to show the evolution of zombie subpopulations
        
        Parameters
        ----------
            ax: matplotlib.pyplot.axes, optional (default : None)
                Axes to draw the evolution plot. By default in inner axes

            **kwargs: dict
                Optional parameters to plt.subplots() function
        
        Returns
        -------
            ax_plot: matplotlib.pyplot.axes
                Axes where the evolution plot was drawed.
        """
        ax_plot = self.__preplot('_fig_zombie', '_ax_zombie', ax) # Preprocess
        ax_plot = self.evolution.plot(use_index = True, y = ['age_' + str(x) for x in range(self.MAX_ZOMBIE_AGE)], kind = 'line', 
                                ax = ax_plot, xlabel = 'Date', ylabel = 'Zombie population', marker = '.', cmap = plt.get_cmap('tab20'))
        self.__postplot('_fig_zombie') # Postprocess
        return ax_plot

    def plot_graph(self, ax: plt.axes = None, type: str = 'both', **kwargs: dict):
        """
        Plot current state of network, with three different behaviours:
            - Zombie population (type = 'zombie')
            - Human population (type = 'human')
            - Difference between human and zombie population (type = 'both')

        Parameters
        ----------
            ax: matplotlib.pyplot.axes, optional (default : None)
                Axes to draw the network states. By default in inner axes

            type: str, optional (default : None)
                Definition of kind of plot (human, zombie or both)

            **kwargs: dict
                Optional parameters to plt.subplots() function
        
        Returns
        -------
            ax_plot: matplotlib.pyplot.axes
                Axes where the network states plot was drawed.
        """
        # Preprocess
        if self._graph_pos is None: self._graph_pos = nx.spring_layout(self.graph, iterations = 1000)
        ax_plot = self.__preplot('_fig_graph', '_ax_graph', ax, **kwargs)

        # Calculate populations in each node, and define them as color
        vmin, vmax = 0, self.evolution.loc[self.current_date, 'max_human_pop_node']
        midpoint = None
        if type == "zombie":
            vmax = self.evolution.loc[self.current_date, 'max_zombie_pop_node']
            if vmax == 0: node_color = [-1]*len(self.graph)
            else: node_color = [2*self.graph.nodes[node]['zombie_pop']/vmax - 1 for node in self.graph]
            label = "Zombie population per node"
        elif type == "human":
            if vmax == 0: node_color = [-1]*len(self.graph)
            else: node_color = [2*self.graph.nodes[node]['human_pop']/vmax - 1 for node in self.graph]
            label = "Zombie population per node"
        elif type == "both":
            midpoint = 0
            vmin = self.evolution.loc[self.current_date, 'max_zombie_pop_node']
            node_color = []
            for node in self.graph: # Change scale of color
                node_color.append(self.graph.nodes[node]['human_pop'] - self.graph.nodes[node]['zombie_pop'])
                if node_color[-1] < 0: node_color[-1] = 0 if vmin == 0 else node_color[-1]/vmin
                else: node_color[-1] = 0 if vmax == 0 else node_color[-1]/vmax
            label = "Human - zombie difference per node"
        else:
            raise colorama.Fore.RED + "[ERROR] Choose one type inside ['zombie','human','both']." + colorama.Fore.RESET
        if midpoint is None: midpoint = vmax/2

        # Plot network and colorbar
        nx.draw_networkx_edges(self.graph, self._graph_pos, edge_color = 'k', ax = ax_plot, arrows = False, alpha = 0.4)
        plot = nx.draw_networkx_nodes(self.graph, self._graph_pos, cmap = plt.get_cmap('jet'), ax = ax_plot,
                            node_size = 10, node_color = node_color, vmin = -1, vmax = 1)
        
        if self._colorbar is not None: self._colorbar.remove() # Remove previous colorbar

        self._colorbar = ax_plot.figure.colorbar(plot, ax = ax_plot, label = label, ticks = [-1, 0, 1])
        self._colorbar.ax.set_yticklabels(["{} zom.".format(vmin), "{} pop.".format(midpoint), "{} hum.".format(vmax)])
        ax_plot.set_xlabel("Current day : {0:%b. %d, %Y}".format(self.current_date))

        # limits = np.array(list(self._graph_pos.values()))
        # limits = [cut * limits[:,0].max(), cut * limits[:,1].max()]
        # ax_plot.set_xlim([-limits[0], limits[0]])
        # ax_plot.set_ylim([-limits[1], limits[1]])
        
        self.__postplot('_fig_graph') # Postprocess
        return ax_plot

    def plot_all(self, axs: str = None, type : str = 'both', **kwargs: dict):
        """
        Plot current states of networks, in two plots:
            - Network architecture
            - Population evolution

        Parameters
        ----------
            axs: list of matplotlib.pyplot.axes, optional (default : None)
                List of axes to draw the network states. len(axs) == 2

            type: str, optional (default : None)
                Definition of kind of plot for plot_graph (human, zombie or both)
            
            **kwargs: dict
                Optional parameters to plt.subplots() function
        
        Returns
        -------
            axs_plot: list of matplotlib.pyplot.axes
                List of axes where the network states plot was drawed.
        """
        axs_plot = self.__preplot('_fig_all', '_axs_all', axs, ncols = 2, figsize = (10,5), **kwargs)
        self.plot_evolution(ax = axs_plot[0])
        self.plot_graph(ax = axs_plot[1], type = type)
        self.__postplot('_fig_all')
        return axs_plot

    ## Inner methods
    def _age_update(self):
        """
        Update zombies age in 1 day for all graph, killing zombies pop with age > MAX_ZOMBIE_AGE
        """
        self._subpop_zombies.iloc[:,1:] = self._subpop_zombies.iloc[:,:-1]
        self._subpop_zombies.iloc[:,0] = 0

        # Zombies will kill by external agents if condition apply
        if self._trigger: 
            self._subpop_zombies.loc[self._military_nodes | self._nuclear_nodes] = 0
        
        # Human will kill by external agents if condition apply
        if self._trigger: 
            df_pop = pd.DataFrame(pd.Series(nx.get_node_attributes(self.graph, 'human_pop'), name = 'human_pop'))
            df_pop.loc[self._nuclear_nodes] = 0
            nx.set_node_attributes(self.graph, df_pop['human_pop'].to_dict(), name = 'human_pop')

    def _zombies_propagation(self):
        """
        Estimate contribution of zombies from neighboring nodes to current node (C(c0,ci)) to update zombies population.
        """
        # Zombie contribution in all graph + (ci,ci) contribution (with itself)
        index_edges = set(list(self.graph.edges) + [(n,n) for n in self.graph.nodes])
        df_C = pd.DataFrame(index = index_edges, columns = self._subpop_zombies.columns)
        df_C = df_C.apply(lambda x: self.__zombies_contribution(x.name), axis = 1, result_type = 'broadcast')
        df_C.index = pd.MultiIndex.from_tuples(df_C.index, names = ('c0','ci'))

        # Calculate zombie pop that didn't move
        df_zhat = self._subpop_zombies - df_C.groupby(level = 'c0').agg(np.sum)

        # Zombie spread and update in network
        self._subpop_zombies = df_C.groupby(level = 'ci').agg(np.sum) + df_zhat
        nx.set_node_attributes(self.graph, self._subpop_zombies.sum(axis = 1).to_dict(), name = 'zombie_pop')

    def _zombie_human_interactions(self):
        """
        Population control of humans and zombies after their interaction at the same node, under two main steps:
        - Zombies kill humans, which are transformed into zombies.
        - Humans kill zombies, which disappear from the total population.
        """
        # Step 0: Current human and zombie population
        df_pop = pd.DataFrame(pd.Series(nx.get_node_attributes(self.graph, 'human_pop'), name = 'human_pop'))
        df_pop['zombie_pop'] = pd.Series(nx.get_node_attributes(self.graph, 'zombie_pop'))

        # Step 1: Zombies kill humans : No negative values its possible
        df_pop['survivors'] = (df_pop['human_pop'] - 10*df_pop['zombie_pop']).clip(lower = 0)
        
        # Step 2: Humans dead are transform into zombies with '0' days as age
        self._subpop_zombies.iloc[:,0] = df_pop['human_pop'] - df_pop['survivors']

        # Step 3: Humans kill zombies, with the same proba for all ages
        df_pop['human_pop'] = df_pop['survivors']
        df_pop['zombie_pop'] = self._subpop_zombies.sum(axis = 1) # Update new total zombie population
        df_pop['proba'] = (1 - 10*df_pop['human_pop']/df_pop['zombie_pop']).clip(lower = 0.0, upper = 1.0).fillna(0.0)
        self._subpop_zombies = self._subpop_zombies.mul(df_pop['proba'], axis = "index").applymap(np.floor).applymap(int)

        # Step 4: Update values on graph
        df_pop['zombie_pop'] = self._subpop_zombies.sum(axis = 1) # Update new total zombie population
        nx.set_node_attributes(self.graph, df_pop[['zombie_pop', 'human_pop']].T.to_dict())

    def _update(self):
        """
        Update all attributes
        """
        # Update population
        humans = nx.get_node_attributes(self.graph, 'human_pop').values()
        zombies = nx.get_node_attributes(self.graph, 'zombie_pop').values()
        self.total_humans, self.total_zombies = sum(humans), sum(zombies)
        
        # Update new register
        new_date = {'human_pop': self.total_humans, 'zombie_pop': self.total_zombies,
                    'max_human_pop_node': max(humans), 'max_zombie_pop_node': max(zombies)} # New info of populations
        new_date.update(self._subpop_zombies.sum().to_dict()) # With subpopulation summary
        self.current_date = self.current_date + dt.timedelta(days = 1)
        new_date = pd.DataFrame(new_date, index = [self.current_date])
        self.evolution = self.evolution.append(new_date) # Add new population with date
        
        # Update military and nuclear cells and trigger event
        self._trigger = False
        if self.MILITARY_TROPS is not None and len([x for x in self.MILITARY_TROPS.keys() if x == self.current_date]) > 0:
            self._trigger = True
            self._military_nodes = filter(lambda x: x[0] <= self.current_date, self.MILITARY_TROPS.items())
            self._military_nodes = set(sum(map(lambda x: x[1], self._military_nodes), []))
            self._military_nodes = self._military_nodes.intersection(self.graph.nodes)
        if self.NUCLEAR_BOMBS is not None and len([x for x in self.NUCLEAR_BOMBS.keys() if x == self.current_date]) > 0:
            self._trigger = True
            self._nuclear_nodes = filter(lambda x: x[0] <= self.current_date, self.NUCLEAR_BOMBS.items())
            self._nuclear_nodes = set(sum(map(lambda x: x[1], self._nuclear_nodes), []))
            self._nuclear_nodes = self._nuclear_nodes.intersection(self.graph.nodes)

    ## Other methods
    def __zombies_contribution(self, edge):
        """
        Estimate contribution of zombies C(c0,ci), with ci neighbord of c0 or ci = c0, taking into account 
        military trops or nuclear bombs effects
        """
        forbidden_cells = self._military_nodes | self._nuclear_nodes
        sum_human_pop = sum([self.graph.nodes[n]['human_pop']*self.graph.edges[(edge[0],n)]['elev_factor'] 
                            for n in nx.neighbors(self.graph, edge[0]) if n not in forbidden_cells])
        elev_factor = self.graph.edges[edge]['elev_factor'] if edge[1] not in forbidden_cells and edge[0] != edge[1] else 0.0
        
        if edge[0] != edge[1] and sum_human_pop > 0:
            C = elev_factor*self.graph.nodes[edge[1]]['human_pop']/sum_human_pop
            C = np.floor(C * self._subpop_zombies.loc[edge[0]].values).astype(int)
        elif edge[0] == edge[1] and sum_human_pop == 0:
            C = self._subpop_zombies.loc[edge[0]].values
        else:
            C = np.array([0]*self.MAX_ZOMBIE_AGE)
        return C

    def __preplot(self, figname, axname, ax, **kwargs):
        self._trigger = False
        if self.__dict__[figname] is None and ax is None: 
            if '_fig_all' == figname:
                self.__dict__[figname], self.__dict__[axname] = plt.subplots(**kwargs)
            else:
                self.__dict__[figname], self.__dict__[axname] = plt.subplots(**kwargs)
            self._trigger = True
        
        ax_plot = self.__dict__[axname] if ax is None else ax
        plt.ion() # Enable interactive plots
        if '_fig_all' != figname: ax_plot.cla() # Remove previous plots
        return ax_plot

    def __postplot(self, figname):
        if self._trigger: self.__dict__[figname].tight_layout() # Border reduction
        plt.pause(0.001) # Short time to redraw plot
        plt.ioff() # Disable interactive plots

## Main ################################################################
def graph_by_default(nodes = 5, isprint = False):
    G = nx.grid_2d_graph(nodes, nodes).to_directed()
    new_edges = [[((i,j),(i-1,j-1)),((i,j),(i-1,j+1)), ((i,j),(i+1,j-1)),((i,j),(i+1,j+1))] 
                for i in range(1,nodes-1) for j in range(1,nodes-1)]
    new_edges += [[((1,0),(0,1)), ((0,nodes-2),(1,nodes-1)), ((nodes-2,0),(nodes-1,1)), ((nodes-2,nodes-1),(nodes-1,nodes-2))]]
    G.add_edges_from(sum(new_edges,[]))
    nx.set_node_attributes(G, {n: {'node_id': 'U' + str(i), 'human_pop': 1500, 'zombie_pop': 0} 
                                for i, n in enumerate(G.nodes)})
    G = nx.relabel_nodes(G, nx.get_node_attributes(G, 'node_id')) # Rename node_label

    # Initial zombie pop in only one node ('middle')
    G.nodes['U' + str(nodes//2)].update({'zombie_pop': 1000, 'human_pop': 0}) 
    G.add_edges_from([(e[1],e[0]) for e in G.edges]) # Bidirectional edges
    nx.set_edge_attributes(G, {e:0.3 for e in G.edges}, name = 'elev_factor')

    if isprint:
        print("[INFO] Graph info: ")
        print(nx.info(G))
        print("\n[INFO] Attributes info: ")
        for key in ['node_id', 'human_pop', 'zombie_pop']:
            print(key, ":", nx.get_node_attributes(G, key))
        print("\n[INFO] Edge info: \nelev_factor:", nx.get_edge_attributes(G, 'elev_factor'))
    return G

if __name__ == "__main__":
    os.system('clear'); os.system('clear')
    G = graph_by_default(20)
    initial_date = dt.datetime(year = 2019, month = 8, day = 18)
    spread_dynamic = spread_zombie_dynamics(G, INTIAL_DATE = initial_date)
    for i in tqdm.tqdm(range(50)):
        spread_dynamic.plot_all(type = 'both')
        if i % 5 == 0 or i == 29:
            print(spread_dynamic)
        spread_dynamic.run()
    plt.show()
