import re

import networkx as nx


class GraphMixin:
    """Mermaid and NetworkX graph representations of the system topology."""

    def _node(self, name: str) -> str:
        """Unit name relative to the system (``socsys.HEX_F`` -> ``HEX_F``; also inside a Mermaid shape, ``t.feed((t.feed))`` ->
        ``feed((feed))``); the system itself keeps its name."""
        return re.sub(rf'(?<![\w.]){re.escape(self.name)}\.', '', name)

    def _edge(self, connection: list) -> tuple[str, str, str]:
        """
        (source, label, target) of a recorded connection [unit, port, port, unit]. A flange of the system is a node of its
        own (``F0 --C0--> HEX_F``): its name replaces the system name and the label is the port of the unit only.
        """
        unit0, out, inlet, unit1 = connection
        source, target = self._node(unit0), self._node(unit1)
        if unit0 == self.name and unit1 == self.name:
            return out, '', inlet
        if unit0 == self.name:
            return out, inlet, target
        if unit1 == self.name:
            return source, out, inlet
        return source, f'{out}_{inlet}', target

    def make_mermaid_graph(self, save: bool | str = True) -> str:
        """
        Create a Mermaid graph representation of the system.

        Args:
            save: ``True`` writes ``<system name>.mmd`` into the working directory (default),
                  a string is used as the file path, ``False`` writes nothing.

        Returns:
            str: Mermaid syntax for the graph.
        """
        mermaid_lines = ["flowchart LR\n"]

        for obj in self.getObjs():
            if obj != self:
                mermaid_lines.append(f"{self._node(obj.name)}\n")

        edges = [self._edge(connection) for connection in self.connections]
        known = {self._node(obj.name) for obj in self.getObjs() if obj != self}
        for node in dict.fromkeys(n for source, _, target in edges for n in (source, target)):
            if node not in known:                                   # a flange of the system
                mermaid_lines.append(f"{node}\n")
        for source, label, target in edges:
            mermaid_lines.append(f"\t{source} --{label}--> {target}\n" if label else f"\t{source} --> {target}\n")

        text = "".join(mermaid_lines)
        if save:
            path = save if isinstance(save, str) else f'{self.name}.mmd'
            with open(path, 'w') as f:
                f.write(text)
        return text

    def make_networkx_graph(self) -> nx.DiGraph:
        """
        Create a NetworkX graph representation of the system.

        Returns:
            nx.DiGraph: A directed graph representing the system's components and connections.
        """
        import matplotlib.pyplot as plt
        G = nx.DiGraph()

        for obj in self.getObjs():
            if obj != self:
                G.add_node(self._node(obj.name), type=obj.__class__.__name__, attributes=obj.__dict__)

        edge_labels = {}
        for connection in self.connections:
            source, label, target = self._edge(connection)
            G.add_edge(source, target)
            edge_labels[(source, target)] = label

        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
        return G

    def create_graph(self, graph_type: str = "mermaid", save: bool | str = True) -> nx.DiGraph | str:
        """
        Create a graph representation of the system based on the specified type.

        Args:
            graph_type (str): The type of graph to create ("networkx" or "mermaid").
            save: for "mermaid": write the graph to ``<system name>.mmd`` (True), to a given path (str) or not at all (False).

        Returns:
            Union[nx.DiGraph, str]: The generated graph (NetworkX object or Mermaid syntax).

        Raises:
            ValueError: If an unsupported graph type is specified.
        """
        self.connections = []
        self.seq_sol(store=True)
        if graph_type.lower() == "networkx":
            return self.make_networkx_graph()
        elif graph_type.lower() == "mermaid":
            return self.make_mermaid_graph(save=save)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}. Choose 'networkx' or 'mermaid'.")
