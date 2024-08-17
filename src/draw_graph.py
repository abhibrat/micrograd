from graphviz import Digraph

from src.engine import Value


def draw(root):
  g= Digraph(format='svg', graph_attr={'rankdir':'LR'})
  visited=set()
  def build(node: Value, successor=None):
    # if node in visited:
    #   return
    # visited.add(node)
    g.node(name=str(id(node)), label="{%s | data %0.4f | grad %0.4f}" %(node.label, node.data, node.grad), shape='record')
    if successor and successor._op:
      g.edge(str(id(node)), str(id(successor))+successor._op)

    if node in visited:
      return
    visited.add(node)

    if node._op:
      g.node(name=str(id(node))+node._op, label=node._op)
      g.edge(str(id(node))+node._op, str(id(node)))

    for x in node._prev:
      build(x, node)
  build(root)
  return g

