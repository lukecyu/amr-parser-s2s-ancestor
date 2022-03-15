import argparse
import penman
from penman.models.amr import model
from penman.layout import appears_inverted
from collections import Counter
import json


def convert_from_amr_to_graph(amr_graph):
    node_information = {}
    edge_information = {}
    attribute_information = {}

    for v, _, instance in amr_graph.instances():
        node_information[v] = instance
        attribute_information[v] = {}
        edge_information[v] = []

    for edge in amr_graph.edges():
        source, role, target = edge
        if appears_inverted(amr_graph, edge):
            source, role, target = model.invert(edge)
        edge_information[source].append((role[1:], target))

    for attribute in amr_graph.attributes():
        source, role, target = attribute

        attribute_information[source][role[1:]] = target

    return node_information, attribute_information, edge_information


def convert_amr_dfs(graph):
    top = graph.top
    node_info, attr_info, edge_info = graph.graph
    
    visited = set()
    
    s = []
    
    def explore(n):
        visited.add(n)
        s.extend('( ')
        s.extend(n)
        s.extend(' / ')
        s.extend(node_info[n])
        s.extend('\n')
        
        for edge, target in edge_info[n]:
            s.extend(':' + edge + ' ')
            if target not in visited:
                explore(target)
                s.extend('\n')
            else:
                s.extend(target)
                s.extend('\n')
        
        for edge, target in attr_info[n].items():
            s.extend(':' + edge + ' ')
            s.extend(target + '\n')
            
        s.extend(')')
        
    explore(top)
    
    return ''.join(s)
        
        


class AMRGraph:
    def __init__(self, amr_graph):
        self.top = amr_graph.top
        self.graph = convert_from_amr_to_graph(amr_graph)


def read_annotated_amr(file_path):
    for g in penman.iterdecode(open(file_path)):
        sentence = g.metadata['snt']
        idx = g.metadata['id']
        graph = AMRGraph(g)
        yield sentence, idx, graph
        
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', nargs='+')
    args = parser.parse_args()
    
#     no_quote_e = set()
    s = set()

    for f in args.file_path:
        for i, (sentence, idx, graph) in enumerate(read_annotated_amr(f)):
            node_info, attr_info, edge_info = graph.graph
            for v, attrs in attr_info.items():
                for e, a in attrs.items():
#                     if not a.startswith('"') or not a.endswith('"'):
#                         if not is_number(a):
#                             if a not in {'-', 'expressive', 'imperative', '+'}:
#                                 print(a)
#                                 no_quote_e.add(e)
#                     if (a.startswith('"') and not a.endswith('"')) or (not a.startswith('"') and a.endswith('"')):
#                         print(sentence)
                    if a.startswith('"') and a.endswith('"'):
                        if ' ' in a:
                            print(sentence)
                            print(a, node_info[v])
                            s.add(node_info[v])
#                         if '_' in a and node_info[v] == 'string-entity':
#                             print(sentence)
#                             print(a)
#                             print()
#                         if '_' in a and e not in {'wiki'} and node_info[v] != 'url-entity':
#                             print(sentence)
                            
#                             print(a, node_info[v])
#                             print()
#                         if a == '"United"':
#                             print(node_info[v])
        
                            
    
    print(s)
                    


    