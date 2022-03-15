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
        
#         if role == ':wiki':
#             attribute_information[source][role[1:]] = '+'
#         else:

        attribute_information[source][role[1:]] = target

    return node_information, attribute_information, edge_information


def convert_amr_dfs(graph, remove_wiki=False):
    top = graph.top
    node_info, attr_info, edge_info = graph.graph
#    print(attr_info)    
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
#            print(edge, target)
            if remove_wiki == True:
                if edge != 'wiki':
                    s.extend(':' + edge + ' ')
                    s.extend(target + '\n')
            else:
                s.extend(':' + edge + ' ')
                s.extend(target + '\n')


#            s.extend(':' + edge + ' ')
#            s.extend(target + '\n')
            
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', nargs='+')
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

#     attribute_role_set = set()
#     op_set = set()
#     snt_set = set()
#     pp_value = set()
#     name_role_set = set()
#     concept_set = set()
#     concept_new_set = set()
#     n1 = 0
#     n2 = 0
#     n3 = 0

#     node_number_dict = Counter()
#     edge_number_dict = Counter()
#     sum_number_dict = Counter()
#     source_len_dict = Counter()
    with open(args.output_path, 'w') as f_w:
        for f in args.file_path:
            for i, (sentence, idx, graph) in enumerate(read_annotated_amr(f)):
                f_w.write('# ::id ')
                f_w.write(idx)
                f_w.write('\n')
                f_w.write('# ::snt ')
                f_w.write(sentence)
                f_w.write('\n')
                f_w.write(convert_amr_dfs(graph, remove_wiki=True))
                f_w.write('\n\n')

#             print(convert_amr_dfs(graph))
#             print()
#             print()
        # if len(sentence) <= 200:
#         flag = 0
#         nodes, attributes, edges = graph.graph
#         node_number = len(nodes)
#         edge_number = 0
#         for _, attribute_dict in attributes.items():
#             edge_number += len(attribute_dict)
#         for _, edge_dict in edges.items():
#             edge_number += len(edge_dict)
#         if node_number + edge_number <= 99 and len(sentence.split()) <= 64:
#             node_number_dict[node_number] += 1
#             edge_number_dict[edge_number] += 1
#             sum_number_dict[node_number + edge_number] += 1
#             source_len_dict[len(sentence.split())] += 1

#     node_number_list = [(key, value) for key, value in node_number_dict.items()]
#     edge_number_list = [(key, value) for key, value in edge_number_dict.items()]
#     source_len_list = [(key, value) for key, value in source_len_dict.items()]
#     sum_number_list = [(key, value) for key, value in sum_number_dict.items()]

#     node_number_list.sort()
#     edge_number_list.sort()
#     sum_number_list.sort()
#     source_len_list.sort()

#     print(node_number_list)
#     print(edge_number_list)
#     print(sum_number_list)
#     print(source_len_list)

#     print(sum(node_number_dict.values()), sum(edge_number_dict.values()), sum(source_len_dict.values()))

    
