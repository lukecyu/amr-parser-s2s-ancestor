import glob
from typing import List, Union, Iterable
from pathlib import Path
from ancestor_amr.penman import load as pm_load
from ancestor_amr.dfs import AMRGraph, convert_amr_dfs, read_annotated_amr

def read_raw_amr_data(
        paths: List[Union[str, Path]],
        use_recategorization=False,
        dereify=True,
        remove_wiki=False,
):
    print(paths)
    assert paths

    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = []
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)    
            graphs.extend(pm_load(path, dereify=dereify, remove_wiki=remove_wiki))

    assert graphs
    
    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata['snt_orig'] = metadata['snt']
            tokens = eval(metadata['tokens'])
            metadata['snt'] = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])

    return graphs

def read_raw_amr_data_new(
        paths: List[Union[str, Path]],
        use_recategorization=False,
        dereify=True,
        remove_wiki=False,
):
    print(paths)
    assert paths

    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = []
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)
            
            for i, (sentence, idx, graph, g_origin) in enumerate(read_annotated_amr(path)):
                g = convert_amr_dfs(graph, remove_wiki)
                if use_recategorization:
                    metadata = g_origin.metadata
                    metadata['snt_orig'] = metadata['snt']
                    tokens = eval(metadata['tokens'])
                    sentence = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])
                graphs.append([sentence, idx, g, graph, g_origin])
#                 f_w.write('# ::id ')
#                 f_w.write(idx)
#                 f_w.write('\n')
#                 f_w.write('# ::snt ')
#                 f_w.write(sentence)
#                 f_w.write('\n')
#                 f_w.write(convert_amr_dfs(graph))
#                 f_w.write('\n\n')
                
                
#             graphs.extend(pm_load(path, dereify=dereify, remove_wiki=remove_wiki))

#     assert graphs
    
#     if use_recategorization:
#         for g in graphs:
#             metadata = g.metadata
#             metadata['snt_orig'] = metadata['snt']
#             tokens = eval(metadata['tokens'])
#             metadata['snt'] = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])

    return graphs
