import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from ancestor_amr.IO import read_raw_amr_data, read_raw_amr_data_new

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        add_parents_attention=False,
        add_parents_embedding=False,
        add_siblings_attention=False,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data_new(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.graph_origins = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.add_parents_attention = add_parents_attention
        self.add_parents_embedding = add_parents_embedding
        self.add_siblings_attention = add_siblings_attention
        if self.add_parents_attention or self.add_parents_embedding:
            self.parents = []
        if self.add_siblings_attention:
            self.siblings = []
        print(len(graphs))
        for s, i, g, graph, g_origin in graphs:
            l_return_dict = self.tokenizer.linearize(g, s)
            
            l = l_return_dict['token_uni_ids']
            e = l_return_dict['extra']
            if self.add_parents_attention or self.add_parents_embedding:
                parent = l_return_dict['bpe_total_embeddings']
            if self.add_siblings_attention:
                sibling = l_return_dict['siblings_total_attentions']
#                 l, e, parent = self.tokenizer.linearize(g, s)
#             else:
#                 l, e = self.tokenizer.linearize(g, s)
            
            try:
                self.tokenizer.batch_encode_sentences([s])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(s)
            self.graphs.append(graph)
            self.graph_origins.append(g_origin)
            self.linearized.append(l)
            if self.add_parents_attention or self.add_parents_embedding:
                self.parents.append(parent)
            if self.add_siblings_attention:
                self.siblings.append(sibling)
            self.linearized_extra.append(e)
        print(len(self.graphs))

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.add_parents_attention or self.add_parents_embedding:
            sample['parent'] = self.parents[idx]
        if self.add_siblings_attention:
            sample['sibling'] = self.siblings[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
            if self.add_parents_attention or self.add_parents_embedding:
                z = [s['parent'] for s in samples]
                z = self.tokenizer.batch_encode_parent(z, device=device)
            if self.add_siblings_attention:
                t = [s['sibling'] for s in samples]
                t = self.tokenizer.batch_encode_sibling(t, device=device)
        else:
            y = None
            if self.add_parents_attention or self.add_parents_embedding:
                z = None
        extra['ids'] = [s['id'] for s in samples]
        
        return_dict = {"x": x,
                      "y": y,
                      "extra": extra}
        
        if self.add_parents_attention or self.add_parents_embedding:
            return_dict["z"] = z
        if self.add_siblings_attention:
            return_dict["t"] = t
        
        return return_dict
#             return x, y, z, extra
#         else:
#             return x, y, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
    
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
