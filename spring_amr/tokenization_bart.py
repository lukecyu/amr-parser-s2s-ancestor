import copy
import sys
from pathlib import Path

import penman
import regex as re
import torch
from transformers import BartTokenizer

from spring_amr import ROOT, postprocessing
from spring_amr.linearization import AMRTokens, AMRLinearizer
from spring_amr.penman import encode
from spring_amr.dfs import AMRGraph, convert_amr_dfs



class AMRBartTokenizer(BartTokenizer):

    INIT = 'Ġ'

    ADDITIONAL = [
        AMRTokens.PNTR_N,
        AMRTokens.STOP_N,
        AMRTokens.LIT_START,
        AMRTokens.LIT_END,
        AMRTokens.BACKR_SRC_N,
        AMRTokens.BACKR_TRG_N,]

    def __init__(self, *args, use_pointer_tokens=False, collapse_name_ops=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.linearizer = AMRLinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops)
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary(pred_min=pred_min)
        return inst

    def init_amr_vocabulary(self, pred_min=5):
#         print(len(self.encoder))
#         print(self.encoder[self.INIT + ':'])
#         print(self.encoder[self.INIT + '('])
#         print(self.encoder[self.INIT + ')'])
#         print(self.encoder[self.INIT + '"'])
        rel_tok_set = {self.INIT + ':'}
        left_tok = self.INIT + '('
        right_tok = self.INIT + ')'
        string_tok = {self.INIT + AMRTokens.LIT_START, self.INIT + AMRTokens.LIT_END}        
        
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>']:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i

        tokens = []
        for line in Path(ROOT/'data/vocab/predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if int(count) >= pred_min:
                tokens.append(tok)
                
        for tok in Path(ROOT/'data/vocab/additions.txt').read_text().strip().splitlines():
            tokens.append(tok)
            if tok.startswith(':'):
                rel_tok_set.add(self.INIT + tok)
            

        for tok in Path(ROOT/'data/vocab/recategorizations.txt').read_text().strip().splitlines():
            if not tok.startswith('_'):
                self.recategorizations.add(tok)
            tokens.append(tok)

        if self.use_pointer_tokens:
            for cnt in range(512):
                tokens.append(f"<pointer:{cnt}>")

        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        
        rec_tokens = [self.INIT + t for t in tokens if t[0] == ':']
        
        tokens = [t for t in tokens if t not in self.encoder]
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start= old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)
        
        self.rel_ids = {self.encoder[tok] for tok in rel_tok_set}
        self.left_id = self.encoder[left_tok]
        self.right_id = self.encoder[right_tok]
        self.string_ids = {self.encoder[tok] for tok in string_tok}
        
        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.unk_token = self.INIT + '<unk>'

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                for token in re.findall(self.pat, ' ' + tok_span):
                    token = "".join(
                        self.byte_encoder[b] for b in token.encode("utf-8")
                    )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def _get_nodes_and_backreferences(self, graph):
        lin = self.linearizer.linearize(graph)
        linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences

    def tokenize_amr(self, graph, sentence):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph, sentence)
#         print(linearized_nodes)
#         print()
        if self.add_parents_attention or self.add_parents_embedding:
            total_embs, last_emb = self.construct_parent_embedding(linearized_nodes)
#         print(total_embs)
        
        if self.add_siblings_attention:
            siblings_attention = self.construct_sibling_embedding(linearized_nodes)

        bpe_tokens = []
        bpe_backreferences = []
        counter = 0
        
        if self.add_parents_attention or self.add_parents_embedding:
            parent_embedding_back = []
    
        
        string_mode = 0
        
        wiki_mode = 0
        
#         print(linearized_nodes)
        
        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of  = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d\d?', tokk) is not None
            
            if tokk == ':wiki':
                wiki_mode = 1
            
            # add process of '"'
            if tokk == '"':
                if string_mode == 0:
                    string_mode = 1
                    bpe_toks = [self.INIT + AMRTokens.LIT_START]
                else:
                    string_mode = 0
                    bpe_toks = [self.INIT + AMRTokens.LIT_END]
                if wiki_mode == 1:
                    wiki_mode = 2
                elif wiki_mode == 2:
                    wiki_mode = 0
            
            elif string_mode == 1:
                if wiki_mode == 2:
                    tokk = tokk.replace('_', ' ')
                bpe_toks = self._tok_bpe(tokk, add_space=True)

#             if tokk.startswith('"') and tokk.endswith('"'):
#                 tokk = tokk[1:-1].replace('_', ' ')
#                 bpe_toks = [self.INIT + AMRTokens.LIT_START]
#                 bpe_toks += self._tok_bpe(tokk, add_space=True)
#                 bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3], add_space=True) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:], add_space=True) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:], add_space=True)
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)
            
            if self.add_parents_attention or self.add_parents_embedding:
                parent_embedding_back.append(len(bpe_toks))


            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1  
#         print(parent_embedding_back)

        if self.add_parents_attention or self.add_parents_embedding:
            bpe_total_embeddings = [sum([[1] * parent_embedding_back[j] if e == 1 else [0] * parent_embedding_back[j] for j, e in enumerate(emb)], []) + [1] * i for emb in total_embs[:-1] for i in range(parent_embedding_back[len(emb)])]
#             bpe_total_embeddings = [sum([[1] * parent_embedding_back[j] if e == 1 else [0] * parent_embedding_back[j] for j, e in enumerate(emb[:-1])], []) + [0 for _ in range(i)] + [1] for emb in total_embs for i in range(parent_embedding_back[len(emb)-1])]
    #         print(bpe_total_embeddings)
    #         print()

            bpe_last_emb = sum([[1] * parent_embedding_back[j] if e == 1 else [0] * parent_embedding_back[j] for j, e in enumerate(total_embs[-1])], [])
#         print(bpe_last_emb)
        
        if self.add_siblings_attention:
            siblings_total_attentions = [sum([[1] * parent_embedding_back[j] if e == 1 else [0] * parent_embedding_back[j] for j, e in enumerate(emb)], []) + [1] * i for emb in siblings_attention[:-1] for i in range(parent_embedding_back[len(emb)])]

            siblings_last_attention = sum([[1] * parent_embedding_back[j] if e == 1 else [0] * parent_embedding_back[j] for j, e in enumerate(siblings_attention[-1])], [])
            
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
#         assert len(bpe_total_embeddings) == len(bpe_token_ids)
#         assert len(bpe_total_embeddings[-1]) == len(bpe_token_ids)
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]
    
        return_dict = {'bpe_tokens': bpe_tokens,
                      'bpe_token_ids': bpe_token_ids,
                      'bpe_backreferences': bpe_backreferences,
                      }
        
        if self.add_parents_attention or self.add_parents_embedding:
            return_dict['bpe_total_embeddings'] = bpe_total_embeddings
            return_dict['bpe_last_emb'] = bpe_last_emb
            
        if self.add_siblings_attention:
            return_dict['siblings_total_attentions'] = siblings_total_attentions
            return_dict['siblings_last_attention'] = siblings_last_attention
        
        return return_dict
        
#         if self.add_parents_attention or self.add_parents_embedding:
#             return bpe_tokens, bpe_token_ids, bpe_backreferences, bpe_total_embeddings, bpe_last_emb
#         else:
#             return bpe_tokens, bpe_token_ids, bpe_backreferences
            
    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = super().batch_encode_plus(sentences, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra
    
    def linearize(self, graph, sentence):
        shift = len(self.encoder)
        
        return_dict = self.tokenize_amr(graph, sentence)
        tokens = return_dict['bpe_tokens']
        token_ids = return_dict['bpe_token_ids']
        backreferences = return_dict['bpe_backreferences']
        
        if self.add_parents_attention or self.add_parents_embedding:
            bpe_total_embeddings = return_dict['bpe_total_embeddings']
            bpe_last_emb = return_dict['bpe_last_emb']
        if self.add_siblings_attention:
            siblings_total_attentions = return_dict['siblings_total_attentions']
            siblings_last_attention = return_dict['siblings_last_attention']
#             tokens, token_ids, backreferences, bpe_total_embeddings, bpe_last_emb = self.tokenize_amr(graph, sentence)
#         else:
#             tokens, token_ids, backreferences = self.tokenize_amr(graph, sentence)
        extra = {'linearized_graphs': tokens, 'graphs': graph}
        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (self.INIT + AMRTokens.EOS_N):
            tokens.append(self.INIT + AMRTokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))
            if self.add_parents_attention or self.add_parents_embedding:
                bpe_total_embeddings.append(bpe_last_emb)
            if self.add_siblings_attention:
                siblings_total_attentions.append(siblings_last_attention)
    
        l_return_dict = {'token_uni_ids': token_uni_ids,
                        'extra': extra}
        
        if self.add_parents_attention or self.add_parents_embedding:
            l_return_dict['bpe_total_embeddings'] = bpe_total_embeddings
        if self.add_siblings_attention:
            l_return_dict['siblings_total_attentions'] = siblings_total_attentions
        
        return l_return_dict
#             return token_uni_ids, extra, bpe_total_embeddings
#         else:
#             return token_uni_ids, extra
        
    def batch_encode_graphs(self, graphs, device=torch.device('cpu')):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)
    
    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_graphs': [], 'graphs': []}
            for extra in extras:
                batch_extra['graphs'].append(extra['graphs'])
                batch_extra['linearized_graphs'].append(extra['linearized_graphs'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'lm_labels': batch[:, 1:]}
        return batch, batch_extra
    
    def batch_encode_parent(self, parents, device=torch.device('cpu')):
        batch = []
        for parent in parents:
            batch.append([e + [0] * (1024 - len(e)) for e in parent])
        maxlen = 0
        for p in batch:
            maxlen = max(len(p), maxlen)
        batch = [parent + [[0] * 1024 for _ in range(maxlen - len(parent))] for parent in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'parents': batch}
        
        return batch
    
    def batch_encode_sibling(self, siblings, device=torch.device('cpu')):
        batch = []
        for sibling in siblings:
            batch.append([e + [0] * (1024 - len(e)) for e in sibling])
        maxlen = 0
        for p in batch:
            maxlen = max(len(p), maxlen)
        batch = [sibling + [[0] * 1024 for _ in range(maxlen - len(sibling))] for sibling in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'siblings': batch}
        
        return batch

    def decode_amr(self, tokens, restore_name_ops=False):
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        if self.use_pointer_tokens:
            nodes, backreferences = postprocessing.restore_backreferences_from_pointers(nodes)
        try:
            graph_ = graph = postprocessing.build_graph(nodes, backreferences, restore_name_ops=restore_name_ops)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)

class PENMANBartTokenizer(AMRBartTokenizer):

    def __init__(self, *args, raw_graph=False, add_parents_attention=False, add_parents_embedding=False, add_siblings_attention=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearizer = None
        self.remove_pars = False
        self.raw_graph = raw_graph
        self.add_parents_attention=add_parents_attention
        self.add_parents_embedding=add_parents_embedding
        self.add_siblings_attention=add_siblings_attention

    def _tokenize_encoded_graph(self, encoded):
        linearized = re.sub(r"(\".+?\")", r' \1 ', encoded)
        pieces = []
        for piece in linearized.split():
#             if piece.startswith('"') and piece.endswith('"'):
#                 pieces.append(piece)
#             else:
            piece = piece.replace('"', ' " ')
            piece = piece.replace('(', ' ( ')
            piece = piece.replace(')', ' ) ')
            piece = piece.replace(':', ' :')
            piece = piece.replace('/', ' / ')
            piece = piece.strip()
            pieces.append(piece)
        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()
        linearized_nodes = [AMRTokens.BOS_N] + linearized.split(' ')
        return linearized_nodes

    def tokenize_amr(self, graph, sentence):
        if self.raw_graph:
            graph_ = copy.deepcopy(graph)
            graph_.metadata = {}
            linearized = penman.encode(graph_)
            linearized = re.sub(r"\s+", ' ', linearized)
            bpe_tokens = [self.bos_token] + self._tokenize(linearized)[:1022]
            bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
            bpe_backreferences = list(range(len(bpe_token_ids)))
            return bpe_tokens, bpe_token_ids, bpe_backreferences
        else:
            return super().tokenize_amr(graph, sentence)
        
    def construct_sibling_embedding(self, linearized):
#         print(linearized)
        total_embs = [[1]]
        emb = [1]
        emb_emerge = [1]
        
        emb_real = [0]
        
#         stackposition = [0]
#         stacks = [0]
        
        string_mode = 0
        
        status_stack = []
        status_p = [0]
        
        real_stack = []
        
        for i in range(1, len(linearized)):
            if string_mode == 0 and linearized[i] == '"':
                string_mode = 1
                emb_emerge.append(1)
                
                emb_real.append(0)
#                 stacks.append(i)
            elif string_mode == 1 and linearized[i] == '"':
                string_mode = 0
                emb_emerge.append(1)
                
                emb_real.append(0)
#                 stacks.append(i)
            elif string_mode == 0:
                if linearized[i] == '(':
                    status_stack.append(0)
                    emb_emerge.append(1)
                    
                    emb_real.append(0)
#                     stacks.append(i)
                elif linearized[i].startswith(':'):
                    if len(status_stack) == 0:
                        emb_emerge.append(1)
                    
                        emb_real.append(0)
#                         stacks.append(i)
                    elif status_stack[-1] == 0:
                        status_stack[-1] = 1
                        
                        emb = emb[:status_p[-1]] + emb_emerge[status_p[-1]:]
                        emb_emerge = [0] * len(emb)
                        
                        emb_real = [0] * len(emb)
                                                
                        status_p.append(i)
                        emb_emerge.append(1)
                        
                        emb_real.append(0)
                        
                        real_stack.append(i)
#                         stacks.append(i)
                    else:
                        emb_real = emb_real[:real_stack[-1]] + emb_emerge[real_stack[-1]:]
                        real_stack.pop()
                        real_stack.append(i)
            
                        emb_emerge.append(1)
            
                        emb_real.append(0)
#                         stacks.append(i)
                elif linearized[i] == ')':
                    if len(status_stack) == 0:
                        emb_emerge.append(1)
                    
                        emb_real.append(0)

#                         stacks.append(i)
                    elif status_stack[-1] == 0:
                        status_stack.pop()
                        emb_emerge.append(1)
                        emb_real.append(0)
#                         stacks.append(i)
                    else:
                        status_stack.pop()
#                         emb_emerge.append(1)
                        emb = emb[:status_p[-1]] + [0] * (len(emb_emerge) - status_p[-1])
                        status_p.pop()
                        emb_emerge = [0] * status_p[-1] + emb[status_p[-1]:] + [1]
            
                        real_stack.pop()
                        if len(real_stack) == 0:
                            emb_real = emb_emerge[:]
                        else:
                            emb_real = [0] * len(emb_emerge)
                        
#                         while stack_p[-1] != 0:
#                             stackposition.pop()
#                             stack_p.pop()
#                         emb.append(1)
#                         stacks.append(i)
#     #                     print(len(emb))
#                         while stacks[-1] != stackposition[-1]:
#                             emb[stacks[-1]] = 0
#                             stacks.pop()
#                         emb[stacks[-1]] = 0
#                         stacks.pop()
#                         stackposition.pop()
#                         stack_p.pop()
                else:
                    emb_emerge.append(1)
                    emb_real.append(0)
#                         stacks.append(i)
            else:
                emb_emerge.append(1)
                emb_real.append(0)
            
            total_embs.append(emb_real[:])
#         print(total_embs)
        
        return total_embs
                
        
        
        
    def construct_parent_embedding(self, linearized):
#         print(linearized)
        total_embs = [[1]]
        emb = [1]
        stackposition = [0]
        stacks = [0]
        
        string_mode = 0
        previous = 0
        
        start = 0
        
        # -1: start 0: ( 1: relation
        stack_p = [-1]
        
#         print(linearized)
        
        for i in range(1, len(linearized)):
#             total_embs.append(emb[:])
#             print(stackposition)
#             print(stack_p)
#             print(stacks)
            
#             print(i)
#             print(len(emb))
#             print()
            if string_mode == 0 and linearized[i] == '"':
                string_mode = 1
                emb.append(1)
                stacks.append(i)
            elif string_mode == 1 and linearized[i] == '"':
                string_mode = 0
                emb.append(1)
                stacks.append(i)
            elif string_mode == 0:
                if linearized[i] == '(':
                    stackposition.append(i)
                    stack_p.append(0)
                    emb.append(1)
                    stacks.append(i)
                elif linearized[i].startswith(':'):
                    if stack_p[-1] == 0:
                        stackposition.append(i)
                        stack_p.append(1)
                        emb.append(1)
                        stacks.append(i)
                    else:
                        while stacks[-1] != stackposition[-1]:
                            emb[stacks[-1]] = 0
                            stacks.pop()
                        emb[stacks[-1]] = 0
                        stacks.pop()
                        stackposition.pop()
                        stack_p.pop()
                        
                        stackposition.append(i)
                        stack_p.append(1)
                        emb.append(1)
                        stacks.append(i)
                elif linearized[i] == ')':
                    while stack_p[-1] != 0:
                        stackposition.pop()
                        stack_p.pop()
                    emb.append(1)
                    stacks.append(i)
#                     print(len(emb))
                    while stacks[-1] != stackposition[-1]:
                        emb[stacks[-1]] = 0
                        stacks.pop()
                    emb[stacks[-1]] = 0
                    stacks.pop()
                    stackposition.pop()
                    stack_p.pop()
#                 if (linearized[previous].startswith(':') and len(linearized) > previous+1 and linearized[previous+1] != '(') linearized[i] == ')':
#                     while stacks[-1] != stackposition[-1]:
#                         emb[stacks[-1]] = 0
#                         stacks.pop()
#                     if stacks[-1] != 0:
#                         emb[stacks[-1]] = 0
#                         stacks.pop()
#                         stackposition.pop()
                else:
                    emb.append(1)
                    stacks.append(i)
                previous = i
            else:
                emb.append(1)
                stacks.append(i)
            total_embs.append(emb[:])
            
        emb.append(1)
        
        return total_embs, emb
                            

    def _get_nodes_and_backreferences(self, graph, sentence):
#         graph_ = copy.deepcopy(graph)
#         graph_.metadata = {}
#         linearized = penman.encode(graph_)
        linearized = graph
#         print(linearized)
        linearized_nodes = self._tokenize_encoded_graph(linearized)
#         print(linearized_nodes)
        
        string_mode = 0

        if self.use_pointer_tokens:
            remap = {}
            remap_word = {}
            remap_visit = {}
            for i in range(1, len(linearized_nodes)):
                if linearized_nodes[i] == '"':
                    string_mode = 1 - string_mode
                if string_mode == 0:
                    
                    nxt = linearized_nodes[i]
                    lst = linearized_nodes[i-1]
                    if nxt == '/':
                        remap[lst] = f'<pointer:{len(remap)}>'
                        remap_word[lst] = linearized_nodes[i+1]
                        remap_visit[lst] = 0
            i = 1
            linearized_nodes_ = [linearized_nodes[0]]
            while i < (len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes_[-1]
                if nxt in remap:
                    if lst == '(' and linearized_nodes[i+1] == '/':
                        nxt1 = remap[nxt]
                        i += 2
                    elif lst.startswith(':'):
                        nxt1 = remap[nxt]
                        if remap_visit[nxt] == 0:
                            print(sentence)
#                             print(linearized)
                            print(nxt)
                    linearized_nodes_.append(nxt1)
                    if remap_visit[nxt] == 0:
                        linearized_nodes_.append(remap_word[nxt])
                        remap_visit[nxt] = 1
                else:
                    linearized_nodes_.append(nxt)
                i += 1
            linearized_nodes = linearized_nodes_
            if self.remove_pars:
                linearized_nodes = [n for n in linearized_nodes if n != '(']
        backreferences = list(range(len(linearized_nodes)))
        return linearized_nodes, backreferences

    def _classify(self, node):
        if not isinstance(node, str):
            return "CONST"
        elif node == 'i':
            return "I"
        elif re.match(r'^[a-z]\d*$', node) is not None:
            return "VAR"
        elif node[0].isdigit():
            return "CONST"
        elif node.startswith('"') and node.endswith('"'):
            return "CONST"
        elif node in ('+', '-'):
            return "CONST"
        elif node == ':mode':
            return 'MODE'
        elif node.startswith(':'):
            return "EDGE"
        elif node in ['/', '(', ')']:
            return node
        elif node[0].isalpha():
            for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\'):
                if char in node:
                    return "CONST"
            return "INST"
        else:
            return 'CONST'

    def _fix_and_make_graph(self, nodes):

        nodes_ = []
        for n in nodes:
            if isinstance(n, str):
                if n.startswith('<') and n.endswith('>') and (not n.startswith('<pointer:')):
                    pass
                else:
                    nodes_.append(n)
            else:
                nodes_.append(n)
        nodes = nodes_

        if self.use_pointer_tokens:

            i = 0
            nodes_ = []
            while i < len(nodes):
                nxt = nodes[i]
                pst = None
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    e = nxt.find('>')
                    if e != len(nxt) -1:
                        pst = nxt[e+1:]
                        nxt = nxt[:e+1]
                    nodes_.append(nxt)
                    if pst is not None:
                        nodes_.append(pst)
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

            i = 1
            nodes_ = [nodes[0]]
            while i < len(nodes):
                nxt = nodes[i]
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    nxt = 'z' + nxt[9:-1]
                    fol = nodes[i+1]
                    # is not expansion
                    if isinstance(fol, str) and (fol.startswith(':') or (fol == ')')):
                        nodes_.append(nxt)
                    else:
                        if self.remove_pars:
                            nodes_.append('(')
                        else:
                            if nodes_[-1] != '(':
                                nodes_.append('(')
                                #pass
                        nodes_.append(nxt)
                        nodes_.append('/')
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes) - 1):
            if nodes[i] == ':':
                nodes_.append(nodes[i] + nodes[i+1])
                i += 2
                last = False
            else:
                nodes_.append(nodes[i])
                i += 1
                last = True
        if last:
            nodes_.append(nodes[-1])
        nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes)):
            if i < 2:
                nodes_.append(nodes[i])
                i += 1
            elif nodes_[-2] == '/' and nodes[i] == '/':
                i += 2
            else:
                nodes_.append(nodes[i])
                i += 1
        nodes = nodes_

        i = 0
        newvars = 0
        variables = set()
        remap = {}
        nodes_ = []
        while i < (len(nodes)):

            next = nodes[i]

            if next == '/':
                last = nodes_[-1]
                if last in variables:
                    last_remap = f"z{newvars+1000}"
                    newvars += 1
                    nodes_[-1] = last_remap
                    remap[last] = last_remap
                variables.add(last)
                nodes_.append(next)

            elif self._classify(next) == 'VAR' and next in remap and (i < len(nodes) - 1) and nodes[i+1] != '/':
                next = remap[next]
                nodes_.append(next)

            else:
                nodes_.append(next)

            i += 1

        nodes = nodes_
        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if nodes[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in nodes:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        nodes = pieces_ + [')'] * (open_cnt - closed_cnt)

        pieces = []
        for piece in nodes:
            if not pieces:
                pieces.append('(')
            else:
                piece = str(piece)
                if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                    piece = '"' + piece.replace('"', '') + '"'

                prev = self._classify(pieces[-1])
                next = self._classify(piece)

                if next == 'CONST':
                    quote = False
                    for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\', '_', '='):
                        if char in piece:
                            quote = True
                            break
                    if quote:
                        piece = '"' + piece.strip('"') + '"'

                if  prev == '(':
                    if next in ('VAR', 'I'):
                        pieces.append(piece)
                elif prev == ')':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'VAR':
                    if next in ('/', 'EDGE', 'MODE', ')'):
                        pieces.append(piece)
                elif prev == '/':
                    if next in ('INST', 'I'):
                        pieces.append(piece)
                elif prev == 'INST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'I':
                    if next in ('/', ')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'EDGE':
                    if next in ('(', 'VAR', 'CONST', 'I'):
                        pieces.append(piece)
                    elif next == ')':
                        pieces[-1] = piece
                    elif next in ('EDGE', 'MODE'):
                        pieces[-1] = piece
                elif prev == 'MODE':
                    if next == 'INST':
                        pieces.append(piece)
                elif prev == 'CONST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)

        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if pieces[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in pieces:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        pieces = pieces_ + [')'] * (open_cnt - closed_cnt)

        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()

        """
        line = linearized
        # make sure parentheses match
        # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
        open_count = 0
        close_count = 0
        for i, c in enumerate(line):
            if c == '(':
                open_count += 1
            elif c == ')':
                close_count += 1
            if open_count == close_count and open_count > 0:
                line = line[:i].strip()
                break
        old_line = line
        while True:
            open_count = len(re.findall(r'\(', line))
            close_count = len(re.findall(r'\)', line))
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                for i in range(close_count - open_count):
                    line = line.rstrip(')')
                    line = line.rstrip(' ')
            if old_line == line:
                break
            old_line = line
        """

        graph = penman.decode(linearized + ' ')
        triples = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if x is None:
                pass
            elif rel == ':instance' and y is None:
                triples.append(penman.Triple(x, rel, 'thing'))
            elif y is None:
                var = f'z{newvars}'
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ':instance', 'thing'))
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
        linearized = encode(graph)

        def fix_text(linearized=linearized):
            n = 0
            def _repl1(match):
                nonlocal n
                out = match.group(1) + match.group(2) + str(3000 + n) + ' / ' + match.group(2) + match.group(3)
                n += 1
                return out
            linearized = re.sub(r'(\(\s?)([a-z])([^\/:\)]+[:\)])', _repl1, linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            def _repl2(match):
                return match.group(1)
            linearized = re.sub(r'(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)', _repl2,
                                linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            # adds a ':' to args w/o it
            linearized = re.sub(r'([^:])(ARG)', r'\1 :\2', linearized)

            # removes edges with no node
            # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

            return linearized

        linearized = fix_text(linearized)

        g = penman.decode(linearized)
        return g

    def decode_amr(self, tokens, restore_name_ops=None):
        try:
            if self.raw_graph:
#                 print(self.decode(tokens))
                nodes = self._tokenize_encoded_graph(self.decode(tokens))
                backreferences = list(range(len(nodes)))
            else:
                nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
            nodes_ = nodes
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph_ = graph = self._fix_and_make_graph(nodes)
            if self.collapse_name_ops:
                graph_ = graph = postprocessing._split_name_ops(graph)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes_, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes_, backreferences)