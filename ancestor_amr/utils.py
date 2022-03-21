from glob import glob
from pathlib import Path

import torch
from transformers import AutoConfig

from ancestor_amr.dataset import AMRDataset, AMRDatasetTokenBatcherAndLoader
from ancestor_amr.modeling_bart import AMRBartForConditionalGeneration
from ancestor_amr.tokenization_bart import AMRBartTokenizer, PENMANBartTokenizer


def instantiate_model_and_tokenizer(
        name=None,
        checkpoint=None,
        additional_tokens_smart_init=True,
        dropout = 0.15,
        attention_dropout = 0.15,
        from_pretrained = True,
        init_reverse = False,
        collapse_name_ops = False,
        penman_linearization = False,
        use_pointer_tokens = False,
        raw_graph = False,
        add_parents_attention = False,
        add_parents_embedding = False,
        add_siblings_attention = False,
        tune_attention = False,
        parents_attention_number = 4,
        siblings_attention_number = 4,
        attention_form = 'add',
        layer_parents = False,
        layer_parents_ids = [],
):
    if raw_graph:
        assert penman_linearization

    skip_relations = False

    if name is None:
        name = 'facebook/bart-large'

    if name == 'facebook/bart-base':
        tokenizer_name = 'facebook/bart-large'
    else:
        tokenizer_name = name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout
    config.add_parents_attention = add_parents_attention
    config.add_parents_embedding = add_parents_embedding
    config.add_siblings_attention = add_siblings_attention
    config.parents_attention_number = parents_attention_number
    config.siblings_attention_number = siblings_attention_number
    config.tune_attention = tune_attention
    config.attention_form = attention_form
    config.layer_parents = layer_parents
    config.layer_parents_ids = layer_parents_ids

    if penman_linearization:
        tokenizer = PENMANBartTokenizer.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            raw_graph=raw_graph,
            config=config,
            add_parents_attention=add_parents_attention,
            add_parents_embedding=add_parents_embedding,
            add_siblings_attention=add_siblings_attention
        )
    else:
        tokenizer = AMRBartTokenizer.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            config=config,
            add_parents_attention=add_parents_attention,
            add_parents_embedding=add_parents_embedding,
            add_siblings_attention=add_siblings_attention,
        )

#     config.colon_id = tokenizer.encode(tokenizer.INIT + ':')
#     config.quote_id = tokenizer.encode(tokenizer.INIT + '"')
#     config.paren_id = tokenizer.encode(tokenizer.INIT + ')')
        
    if from_pretrained:
        model = AMRBartForConditionalGeneration.from_pretrained(name, config=config)
    else:
        model = AMRBartForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer.encoder))

    if additional_tokens_smart_init:
        modified = 0
        for tok, idx in tokenizer.encoder.items():
            tok = tok.lstrip(tokenizer.INIT)

            if idx < tokenizer.old_enc_size:
                continue

            elif tok.startswith('<pointer:') and tok.endswith('>'):
                tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

            elif tok.startswith('<'):
                continue

            elif tok.startswith(':'):

                if skip_relations:
                    continue

                elif tok.startswith(':op'):
                    tok_split = ['relation', 'operator', str(int(tok[3:]))]

                elif tok.startswith(':snt'):
                    tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                elif tok.startswith(':ARG'):
                    tok_split = ['relation', 'argument', str(int(tok[4:]))]

                else:
                    tok_split = ['relation'] + tok.lstrip(':').split('-')

            else:
                tok_split = tok.split('-')

            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + tokenizer.INIT
                if s_ in tokenizer.encoder:
                    tok_split.append(s_)
                else:
                    tok_split.extend(tokenizer._tok_bpe(s))

            vecs = []
            for s in tok_split:
                idx_split = tokenizer.encoder.get(s, -1)
                if idx_split > -1:
                    vec_split = model.model.shared.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                model.model.shared.weight.data[idx] = vec + noise
                modified += 1

    if init_reverse:
        model.init_reverse_model()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    return model, tokenizer


def instantiate_loader(
        glob_pattn,
        tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=1024,
        remove_wiki=False,
        dereify=True,
        add_parents_attention=False,
        add_parents_embedding=False,
        add_siblings_attention=False,
):
    paths = []
    print("glob_pattn", glob_pattn)
    if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
        glob_pattn = [glob_pattn]
    for gpattn in glob_pattn:
#         print(gpattn)
        paths += [Path(p) for p in glob(gpattn)]
    if evaluation:
        assert out is not None
        Path(out).write_text(
            '\n\n'.join([p.read_text() for p in paths]))
    print("paths", paths)
    dataset = AMRDataset(
        paths,
        tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        add_parents_attention=add_parents_attention,
        add_parents_embedding=add_parents_embedding,
        add_siblings_attention=add_siblings_attention,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader
