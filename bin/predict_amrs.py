from pathlib import Path

import penman
import torch

from ancestor_amr import ROOT
from ancestor_amr.evaluation import predict_amrs, compute_smatch
from ancestor_amr.penman import encode
from ancestor_amr.utils import instantiate_loader, instantiate_model_and_tokenizer

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
        help="Required. One or more glob patterns to use to load amr files.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=Path, default=ROOT / 'data/tmp/inf-pred.txt',
        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=Path, default=ROOT / 'data/tmp/inf-gold.txt',
        help="Where to write the gold file.")
    parser.add_argument('--use-recategorization', action='store_true',
        help="Predict using Zhang recategorization on top of our linearization (requires recategorized sentences in input).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--add_parents_attention', action='store_true')
    parser.add_argument('--add_parents_embedding', action='store_true')
    parser.add_argument('--add_siblings_attention', action='store_true')
    parser.add_argument('--siblings_attention_number', type=int, default=2)
    parser.add_argument('--tune_attention', action='store_true')
    parser.add_argument('--parents_attention_number', type=int, default=2)
    parser.add_argument('--raw-graph', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--return-all', action='store_true')
    parser.add_argument('--attention_form', type=str, default='add')
    parser.add_argument('--layer_parents', action='store_true')
    parser.add_argument('--layer_parents_ids', nargs='+', type=int)
    parser.add_argument('--max_length', type=int, default=512)

    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=args.raw_graph,
        add_parents_attention=args.add_parents_attention,
        add_parents_embedding=args.add_parents_embedding,
        tune_attention=args.tune_attention,
        parents_attention_number=args.parents_attention_number,
        add_siblings_attention=args.add_siblings_attention,
        siblings_attention_number=args.siblings_attention_number,
        attention_form=args.attention_form,
        layer_parents=args.layer_parents,
        layer_parents_ids=args.layer_parents_ids,
    )
    model.amr_mode = True
    print(model)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)

    gold_path = args.gold_path
    pred_path = args.pred_path
    print("args datasets", args.datasets)
    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True, out=gold_path,
        use_recategorization=args.use_recategorization,
        add_parents_attention=args.add_parents_attention,
        add_parents_embedding=args.add_parents_embedding,
        add_siblings_attention=args.add_siblings_attention,
    )
    loader.device = device

    graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=args.beam_size,
        restore_name_ops=args.restore_name_ops,
        return_all=args.return_all,
        add_parents_attention=args.add_parents_attention,
        add_parents_embedding=args.add_parents_embedding,
        add_siblings_attention=args.add_siblings_attention,
        max_length=args.max_length,
    )
    if args.return_all:
        graphs = [g for gg in graphs for g in gg]

    pieces = [encode(g) for g in graphs]
    pred_path.write_text('\n\n'.join(pieces))

    if not args.return_all:
        score = compute_smatch(gold_path, pred_path)
        print(f'Smatch: {score:.3f}')
