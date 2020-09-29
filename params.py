import argparse


def add_args(parser):
    # Optimization arguments
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--save')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model_debug', action='store_true')
    parser.add_argument('--mode', required=True)  # train, test, resume
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--other_lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=str, default='')
    # parser.add_argument('--oie_model', type=str, default='models/oie_model/epoch=19_eval_acc=0.548.ckpt')
    # parser.add_argument('--conj_model', type=str, default='models/conj_model/epoch=16_eval_acc=0.890.ckpt')
    parser.add_argument('--oie_model', type=str)
    parser.add_argument('--conj_model', type=str)
    parser.add_argument('--val_interval', type=float, default=1.0)
    parser.add_argument('--save_k', type=int, default=1)
    parser.add_argument('--use_tpu', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adamW')

    # Data arguments
    parser.add_argument('--task', type=str)
    parser.add_argument('--backend', type=str)
    parser.add_argument('--train_fp', type=str)
    parser.add_argument('--dev_fp', type=str)
    parser.add_argument('--test_fp', type=str)
    parser.add_argument('--predict_fp', type=str)
    parser.add_argument('--split_fp', type=str, default='')
    parser.add_argument('--predict_out_fp', type=str, default='predictions')
    parser.add_argument('--out_ext', type=str)
    parser.add_argument('--predict_format', type=str,
                        default='oie')  # oie/allennlp
    parser.add_argument('--build_cache', action='store_true')

    # Model arguments
    # bert-large-cased-whole-word-masking, bert-large-cased, bert-base-cased
    parser.add_argument('--model_str', default='bert-base-cased')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--optim_adam', action='store_true')
    parser.add_argument('--optim_lstm', action='store_true')
    parser.add_argument('--optim_adam_lstm', action='store_true')
    parser.add_argument('--iterative_layers', type=int, default=2)
    parser.add_argument('--labelling_dim', type=int, default=300)
    parser.add_argument('--num_extractions', type=int)
    parser.add_argument('--keep_all_predictions', action='store_true')
    parser.add_argument('--oie_split', action='store_true')
    parser.add_argument('--no_lt', action='store_true')
    parser.add_argument('--rescoring', action='store_true')
    parser.add_argument('--rescoring_topk', type=int)
    parser.add_argument('--rescore_model', type=str, default='models/rescore_model')
    parser.add_argument('--write_allennlp', action='store_true')
    parser.add_argument('--write_async', action='store_true')

    # constraints
    parser.add_argument('--wreg', type=float, default=0)
    parser.add_argument('--constraints', type=str, default='')
    parser.add_argument('--cweights', type=str, default='1')
    parser.add_argument('--multi_opt', action='store_true')

    # additional options
    parser.add_argument('--output_labels', type=str)
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--type', type=str, default='')

    return parser
