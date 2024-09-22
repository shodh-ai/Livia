from megatron.training import get_model
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
import torch.nn as nn
import torch 


def main(input_args=None, overwrite_values=None):
    neox_args = NeoXArgs.consume_neox_args(
        input_args=input_args, overwrite_values=overwrite_values
    )
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    initialize_megatron(neox_args=neox_args)
    model = get_model(neox_args=neox_args,use_cache=False)
    print(model)
    for params in model.parameters():
        params = nn.Parameter(torch.ones_like(params))

    for params in model.parameters():
        print(params.mean())

    inp = torch.ones((2,2048)).cuda()
    out = model(inp)
    print(out.shape)
    print(out)


if __name__ == "__main__":
    main()

    

