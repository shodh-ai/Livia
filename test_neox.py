from megatron.utils import setup_for_inference_or_eval
import torch.nn as nn
import torch 
from megatron.utils import get_ltor_masks_and_position_ids

def main(input_args=None, overwrite_values=None):
    model, neox_args = setup_for_inference_or_eval(
        use_cache=False, input_args=input_args, overwrite_values=overwrite_values
    )
    print(model)

    for name, params in model.named_parameters():
        if 'weight' in name:
            params.data = torch.ones_like(params)
        elif 'bias' in name:
            params.data = torch.full_like(params,1)


    # for name, params in model.named_parameters():
    #     if 'weight' in name:
    #         params.data = torch.ones_like(params)
    #     elif 'bias' in name:
    #         params.data = torch.zeros_like(params)

    # for params in model.parameters():
    #     params.data = torch.ones_like(params)

    for name, params in model.named_parameters():
        print(f"Parameter: {name}, Mean: {params.mean().item()}")

    tokens = torch.ones((1,5), dtype=torch.long).cuda()  
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )
   
    logits = model((tokens, position_ids, attention_mask))[0]

    print(logits.shape)
    print(logits)

if __name__ == "__main__":
    main()