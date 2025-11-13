import json
from tqdm import tqdm
import torch
import os

from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model


def generate(prompt_data, steer_values, tokenizer, model,
             prompt_num, prompt_length, num_beams, num_beam_groups,
             do_sample, temperature, top_p, device):
    # for _prompt in tqdm(prompt_data): # original
    for _prompt in tqdm(prompt_data[:100]): # edited @kat debug
        _prompt["generations"] = []
        if os.environ.get("PROMPT_COL", None) is not None:
            prompt_col = os.environ["PROMPT_COL"]
            prompt_text = _prompt[prompt_col]
        else:
            prompt_text = _prompt["prompt"]["text"]
        
        if isinstance(prompt_text, str):
            token_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        else:
            print("Applying chat template...")
            token_length = tokenizer.apply_chat_template(prompt_text, return_tensors="pt").shape[1]
            prompt_text = tokenizer.apply_chat_template(prompt_text, tokenize=False, add_generation_prompt=True)
            
        for _i in range(prompt_num):
            output = model.generate(
                prompt_text,
                steer_values,
                seed=_i,
                max_length=token_length+prompt_length,
                min_length=token_length+prompt_length,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            output = output[len(prompt_text):]
            _prompt["generations"].append({
                "text": output
            })
        if args.verbose:
            print(prompt_text)
            print(_prompt["generations"])


def main(args):
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.adaptor_class,
        args.num_steers,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.to_device(device)
    
    ckpt = torch.load(args.ckpt_name, weights_only=False)
    model.load_state_dict(ckpt[1])
        
    # predicting sentences
    with open(args.eval_file, "r") as f:
        prompt_data = list(map(json.loads, f.readlines()))

    model.eval()
    prompt_num = int(os.environ.get("PROMPT_NUM", 25))
    prompt_length = int(os.environ.get("PROMPT_NUM_TOKENS", 20))
    print(f"Generating: {prompt_num} sequences for each prompt")
    print(f"Generating up to {prompt_length} tokens for each prompt.")
    if args.eval_size is not None:
        prompt_data = prompt_data[:args.eval_size]
    num_beams = 1
    num_beam_groups = 1
    do_sample = True
    temperature = args.temperature
    steer_values = list(map(float, args.steer_values)) \
        if args.steer_values is not None else None

    generate(prompt_data, steer_values, tokenizer, model, prompt_num,
             prompt_length, num_beams, num_beam_groups, do_sample, temperature,
             args.top_p, device)

    with open(args.output_file, "w") as f:
        for _prompt in prompt_data:
            f.write(json.dumps(_prompt) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
