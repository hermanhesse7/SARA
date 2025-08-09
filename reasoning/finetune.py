import os
import sys
import torch
import csv
import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
)
import argparse

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer, LoraConfig
from utils import (
    load_data,
    format_dataset,
    DataCollatorForCausalLM,
    generate_alpaca,
    get_subset,
    get_parameters_count,
    generate_samples,
    preprocess_logits_for_metrics,
    compute_metrics,
    find_all_linear_names,
)

torch.backends.cuda.matmul.allow_tf32 = True

def log_metrics_to_csv(metrics, filename, fieldnames=None):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames or list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)



def validate_args(args):
    """Validate argument combinations"""
    if args.custom_mode == "sara":
        if args.lora_c <= 0:
            raise ValueError("SARA mode requires lora_c > 0")
        if args.lora_r % args.lora_c != 0:
            raise ValueError(f"For SARA mode, lora_r ({args.lora_r}) must be divisible by lora_c ({args.lora_c})")
    
    if args.shared_uv == 1 and args.shared_dim is None:
        raise ValueError("shared_uv=1 requires shared_dim to be specified")
    
    if args.custom_mode == "full" and any([
        args.lora_r != 16,  # Default values suggest LoRA usage
        args.lora_alpha != 1,
        args.target_modules != "lm_head"
    ]):
        print("Warning: LoRA parameters specified but custom_mode is 'full'")

# Add validation call        

def run(args):
    validate_args(args)

    job_id = os.environ.get("SLURM_JOB_ID", "0")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(args.seed)

    # Log configuration to CSV
    config_file = os.path.join(output_dir, "config.csv")
    config_data = {"job_id": job_id, "timestamp": timestamp, **vars(args)}
    log_metrics_to_csv(config_data, config_file)

    print(f"Job ID: {job_id}")
    print(f"Output directory: {output_dir}")

    #imdb = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="right", use_fast=False, legacy=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    # Tokenizer asserts remain unchanged
    assert (
        tokenizer("### Response:", add_special_tokens=False)["input_ids"]
        + tokenizer(
            f"{'' if isinstance(tokenizer, LlamaTokenizer) else ' '}Negative",
            add_special_tokens=False,
        )["input_ids"]
        == tokenizer("### Response: Negative", add_special_tokens=False)["input_ids"]
    )
    assert (
        tokenizer(f"{tokenizer.bos_token}test", add_special_tokens=False)["input_ids"]
        == [tokenizer.bos_token_id]
        + tokenizer("test", add_special_tokens=False)["input_ids"]
    )

    # Dataset functions remain unchanged
    def _imdb_to_alpaca(examples, _instruction, answers, cut_off=1000):
        instruction = []
        input = []
        output = []
        for i in range(len(examples["text"])):
            instruction.append(_instruction)
            input.append(f'"{examples["text"][i][:cut_off]}"')
            output.append(answers[0] if examples["label"][i] == 0 else answers[1])
        return {"output": output, "instruction": instruction, "input": input}

    def imdb_to_alpaca_easy(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", but without quotes.',
            ["Review is negative.", "Review is positive."],
        )

    def imdb_to_alpaca_quotes(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", with quotes.',
            ['"Review is negative."', '"Review is positive."'],
        )

    def imdb_to_alpaca_brackets(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", but without quotes and put your answer in square brackets.',
            ["[Review is negative.]", "[Review is positive.]"],
        )

#    eval_ds = {}
#    ds_names = ["easy", "quotes", "brackets"]
#    for name in ds_names:
#        eval_ds[name] = (
#            imdb["test"]
#            if (args.eval_samples is None or args.eval_samples == 0)
#            else get_subset(imdb["test"], args.eval_samples)
#        )
#        eval_ds[name] = eval_ds[name].map(
#            eval(f"imdb_to_alpaca_{name}"),
#            batched=True,
#            remove_columns=imdb["train"].column_names,
#        )
#        eval_ds[name] = format_dataset(eval_ds[name], "alpaca-clean")

    if args.task == "instruct":
        dataset = load_data(args.dataset)
        dataset = format_dataset(dataset, args.dataset_format)
        train_ds = (
            dataset["train"]
            if (args.train_samples is None or args.train_samples == 0)
            else dataset["train"].select(range(args.train_samples))
        )

    elif args.task == "math":
        dataset = load_data(args.dataset)
        dataset = format_dataset(dataset, args.dataset_format)
        val_set_size = 120
        train_val = dataset["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_ds = train_val["train"].shuffle()
        eval_ds =  train_val["test"].shuffle()

    elif args.task == "imdb":
        name = args.dataset
        train_ds = (
            imdb["train"]
            if (args.train_samples is None or args.train_samples == 0)
            else get_subset(imdb["train"], args.train_samples)
        )
        train_ds = train_ds.map(
            eval(f"imdb_to_alpaca_{name}"),
            batched=True,
            remove_columns=imdb["train"].column_names,
        )
        train_ds = format_dataset(train_ds, "alpaca-clean")
    else:
        raise NotImplementedError

    if args.quantize:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = prepare_model_for_kbit_training(model)
        model.config.torch_dtype = torch.bfloat16
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda:0")

    print(model)

    if args.custom_mode != "full":
        if args.target_modules == "lm_head":
            target_modules = ["lm_head"]
        elif args.target_modules == "all":
            target_modules = find_all_linear_names(
                model, lm_head=True, quantize=args.quantize
            )
        else:
            target_modules = find_all_linear_names(model, quantize=args.quantize)
        print("target_modules:", target_modules)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_c=args.lora_c,
            lora_dropout=0.00,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        config.custom = {
            "mode": args.custom_mode,
            "submode": args.custom_submode,  
            "d_init": args.custom_d_init, 
            "sqrt_a": args.custom_sqrt_a,
            "identity": not args.custom_disable_identity,
            "init_type": args.init_type,
            "d_init_type": args.d_init_type,
            "custom_scaling": args.custom_scaling == 1,
            "shared_dim": {"A": args.shared_dim, "B": args.shared_dim}
            if args.shared_uv == 1
            else None,
            "dynamic_uv": args.dynamic_uv == 1,
            "shared_matrices": None,
            "shared_d": False,
            "shared_d_vector": None,
            "trainable_uv": False,
            "nonlin": 0,
            "use_float64": False,
            "norm_penalty": 0,
            "norm_alpha": 0.0,
        }
            
        print("="*50)
        print("LoRA Configuration:")
        print(f"Mode: {config.custom['mode']}")
        print(f"r: {config.r}")
        print(f"alpha: {config.lora_alpha}")
        if hasattr(config, 'lora_c'):
            print(f"c: {config.lora_c}")
        print(f"Target modules: {target_modules}")
        print(f"Custom config: {config.custom}")
        print("="*50)

        model = get_peft_model(model, config)
        print("model:",model)
        if args.quantize:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

    training_args = TrainingArguments(
        output_dir=output_dir,
        optim="adamw_torch",
        remove_unused_columns=False,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        dataloader_num_workers=4,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        eval_strategy="steps",
        save_strategy="no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.accumulation_steps,
        bf16=args.quantize,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=768,
        target_max_len=256,
        train_on_source=False,
        predict_with_generate=False,
    )

    # Remove wandb-related callbacks
    callbacks = []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    params_trainable = get_parameters_count(model, requires_grad=True)
    params_total = get_parameters_count(model, requires_grad=False)

    print(f"Trainable parameters: {params_trainable}")
    print(f"Total number of parameters: {params_total}")
    
    # Log parameters to CSV
    params_file = os.path.join(output_dir, "parameters.csv")
    log_metrics_to_csv(
        {"params_trainable": params_trainable, "params_total": params_total},
        params_file
    )

    model.train()
    trainer.train()

    # Final evaluation and logging
    eval_results_file = os.path.join(output_dir, "final_evaluation.csv")
    with torch.no_grad():
        for name in ds_names:
            res = trainer.evaluate(eval_ds[name], metric_key_prefix=f"eval_{name}")
            print(f"final eval {name}:", res)
            # Log each evaluation result
            log_metrics_to_csv(
                {"dataset": name, **res},
                eval_results_file,
                fieldnames=["dataset"] + list(res.keys())
            )

        if args.generate_samples:
            samples = generate_samples(model, tokenizer, [], model_id=job_id)
            samples_file = os.path.join(output_dir, "generated_samples.txt")
            with open(samples_file, 'w') as f:
                f.write("Generated Samples:\n")
                f.write("="*50 + "\n")
                for i, sample in enumerate(samples):
                    f.write(f"Sample {i+1}:\n{sample}\n")
                    f.write("="*50 + "\n")


if __name__ == "__main__":

    def shared_dim_type(value):
        if value.lower() == 'none':
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("shared_dim must be an integer or 'none'")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_mode",
        type=str,
        default="full",
        choices=["full", "lora", "only_b", "only_d", "elora", "sara"],
        help="mode of finetuning",
    )
    parser.add_argument(
        "--custom_submode",
        type=str,
        default="none",
        choices=["none", "lora_svd", "lora_half", "lora_half_svd"],
        help="submode of finetuning",
    )
    parser.add_argument("--custom_scaling", type=int, default=0)
    parser.add_argument("--shared_dim", type=shared_dim_type, default="none", help="Shared dimension size (integer) or 'none'")
    parser.add_argument("--shared_uv", type=int, default=0)
    parser.add_argument("--dynamic_uv", type=int, default=0)
    parser.add_argument("--custom_d_init", type=float, default=1.0)
    parser.add_argument("--custom_sqrt_a", type=float, default=5)
    parser.add_argument("--custom_disable_identity", action="store_true")
    parser.add_argument("--init_type", type=int, default=1)
    parser.add_argument("--d_init_type", type=int, default=0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_c", type=int, default=1)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--target_modules", type=str, default="lm_head")

    parser.add_argument(
        "--task", type=str, default="instruct", choices=["instruct", "imdb", "math"]
    )
    parser.add_argument("--dataset", type=str, default="alpaca-clean")
    parser.add_argument("--dataset_format", type=str, default="alpaca-clean")
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--metric_samples", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--metric_bs", type=int, default=10)
    parser.add_argument("--eval_bs", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--metrics_enabled", type=int, default=0, choices=[0, 1])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--quantize", type=bool, default = False)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--generate_samples", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    # New argument for output directory
    parser.add_argument("--output_dir", type=str, default="training_output")
    
    args = parser.parse_args()

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    run(args)