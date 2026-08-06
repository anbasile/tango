##################
# Model settings #
##################

local pretrained_model = "gpt2";
# local pretrained_model = "EleutherAI/gpt-j-6B";

# This doesn't seem to work with gpt2, but works fine with gpt-j.
local load_with_low_cpu_mem_usage = std.startsWith(pretrained_model, "EleutherAI/gpt-j");

####################
# Trainer settings #
####################

# Trainer settings, adjust to your use-case.
local training_steps = 200;  # total number of optimization steps to train for
local validate_every = 20;  # how often to validate and save checkpoints

local devices = 1;  # number of devices to train on (will use GPUs if enough are available, otherwise CPU)
local grad_accum = 1;  # number of gradient accumulation steps (changes the effective batch size)
# This is the batch size per GPU, ignoring gradient accumulation:
local batch_size = 8;
# So the effective batch size is `batch_size * grad_accum * devices`

local amp = false;  # use PyTorch's native automatic mixed precision

######################
# Optimizer settings #
######################

local warmup_steps = 20;
local learning_rate = 0.00005;  # you can probably use a higher LR for a small model like "gpt2"


# <----- you probably don't need to edit below this line ----> #


local training_engine = {
    type: "torch",
    optimizer: {
        type: "torch::AdamW",
        lr: learning_rate,
        betas: [0.9, 0.95],
        eps: 1e-6,
    },
    lr_scheduler: {
        type: "transformers::linear",
        num_warmup_steps: warmup_steps,
        num_training_steps: training_steps,
    },
    amp: amp,
};

local distributed_dataloader = {
    batch_size: batch_size,
    collate_fn: { type: "transformers::DefaultDataCollator" },
    sampler: {
        type: "torch::DistributedSampler",
        shuffle: true,
        drop_last: true,
    },
};

local single_device_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: { type: "transformers::DefaultDataCollator" },
};

local dataloader = if devices > 1 then distributed_dataloader else single_device_dataloader;

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "Salesforce/wikitext",
            name: "wikitext-2-raw-v1",
        },
        tokenized_data: {
            type: "tokenize_data",
            dataset: { type: "ref", ref: "raw_data" },
            tokenizer: { pretrained_model_name_or_path: pretrained_model }
        },
        trained_model: {
            type: "torch::train",
            model: {
                type: "transformers::AutoModelForCausalLM::from_pretrained",
                pretrained_model_name_or_path: pretrained_model,
                low_cpu_mem_usage: load_with_low_cpu_mem_usage,
            },
            dataset_dict: { type: "ref", ref: "tokenized_data" },
            train_dataloader: dataloader,
            validation_split: "validation",
            grad_accum: grad_accum,
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            log_every: 1,
            device_count: devices,
            training_engine: training_engine,
        },
        final_metrics: {
            type: "torch::eval",
            model: { type: "ref", ref: "trained_model" },
            dataset_dict: { type: "ref", ref: "tokenized_data" },
            dataloader: single_device_dataloader,
            test_split: "test",
        },
    }
}
