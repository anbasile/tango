##################
# Model settings #
##################

local pretrained_model = "t5-base";
local load_with_low_cpu_mem_usage = false;

####################
# Trainer settings #
####################

# Trainer settings, adjust to your use-case.
local training_steps = 20;  # total number of optimization steps to train for
local validate_every = 5;  # how often to validate and save checkpoints

local devices = 1;  # number of devices to train on (will use GPUs if enough are available, otherwise CPU)
local grad_accum = 1;  # number of gradient accumulation steps (changes the effective batch size)
# This is the batch size per GPU, ignoring gradient accumulation:
local batch_size = 2;
# So the effective batch size is `batch_size * grad_accum * devices`

local amp = false;  # use PyTorch's native automatic mixed precision

######################
# Optimizer settings #
######################

local warmup_steps = 20;
local learning_rate = 0.00005;  # you can probably use a higher LR for a small model like "gpt2"


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
    sampler: {
        type: "torch::DistributedSampler",
        shuffle: true,
        drop_last: true,
    },
};

local single_device_dataloader = {
    shuffle: true,
    batch_size: batch_size,
};

local dataloader = if devices > 1 then distributed_dataloader else single_device_dataloader;

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "snli",
        },
        /*"subset_data": {
            type: "subset-data",
            data: { type: "ref", ref: "raw_data" },
            max_samples: 10,
        },*/
        processed_data: {
            type: "snli-text2text",
            data: { type: "ref", ref: "raw_data" },
        },
        trained_model: {
            type: "transformers::finetune",
            model: {
                type: "transformers::finetune::from_pretrained",
                pretrained_model_name_or_path: pretrained_model,
                low_cpu_mem_usage: load_with_low_cpu_mem_usage,
            },
            tokenizer: {
                pretrained_model_name_or_path: pretrained_model
            },
            dataset_dict: { type: "ref", ref: "processed_data" },
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
        generations: {
            type: "transformers::run_generation_dataset",
            max_length: 5,
            input: {"type": "ref", "ref": "processed_data"},
            batch_size: batch_size,
            model: {"type": "ref", "ref": "trained_model"},
            prompt_field: "source",
            output_field: "generation",
            splits: ["validation"]
        }
    }
}
