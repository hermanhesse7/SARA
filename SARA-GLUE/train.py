import config, GLUE_data_setup, peft_module, engine, utils
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import warnings
warnings.filterwarnings("ignore")

task = config.CONFIG.task
train_dataset = GLUE_data_setup.GLUEDataset(dataset_name=config.CONFIG.task, split = "train", k=config.CONFIG.k, seed=config.CONFIG.seed)
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
validation_dataset = GLUE_data_setup.GLUEDataset(dataset_name=config.CONFIG.task, split = validation_key)

train_loader = DataLoader(train_dataset, batch_size=config.CONFIG.train_batch,
                              num_workers=1, shuffle=True, pin_memory=True)

validation_loader = DataLoader(validation_dataset, batch_size=config.CONFIG.valid_batch,
                              num_workers=1, shuffle=False, pin_memory=True)

torch.manual_seed(config.CONFIG.seed)
torch.cuda.manual_seed(config.CONFIG.seed)
num_labels = 3 if config.CONFIG.task.startswith("mnli") else 1 if config.CONFIG.task=="stsb" else 2
#config = AutoConfig.from_pretrained(config.CONFIG.model_name)
model = AutoModelForSequenceClassification.from_pretrained(config.CONFIG.model_name, num_labels = num_labels, output_attentions = False,
                                                           output_hidden_states = False).to(config.CONFIG.device)

for layer in model.roberta.encoder.layer:
    #Create a new instance of your custom attention class with the same config and is_cross_attention settings
    custom_attn = peft_module.CustomRobertaSelfAttention(model.config)
    #Copy the weights from the original attention to the new custom attention
    custom_attn.load_state_dict(layer.attention.self.state_dict())
    #Replace the original attention with the custom one
    layer.attention.self = custom_attn


peft_module.add_peft_layers(model=model) 
peft_module.freeze_model(model)

# Identify classifier parameters and other parameters
classifier_parameters = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]
other_parameters = [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad]

# Create optimizer with parameter groups
optimizer = torch.optim.AdamW([
    {'params': other_parameters, 'lr': config.CONFIG.learning_rate},
    {'params': classifier_parameters, 'lr': config.CONFIG.classifier_learning_rate}
])

# Define the total number of training steps and warm-up steps
total_steps = len(train_loader) * config.CONFIG.epochs
warmup_steps = int(config.CONFIG.warmup_ratio * total_steps)

# Create the scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

utils.num_parameters(model=model)

engine.train(model=model, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_loader, val_dataloader=validation_loader)
