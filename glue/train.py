# train.py
import config, GLUE_data_setup, peft_module, engine, utils
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
import os

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(config.CONFIG.seed)
torch.cuda.manual_seed(config.CONFIG.seed)

# Load datasets
task = config.CONFIG.task
train_dataset = GLUE_data_setup.GLUEDataset(
    dataset_name=config.CONFIG.task, split="train", k=config.CONFIG.k, seed=config.CONFIG.seed
)

# Determine the appropriate validation split key
validation_key = (
    "validation_mismatched" if task == "mnli-mm" else
    "validation_matched" if task == "mnli" else
    "validation"
)
validation_dataset = GLUE_data_setup.GLUEDataset(
    dataset_name=config.CONFIG.task, split=validation_key
)

# Split validation_dataset into new_validation and test_dataset
val_size = len(validation_dataset)
if val_size > 2000:
    new_val_size = 1000
    test_size = val_size - new_val_size
    print(f"Splitting validation set into {new_val_size} for validation and {test_size} for testing.")
else:
    test_size = val_size // 2
    new_val_size = val_size - test_size
    print(f"Splitting validation set into {new_val_size} for validation and {test_size} for testing.")

# Ensure reproducibility with a fixed seed
generator = torch.Generator().manual_seed(42)
new_validation_dataset, test_dataset = random_split(
    validation_dataset, [new_val_size, test_size], generator=generator
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=config.CONFIG.train_batch,
    num_workers=1, shuffle=True, pin_memory=True
)

validation_loader = DataLoader(
    new_validation_dataset, batch_size=config.CONFIG.valid_batch,
    num_workers=1, shuffle=False, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=config.CONFIG.valid_batch,
    num_workers=1, shuffle=False, pin_memory=True
)

# Save the test set for future reference (optional)
os.makedirs('saved_data', exist_ok=True)
torch.save(test_dataset, 'saved_data/test_dataset.pt')

# Initialize the model
num_labels = 3 if config.CONFIG.task.startswith("mnli") else 1 if config.CONFIG.task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(
    config.CONFIG.model_name, num_labels=num_labels, output_attentions=False,
    output_hidden_states=False
).to(config.CONFIG.device)

for layer in model.roberta.encoder.layer:
    #Create a new instance of your custom attention class with the same config and is_cross_attention settings
    custom_attn = peft_module.CustomRobertaSelfAttention(model.config)
    #Copy the weights from the original attention to the new custom attention
    custom_attn.load_state_dict(layer.attention.self.state_dict())
    #Replace the original attention with the custom one
    layer.attention.self = custom_attn

# Apply PEFT layers
peft_module.add_peft_layers(model=model) 
peft_module.freeze_model(model)

# Identify classifier and other parameters
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
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Display number of parameters (optional)
utils.num_parameters(model=model)

# Train the model and obtain the best checkpoint
best_checkpoint_path = engine.train(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    val_dataloader=validation_loader,
    test_dataloader=test_loader  # Pass the test_loader for final evaluation
)

# Optionally, load the best checkpoint and perform additional evaluations
# model.load_state_dict(torch.load(best_checkpoint_path))
# final_test_loss, final_test_preds, final_test_true = engine.evaluate(model, test_loader)
# final_test_metric = engine.eval_func(final_test_preds, final_test_true)
# print(f'Final Test Metric: {final_test_metric}')
