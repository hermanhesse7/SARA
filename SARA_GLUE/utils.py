def num_parameters(model):
    # Calculate total parameters in the model
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_model_params}")

    # Calculate trainable parameters and classifier parameters
    total_trainable_params = 0
    total_classifier_params = 0

    for param_name, weights in model.named_parameters():
        if weights.requires_grad:
            total_trainable_params += weights.numel()
        if 'classifier' in param_name:
            total_classifier_params += weights.numel()

    # Subtract classifier parameters from trainable parameters (if needed)
    trainable_params = total_trainable_params - total_classifier_params

    # Calculate the percentage of trainable parameters
    percentage_trainable = (trainable_params / total_model_params) * 100

    print(f"Total trainable parameters (excluding classifier): {trainable_params}")
    print(f"Percentage of trainable parameters: {percentage_trainable:.4f}%")


