def num_parameters(model):
    total_params_1 = 0
    total_params_2 = 0
    for param_name, weights in model.named_parameters():
      if weights.requires_grad == True:
        total_params_1 += weights.numel()

    for param_name, weights in model.named_parameters():
      if 'classifier' in param_name:
        total_params_2 += weights.numel()

    print("total_params:", total_params_1-total_params_2)    
