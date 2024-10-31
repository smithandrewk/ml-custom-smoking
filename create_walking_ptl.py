import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torch.load(f'model.pt',weights_only=False,map_location='cpu')
model.eval();

scripted_model = torch.jit.script(model)
scripted_model.save("model.ptl")
optimized_scripted_module = optimize_for_mobile(script_module=scripted_model)
optimized_scripted_module._save_for_lite_interpreter("model.ptl")