import torch
from core import model

# Load your trained model (use the best checkpoint)
net = model.MobileFacenet()
checkpoint = torch.load('./model/CASIA_B128_v2_20251220_101904/024.ckpt')
net.load_state_dict(checkpoint['net_state_dict'])
net.eval()  # Important: switch to eval mode (uses running stats)

# Dummy input (same size as your training input: likely 3x112x112)
dummy_input = torch.randn(1, 3, 112, 112)

# Trace with TorchScript + automatic optimizations (includes BN folding)
traced_model = torch.jit.trace(net, dummy_input)

# Optional: optimize_for_inference (explicitly fuses BN where possible)
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_model = optimize_for_mobile(traced_model)

# Save for deployment
optimized_model._save_for_lite_interpreter("mobilefacenet_optimized.ptl")  # for mobile
# or
torch.jit.save(traced_model, "mobilefacenet_scripted.pt")