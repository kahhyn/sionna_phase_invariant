from utils.count_model_params import summarize_model
from models.phase_invariant_net import PhaseInvariantReceiver
from models.baseline_cnn import RealImagCNN, PhysicalFeatureCNN
from models.complex_no_interaction_cnn import ComplexCNNNoInteraction
from models.single_invariant_net import SingleBranchPhaseInvariantReceiver

# print("--------RealImage_CNN_size---------")
# model = RealImagCNN()
# summary = summarize_model(model, verbose=True)
#
# print("--------PhysicalFeature_CNN_size---------")
# model = PhysicalFeatureCNN(hidden=64,zero_complex=32,hidden_real=64,branch_layers=3,
#                            bits_per_symbol=2, kernel_size=4,use_norm=True)
# summary = summarize_model(model, verbose=True)
#
# print("--------PhaseInvariantNet_size---------")
# model = PhaseInvariantReceiver(
#             hidden_complex=32,
#             zero_complex=32,
#             hidden_real=64,
#             bits_per_symbol=2,
#             branch_layers=3,
#             kernel_size=3,
#             use_norm=True,
#             gate_type="swiglu"
#         )
# summary = summarize_model(model, verbose=True)

model = ComplexCNNNoInteraction(hidden_complex=64, hidden_real=32,
                                branch_layers=3, gate_type="swiglu")
summarize_model(model,verbose=True)

# model = PhysicalFeatureCNNMatched()
# summary = summarize_model(model, verbose=True)
model = SingleBranchPhaseInvariantReceiver(hidden_complex=32, hidden_real=32)
summarize_model(model, verbose=True)
