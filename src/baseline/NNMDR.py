# The results of this baseline is available in the original paper
import torch
from classifier import MLP, MaskedMLPNMMDR

if __name__=='__main__':
    results = torch.load('results/result.pth')
    state_dict = results['param_original']
    image_size = 784  # 28x28 images flattened
    hidden_sizes = 16
    num_mode = 4
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with open('data/submodels/params.pkl', 'rb') as f:
    classifier = MLP(image_size, hidden_sizes, num_classes).to(device)

    # Initialize the mask, mask has the same dimension as the model parameters
    classifier_masked = MaskedMLPNMMDR(classifier)
    