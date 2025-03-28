import torch
import numpy as np
import pickle
import copy
import torch.nn as nn
import torch.optim as optim
from classifier import MLP
from datasets.MNISTPerClass import MNISTPerClass
from datasets.MNIST import MNIST
from Autoencoder_class import KoopmanAutoencoder
from Autoencoder_functions import koopman_loss, collect_latent_states
from torch.nn.utils import parameters_to_vector
from scipy.linalg import eig, inv

def compute_l_classifier(model, images, labels):
    # Move tensors to device
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion_classifier(outputs, labels)
    return loss

def compute_l_kae(kae, params_snapshots):
    x = torch.stack(params_snapshots, dim=0).to(device)
    latents, latents_next = collect_latent_states(kae, x)
    kae.compute_koopman_operator(latents, latents_next)
    x_hat, z, z_pred = kae(x)
    recon_loss, state_pred_loss, koopman_pred_loss = koopman_loss(x, x_hat, z_pred, p, kae)
    loss_kae = c1*recon_loss + c2*state_pred_loss + c3*koopman_pred_loss # + c4*k_norm_loss      
    return loss_kae, z

def compute_theta_sub(kae, z, idx_sub):
    ko = kae.K.detach().cpu().numpy()
    eigval, eigvec_left = eig(ko, left=True, right=False)
    B = np.pad(np.eye(n_params), ((0, 0), (0, N_O - n_params)), mode='constant')
    eigvec_left_inv = inv(eigvec_left)
    v = torch.tensor((B @ eigvec_left_inv)[:, idx_sub], dtype=torch.float32, device=device)
    phi_i = torch.tensor(eigvec_left[idx_sub, :], dtype=torch.float32, device=device) @ z[-1, :]
    param_sub = phi_i * v
    return param_sub

def compute_theta_sub_all(kae, z):
    ko = kae.K.detach().cpu().numpy()
    eigval, eigvec_left = eig(ko, left=True, right=False)
    B = np.pad(np.eye(n_params), ((0, 0), (0, N_O - n_params)), mode='constant')
    eigvec_left_inv = inv(eigvec_left)
    v = torch.tensor((B @ eigvec_left_inv), dtype=torch.float32, device=device)[:, :num_classes]
    phi = torch.tensor(eigvec_left, dtype=torch.float32, device=device) @ z[-1, :]
    param_sub_all = v @ torch.diag(phi[:num_classes])
    return param_sub_all

def compute_l_sub(param_sub, images, labels):
    classifier_sub = copy.deepcopy(classifier)
    nn.utils.vector_to_parameters(param_sub, classifier_sub.parameters())
    
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = classifier_sub(images)
    loss = criterion_classifier(outputs, labels)
    return loss

def test_classifier(model, test_loader):
    model.eval()  # evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')


    

if __name__=='__main__':
    # Hyperparameters
    image_size = 784  # 28x28 images flattened
    hidden_sizes = 4
    num_classes = 10
    batch_size = 64
    lr_classifier = 1e-3
    lr_kae = 1e-3
    num_epochs = 10
    T = 3
    c1 = 1
    c2 = 1
    c3 = 1
    p = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    mnist_per_class = MNISTPerClass(batch_size=batch_size)
    mnist = MNIST(batch_size=batch_size)

    # Build the classifier
    classifier = MLP(image_size, hidden_sizes, num_classes).to(device)
    classifier.train()
    criterion_classifier = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr_classifier)
    # Build the KAE
    param_vec = parameters_to_vector(classifier.parameters()) 
    state_dim = param_vec.shape[0]
    hidden_dim = 64
    kae = KoopmanAutoencoder(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    kae.train()
    criterion_kae = koopman_loss
    optimizer_kae = optim.Adam(kae.parameters(), lr=lr_kae)
    
    # Get T number of snapshots first
    train_loader_classifier = mnist.train_loader
    test_loader = mnist.test_loader
    params_snapshots = []
    for epoch in range(T):
        running_loss = 0.0
        params_snapshots.append(parameters_to_vector(classifier.parameters()))
        for images, labels in train_loader_classifier:
            # train_loader_per_class = mnist_per_class.sub_trainloaders[epoch%10]
            loss_classifier = compute_l_classifier(classifier, images, labels)
            optimizer_classifier.zero_grad()
            loss_classifier.backward()
            optimizer_classifier.step()
            running_loss += loss_classifier.item()
        print(f'Epoch [{epoch+1}/{T}], Loss: {running_loss/len(train_loader_classifier):.4f}')
    test_classifier(classifier, test_loader)
    
    n_params = len(params_snapshots[0])
    print(n_params)
    for epoch in range(num_epochs):
        # current_params = parameters_to_vector(classifier.parameters())
        for images, labels in train_loader_classifier:
            idx_sub = epoch % num_classes
            trainloader_sub = mnist_per_class.sub_trainloaders[idx_sub]
            images_sub, labels_sub = next(iter(trainloader_sub))

            loss_classifier = compute_l_classifier(classifier, images, labels)
            loss_kae, z = compute_l_kae(kae, params_snapshots)
            N_O = z.shape[-1]
            param_sub = compute_theta_sub(kae, z, idx_sub)
            loss_sub = compute_l_sub(param_sub, images_sub, labels_sub)
            loss = loss_classifier + loss_kae + loss_sub

            optimizer_kae.zero_grad()
            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_kae.step()
            optimizer_classifier.step()
            params_snapshots.append(parameters_to_vector(classifier.parameters()))
            params_snapshots.pop(0)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()/len(train_loader_classifier):.4f}')

    # Save parameters and Koopman operator
    loss_kae, z = compute_l_kae(kae, params_snapshots)
    param_sub_all = compute_theta_sub_all(kae, z)
    with open('data/submodels/params.pkl', 'wb') as f:
        pickle.dump(param_sub_all, f)
    _, z = compute_l_kae(kae, params_snapshots)

    with torch.autograd.no_grad():
        for idx_sub in range(num_classes):
            # param_sub = compute_theta_sub(kae, z, idx_sub)
            param_sub = param_sub_all[:, idx_sub]
            classifier_sub = copy.deepcopy(classifier)
            classifier_sub.eval()
            nn.utils.vector_to_parameters(param_sub, classifier_sub.parameters())
            testloader = mnist_per_class.sub_testloaders[idx_sub]
            total = 0
            correct = 0
            for images, labels in testloader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = classifier_sub(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Test Accuracy of class {idx_sub:d}: {accuracy:.2f}%')
    # with open('.data/snapshots/params_snapshots.pkl', 'wb') as f:
    #     pickle.dump(params_snapshots, f)
