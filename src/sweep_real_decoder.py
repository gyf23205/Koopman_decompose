import os
import math
import torch
import wandb
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from classifier import MLP
from datasets.MNISTPerClass import MNISTPerClass
from datasets.MNIST import MNIST
from Autoencoder_real import KoopmanAutoencoder
from Autoencoder_functions import koopman_loss, collect_latent_states
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter


def test_sub_all(param_sub_all):
    with torch.autograd.no_grad():
        acc_sub = []
        for idx_model in range(hidden_k):
            model_acc = np.zeros((num_classes,))
            for idx_class in range(num_classes):
                total = 0
                correct = 0
                param_sub = param_sub_all[:, idx_model]
                classifier_sub = copy.deepcopy(classifier)
                classifier_sub.eval()
                nn.utils.vector_to_parameters(param_sub, classifier_sub.parameters())
                testloader = mnist_per_class.sub_trainloaders[idx_class]
                total = 0
                correct = 0
                for images, labels in testloader:
                    images = images.reshape(-1, 28*28).to(device)
                    labels = labels.to(device)
                    outputs = classifier_sub(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / total
                model_acc[idx_class] = accuracy
            acc_sub.append(model_acc)
        acc_sub = np.array(acc_sub)
        return np.sum(np.max(acc_sub, axis=0)) / num_classes

        


def test_recon_all(param_sub_all):
    with torch.autograd.no_grad():
        param_re = param_sub_all.sum(1)
        classifier_sub = copy.deepcopy(classifier)
        classifier_sub.eval()
        nn.utils.vector_to_parameters(param_re, classifier_sub.parameters())
        testloader = mnist.test_loader
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
        return accuracy

def compute_gradient_norm(model, norm_type=2):
    with torch.no_grad():
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def classifier_sub(x, p_vec):
    '''
    The forward propgation in this function must be the same as in the classifier
    '''
    idx, p_recon = 0, []
    for layer in classifier_shapes:
        layer_params = []
        for shape in layer:
            offset = np.prod(shape)
            layer_params.append(p_vec[idx:idx+offset].reshape(shape))
            idx += offset
        p_recon.append(layer_params)
    
    w0, b0 = p_recon[0]
    w1, b1 = p_recon[1]
    x = F.linear(x, w0, b0)
    x = F.relu(x)
    x = F.linear(x, w1, b1)
    return x

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


def compute_theta_sub_all(kae, z):
    ko = kae.K
    eigvals, eigvec_left = torch.linalg.eig(ko)
    eigvec_left = eigvec_left.real.detach()
    for e in range(hidden_k):
        wandb.log({f'Eigval/{e}':torch.abs(eigvals[e])}, step = n_batch * epoch + inner)
    # B = np.pad(np.eye(n_params), ((0, 0), (0, N_O - n_params)), mode='constant')
    eigvec_left_inv = torch.linalg.inv(eigvec_left)
    v = (kae.decoder(eigvec_left_inv)).T
    phi = eigvec_left @ z[-1, :]
    param_sub_all = v @ torch.diag(phi)
    return param_sub_all, eigvals

def compute_l_sub(param_sub, target_classes, images, labels):
    # classifier_sub = MLP(image_size, hidden_c, num_classes).to(device)
    # nn.utils.vector_to_parameters(param_sub, classifier_sub.parameters())
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    mask = torch.isin(labels, target_classes.clone().detach())
    images = images[mask]
    labels = labels[mask]
    outputs = classifier_sub(images, param_sub)
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
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

def get_target_classes(param_sub, candidates, images, labels):
    with torch.autograd.no_grad():
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = classifier_sub(images, param_sub)
        _, pred = torch.max(outputs, 1)
        acc = []
        for i in candidates:
            if i != -1:
                mask = labels == i
                acc.append((pred[mask] == labels[mask]).sum()/mask.sum())
            else:
                acc.append(-1)
        _, top_idx = torch.topk(torch.tensor(acc), num_class_per_mode)
        return candidates[top_idx]
        

    

if __name__=='__main__':
    # Hyperparameters
    # Set training to be deterministic
    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # writer = SummaryWriter(log_dir='results/log_eig')

    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')

    save = True
    image_size = 784  # 28x28 images flattened
    num_classes = 10
    batch_size = 128
    num_epochs = 10
    T = 15
    lr_classifier = 1e-3
    lr_kae = 1e-2
    # T = 5
    c1 = 1
    c2 = 1
    c3 = 1
    p = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(
        project='Koopman_decompose'
    )

    
    # Hyperparameters to be tuned
    hidden_c = wandb.config.hidden_c
    hidden_k = wandb.config.hidden_k
    num_mode_dom = wandb.config.num_mode_dom

    # hidden_c = 16
    # hidden_k = 4
    # num_mode_dom = 3

    num_class_per_mode = int(math.ceil(num_classes/num_mode_dom))

    # T = 5
    # hidden_c = 8
    # hidden_k = 10
    # lr_classifier = 1e-3
    # lr_kae = 1e-3

    # Load datasets
    mnist_per_class = MNISTPerClass(batch_size=batch_size)
    mnist = MNIST(batch_size=batch_size)

    # Build the classifier
    classifier = MLP(image_size, hidden_c, num_classes).to(device)
    # for name, param in classifier.named_parameters():
    #     print(name)
    #     print(param.shape)
    #     print()
    classifier.train()
    criterion_classifier = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr_classifier)
    classifier_shapes = [
        [(hidden_c, image_size), (hidden_c)],
        [(num_classes, hidden_c), (num_classes)]
    ]
    
    # Build the KAE
    param_vec = parameters_to_vector(classifier.parameters()) 
    state_dim = param_vec.shape[0]
    kae = KoopmanAutoencoder(state_dim=state_dim, hidden_dim=hidden_k).to(device)
    kae.train()
    criterion_kae = koopman_loss
    mse = torch.nn.MSELoss()
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
    

    # for g in optimizer_classifier.param_groups:
    #     g['lr'] = lr_classifier

    n_params = len(params_snapshots[0])
    print(n_params)
    classifier.train()
    kae.train()
    n_batch = len(train_loader_classifier)
    for epoch in range(num_epochs):
        running_loss = 0.0
        # current_params = parameters_to_vector(classifier.parameters())
        for inner, (images, labels) in enumerate(train_loader_classifier):
            loss_sub = 0.0
            # loss_eig = 0.0
            loss_classifier = compute_l_classifier(classifier, images, labels)
            loss_kae, z = compute_l_kae(kae, params_snapshots) # Koopman operator is updated here.
            N_O = z.shape[-1]
            param_sub_all, eigvals = compute_theta_sub_all(kae, z)

            order = torch.argsort(eigvals.abs())
            candidates = torch.linspace(0, 9, 10, dtype=int, device=device)
            for _ in range(num_mode_dom):
                best_mode = torch.argmax(order)
                order[best_mode] = -1
                target_classes = get_target_classes(param_sub_all[:, best_mode], candidates, images, labels)
                if -1 in candidates[target_classes]:
                    rest = ~(candidates == -1)
                    target_classes = candidates[rest]
                candidates[target_classes] = -1

            #     trainloader_sub = mnist_per_class.sub_trainloaders[idx_sub]
            #     images_sub, labels_sub = next(iter(trainloader_sub))
                loss_sub = loss_sub + compute_l_sub(param_sub_all[:, best_mode], target_classes, images, labels)
            loss = loss_kae + loss_sub
            wandb.log({'loss_c': loss_classifier.item()}, step = n_batch * epoch + inner)
            # writer.add_scalar('Loss/classification', loss_classifier.item(), epoch * inner)
            # writer.add_scalar('Loss/eig', loss_eig.item(), epoch * inner)
            wandb.log({'loss_k': loss_kae.item()}, step = n_batch * epoch + inner)
            # writer.add_scalar('Loss/kae', loss_kae.item(), epoch * inner)
            # writer.add_scalar('Loss/sub', loss_sub.item(), epoch * inner)
            wandb.log({'loss_all': loss.item()}, step = n_batch * epoch + inner)
            # writer.add_scalar('Loss/all', loss.item(), epoch * inner)

            optimizer_kae.zero_grad()
            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_kae.step()
            optimizer_classifier.step()
            params_snapshots.append(parameters_to_vector(classifier.parameters()))
            params_snapshots.pop(0) # Maybe don't pop?
            running_loss += loss.item()
            # print(compute_gradient_norm(classifier))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader_classifier):.4f}')
    # Test the original classifier again
    test_classifier(classifier, test_loader)

    acc_recon = test_recon_all(param_sub_all)
    acc_sub = test_sub_all(param_sub_all)
    wandb.log({'acc_all':acc_recon})
    wandb.log({'acc_sub':acc_sub})
    wandb.log({'acc':acc_sub + acc_recon})
    # writer.flush()