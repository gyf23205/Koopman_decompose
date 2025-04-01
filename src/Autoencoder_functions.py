from tqdm import tqdm
import math
import pickle
import scipy
from torch.utils.data import DataLoader, TensorDataset
import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# import dill

def collect_latent_states(model, param_traj): # collect lifted (=latent) states 
    z = model.encoder(param_traj)
    latents = z[:-1, :]  # States at time t
    latents_next = z[1:, :]  # States at time t+1
    
    # latent_X = torch.cat(latent_X_list, dim=0)
    # latent_Y = torch.cat(latent_Y_list, dim=0)
    
    return latents, latents_next

def save_checkpoint(epoch, model, optimizer, loss, filename="model/koopman_checkpoint.pkl"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, filename, pickle_module=dill)
    print(f"Checkpoint saved at epoch {epoch}")

def koopman_loss(x, x_hat, z_pred, p, model): # compute loss functions
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(x_hat, x) # Reconstruction loss (between x and x_hat)
    state_pred_loss = 0.0 # Prediction loss (up to p time steps)
    latent_pred_loss = 0.0 # Prediction loss in lifted space (up to p time steps)
    time_steps, _ = x.size()

    # predicted_latent = z_pred
    true_steps = 0
    for step in range(1, p + 1):
        if step >= time_steps:
            break
        true_steps += 1
        # True & lifted future state 
        true_future_state = x[step:, :]
        true_future_latent = model.encoder(true_future_state)

        # Predict future latent states using Koopman operator
        predicted_latent = z_pred[:-step, :]
        for _ in range(step - 1):
            predicted_latent = torch.matmul(predicted_latent, model.K.T)

        # Decoded predicted future states = Identity(z_pred)
        predicted_state = model.decoder(predicted_latent)
        # print(np.shape(predicted_latent))

        # State Prediction Loss
        state_pred_loss += mse_loss(predicted_state, true_future_state[:predicted_state.size(0), :])

        # Latent Prediction Loss
        latent_pred_loss += mse_loss(predicted_latent, true_future_latent[:predicted_latent.size(0), :])


    # Average prediction losses over p time steps
    state_pred_loss /= true_steps
    latent_pred_loss /= true_steps

    return recon_loss, state_pred_loss, latent_pred_loss

def convert_numpy_shape(input_data, return_tensor=True): # just to convert data shape
    reshaped_data = np.transpose(input_data, (2, 1, 0))  # (num_samples, time_steps, state_dim)

    if return_tensor:
        return torch.tensor(reshaped_data, dtype=torch.float32)
    else:
        return reshaped_data 
    
def train_KAE(model, state_dimension, hidden_dimension, batch_size, training_number, lr, 
              Koopman_training_trajectory, c1, c2, c3, c4, p, temperature, state_bounds, dynamics_model, testing_epochs,
              plot_number, testing_number, input_dimension, iteration_len, Koopman_testing_trajectory,
              CP_confidence=0.99, CP_sample_number = 1000, num_epochs = 100, Conformal = False, display = True):
    
    mean_error_history = np.zeros([len(testing_epochs)])
    var_error_history = np.zeros([len(testing_epochs)])
    history_index = 0

    optimizer = optim.Adam(model.parameters(), lr=lr)

    data = convert_numpy_shape(Koopman_training_trajectory, (2, 1, 0))

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=dummy_collate)
    # Ignore collate, it does not do anything. This structure is due to conformal part.

    # separation_ratio = 0.5 is in the custom_collcate func!

    for epoch in range(num_epochs):
        total_loss, total_recon, total_state_pred, total_latent_pred, total_conformal_loss = 0, 0, 0, 0, 0
        
        # Initialize Koopman operator
        if epoch == 0:
            latent_X, latent_Y = collect_latent_states(model, data_loader, True)
            model.compute_koopman_operator(latent_X, latent_Y)  
            
        # for batch in dataloader:
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as progress_bar:
            for x, data_rs_samples in progress_bar:
                optimizer.zero_grad()
                x_hat, z, z_pred = model(x)            
                
                recon_loss, state_pred_loss, koopman_pred_loss = koopman_loss(x, x_hat, z_pred, p, model)
                # k_norm_loss = torch.linalg.matrix_norm(model.K)
                loss = c1*recon_loss + c2*state_pred_loss + c3*koopman_pred_loss # + c4*k_norm_loss      

                loss.backward()
                optimizer.step()

                latent_X, latent_Y = collect_latent_states(model, data_loader, True)
                model.compute_koopman_operator(latent_X, latent_Y)  # Update Koopman operator using EDMD
                
                # Track losses
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_state_pred += state_pred_loss.item()
                total_latent_pred += koopman_pred_loss.item()

        # Print progress
        if display:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Recon: {total_recon:.4f}, State Pred: {total_state_pred:.4f}, Latent Pred: {total_latent_pred:.4f}")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(epoch+1, model, optimizer, total_loss, f"models/"+dynamics_model+"_KAE_cp_epch"+str(epoch+1)+"_trj"+str(training_number)+".pkl")

        if (epoch+1) in testing_epochs:
            KAE_error = test_KAE(model, plot_number, testing_number, state_dimension, input_dimension, iteration_len, Koopman_testing_trajectory)
            mean_error_history[history_index] = np.mean(KAE_error)
            var_error_history[history_index] = np.var(KAE_error)
            history_index += 1

    if display:
        print("Training complete!")
        
    torch.save(model, "models/"+dynamics_model+"_KAE_final"+str(num_epochs)+"_trj"+str(training_number)+".pkl", pickle_module=dill)

    return mean_error_history, var_error_history

def test_KAE(model, plot_number, testing_number, state_dimension, input_dimension, 
            iteration_len, Koopman_testing_trajectory):
    if plot_number > testing_number:
        plot_number = testing_number

    for i in range(0,plot_number):
        plt.plot(Koopman_testing_trajectory[0,0,i],Koopman_testing_trajectory[1,0,i],'ro')
        plt.plot(Koopman_testing_trajectory[0,:,i],Koopman_testing_trajectory[1,:,i],'--')

    Koopman_test_trajectory = np.zeros([state_dimension + input_dimension, iteration_len, plot_number]) # [state, input]
    model.eval()

    for i in range(0,plot_number):
        Koopman_test_trajectory[:,0,i] = Koopman_testing_trajectory[:,0,i]
        for j in range(0,iteration_len-1):
            current_input = torch.tensor(Koopman_test_trajectory[:,j,i],dtype=torch.float32)
            current_input = current_input.unsqueeze(0)
            current_input = current_input.unsqueeze(0)
            _, _, z_pred = model(current_input)
            x_pred = model.decode(z_pred)
            Koopman_test_trajectory[:,j+1,i] = x_pred.numpy()

    for i in range(0,plot_number):
        plt.plot(Koopman_test_trajectory[0,:,i],Koopman_test_trajectory[1,:,i],'-')

    assert Koopman_test_trajectory.shape == Koopman_testing_trajectory.shape # shape check
    return np.linalg.norm(Koopman_test_trajectory - Koopman_testing_trajectory,axis = 0)
    
def dummy_collate(batch): # It is a dummy, don't do anything

    batch = torch.stack([item for item in batch]) 
    batch_size, _, _ = batch.shape  
    separation_ratio = 0

    num_to_separate = int(batch_size * separation_ratio)

    indices = torch.randperm(batch_size)
    separated_indices = indices[:num_to_separate]
    train_indices = indices[num_to_separate:]

    separated_batch = batch[separated_indices, :, :]
    train_batch = batch[train_indices, :, :]

    return train_batch, separated_batch  # Both (batch_size, time_steps, state_dim)
