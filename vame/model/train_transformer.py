import torch
from torch import nn
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
from pathlib import Path
from vame.util import read_config
from vame.model.rnn_vae import *

def run_train(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    pretrained_weights = cfg['pretrained_weights']
    pretrained_model = cfg['pretrained_model']
    
    print(f"Training Transformer VAE - model name: {model_name}\n")
    
    # Create necessary directories
    if not os.path.exists(os.path.join(cfg['project_path'],'model','best_model',"")):
        os.makedirs(os.path.join(cfg['project_path'],'model','best_model',""))
        os.makedirs(os.path.join(cfg['project_path'],'model','best_model','snapshots',""))
        os.makedirs(os.path.join(cfg['project_path'],'model','model_losses',""))

    # Check for GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:', torch.cuda.is_available())
        print('GPU used:', torch.cuda.get_device_name(0))
    else:
        torch.device("cpu")
        print("Warning: GPU not found, using CPU (slow)")

    # Hyperparameters
    TRAIN_BATCH_SIZE = cfg['batch_size']
    TEST_BATCH_SIZE = int(cfg['batch_size']/4)
    EPOCHS = cfg['max_epochs']
    ZDIMS = cfg['zdims']
    BETA = cfg['beta']
    SNAPSHOT = cfg['model_snapshot']
    LEARNING_RATE = cfg['learning_rate']
    NUM_FEATURES = cfg['num_features']
    TEMPORAL_WINDOW = cfg['time_window']*2
    
    # New Transformer-specific hyperparameters
    D_MODEL = cfg.get('d_model', 256)  # Default if not in config
    NHEAD = cfg.get('n_head', 8)
    NUM_LAYERS = cfg.get('transformer_layers', 3)
    DROPOUT = cfg.get('transformer_dropout', 0.1)

    # Loss parameters
    KL_START = cfg['kl_start']
    ANNEALTIME = cfg['annealtime']
    anneal_function = cfg['anneal_function']
    
    # Initialize model
    if use_gpu:
        model = TransformerVAE(
            temporal_window=TEMPORAL_WINDOW,
            zdims=ZDIMS,
            num_features=NUM_FEATURES,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).cuda()
    else:
        model = TransformerVAE(
            temporal_window=TEMPORAL_WINDOW,
            zdims=ZDIMS,
            num_features=NUM_FEATURES,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )

    # Load pretrained weights if specified
    if pretrained_weights:
        try:
            print(f"Loading pretrained weights from model: {pretrained_model}")
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl')))
            KL_START = 0
            ANNEALTIME = 1
        except:
            print(f"Could not load pretrained model from {pretrained_model}")

    # Load datasets
    trainset = SEQ_DATASET(
        os.path.join(cfg['project_path'],"data", "train",""), 
        data='train_seq.npy',
        train=True,
        temporal_window=TEMPORAL_WINDOW
    )
    testset = SEQ_DATASET(
        os.path.join(cfg['project_path'],"data", "train",""),
        data='test_seq.npy',
        train=False,
        temporal_window=TEMPORAL_WINDOW
    )

    train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        'min',
        factor=cfg['scheduler_gamma'],
        patience=cfg['scheduler_step_size'],
        threshold=1e-3,
        threshold_mode='rel',
        verbose=True
    )

    # Training metrics
    BEST_LOSS = float('inf')
    convergence = 0
    train_losses = []
    test_losses = []
    kl_losses = []
    weight_values = []
    mse_losses = []

    print(f'Latent Dimensions: {ZDIMS}, Time window: {cfg["time_window"]}, Batch Size: {TRAIN_BATCH_SIZE}, Beta: {BETA}, lr: {LEARNING_RATE}\n')

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch: {epoch}")
        
        # Train
        weight = kl_annealing(epoch, KL_START, ANNEALTIME, anneal_function)
        train_loss, mse_loss, kl_loss = train_transformer(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            beta=BETA,
            kl_weight=weight,
            use_gpu=use_gpu
        )
        
        # Test
        test_loss, test_mse = test_transformer(
            test_loader=test_loader,
            model=model,
            beta=BETA,
            kl_weight=weight,
            use_gpu=use_gpu
        )

        # Update scheduler
        scheduler.step(test_loss)

        # Log losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        kl_losses.append(kl_loss)
        weight_values.append(weight)
        mse_losses.append(mse_loss)

        # Save best model
        if weight > 0.99 and test_mse < BEST_LOSS:
            BEST_LOSS = test_mse
            print("Saving model!")
            torch.save(model.state_dict(), 
                      os.path.join(cfg['project_path'],"model", "best_model",
                                 model_name+'_'+cfg['Project']+'.pkl'))
            convergence = 0
        else:
            convergence += 1

        # Save snapshot if needed
        if epoch % SNAPSHOT == 0:
            print("Saving model snapshot!")
            torch.save(model.state_dict(), 
                      os.path.join(cfg['project_path'],'model','best_model','snapshots',
                                 f"{model_name}_{cfg['Project']}_epoch_{epoch}.pkl"))

        # Check convergence
        if convergence > cfg['model_convergence']:
            print('Model converged!')
            break

        # Save losses
        np.save(os.path.join(cfg['project_path'],'model','model_losses','train_losses_'+model_name), train_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','test_losses_'+model_name), test_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','kl_losses_'+model_name), kl_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','weight_values_'+model_name), weight_values)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','mse_losses_'+model_name), mse_losses)

def train_transformer(train_loader, model, optimizer, beta, kl_weight, use_gpu):
    model.train()
    total_loss = 0
    total_mse = 0
    total_kl = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.float().permute(0, 2, 1)  # Adjust dimensions for transformer
        if use_gpu:
            data = data.cuda()
        
        # Forward pass
        recon_batch, _, mu, logvar = model(data)
        
        # Compute losses
        mse_loss = nn.MSELoss(reduction='mean')(recon_batch, data)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = mse_loss + beta * kl_weight * kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_kl += kl_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    
    print(f'====> Train set loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, KL: {avg_kl:.4f})')
    return avg_loss, avg_mse, avg_kl

def test_transformer(test_loader, model, beta, kl_weight, use_gpu):
    model.eval()
    total_loss = 0
    total_mse = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.permute(0, 2, 1)  # Adjust dimensions for transformer
            if use_gpu:
                data = data.cuda()
            
            # Forward pass
            recon_batch, _, mu, logvar = model(data)
            
            # Compute losses
            mse_loss = nn.MSELoss(reduction='mean')(recon_batch, data)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss
            loss = mse_loss + beta * kl_weight * kl_loss
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_mse = total_mse / len(test_loader)
    
    print(f'====> Test set loss: {avg_loss:.4f} (MSE: {avg_mse:.4f})')
    return avg_loss, avg_mse
