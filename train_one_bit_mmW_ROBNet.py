# This script is used to train the mmW-ROBNet for a QPSK constellation.

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
# import torchvision as tv
from PIL import Image
from create_data_random_chan import func_create_mmwave_chan_3
from create_data_random_chan import func_create_measurements
from create_data_random_chan import func_create_G_R
from create_data_random_chan import func_multiply_parallel
# from create_data_random_chan import func_tensor_grad
from create_data_random_chan import func_tensor_grad_LR
from create_data_random_chan import func_to_CNN_chan
from create_data_random_chan import func_to_CNN_chan_three_input
# from create_data_random_chan import func_from_CNN_chan

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)






# Section 1: Parameters
N = 64 # Number of antennas x 2
K = 4 # Number of multiple users
snr_db_arr = np.arange(15,16)
# snr_db_arr = np.array([15,30])
snr_lin_arr = 10**(0.1*snr_db_arr)
phi_min = -60*np.pi/180
phi_max = 60*np.pi/180
num_epochs = 2000
num_batches = 2000
num_snapshots = 32
gamma = 0.00001
lam = 5 # Regularization parameter
beta = 2 # Tanh scaling for regularization

if(device=='cuda'):
    num_batches = 2000

flag_load_prev = 0
checkpoint_file = 'Saved_Networks/mmw_robnet_matched_user_scale_mmwave_H_regularized/checkpoint.pth.tar'






# Section 2: Class and optimizer definition

# ROBNet architecture
class Net_One_Bit(nn.Module):
    """
    Neural network class for the mmW-ROBNet.

    Args:
        in_dim (int): Input dimension.
        batch_size (int): Batch size.
        cnn_chan (int): Number of channels in the CNN layer.

    Attributes:
        alpha (nn.Parameter): Parameter for alpha.
        norm_factor (nn.Parameter): Parameter for normalization factor.

    Methods:
        loss_eval: Calculate loss for the network.
        tanh_quant: Apply tanh quantization.
        tanh_loss: Calculate loss for tanh quantization.
        user_scale_tanh_loss: Calculate loss for user-scaled tanh quantization.
        normalize: Normalize the input.

    """
    def __init__(self,in_dim=4,batch_size=32,cnn_chan=64):
        super(Net_One_Bit, self).__init__()
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.cnn_chan = cnn_chan
        self.H0 = torch.zeros([1,batch_size,1024],device=device)

        self.alpha = nn.Parameter(0.01*torch.rand(1,dtype=torch.float32), requires_grad=True)
        self.norm_factor = nn.Parameter(torch.rand(1,dtype=torch.float32), requires_grad=True)
        
        self.bn = nn.BatchNorm1d(cnn_chan)
        self.conv1 = nn.Conv1d(6, cnn_chan, 3, padding=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.func_to_CNN_chan = func_to_CNN_chan_three_input
        self.func_tensor_grad_LR = func_tensor_grad_LR
        self.func_multiply_parallel = func_multiply_parallel
        
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.loss2 = nn.L1Loss()
    
    def loss_eval(self, x_R, x_pilot_R, lam=5, beta=2, user_scale_fact=1):
        """
        Calculate the loss for the network.

        Args:
            x_R (Tensor): The network's output.
            x_pilot_R (Tensor): The pilot data.
            lam (float): Regularization parameter.
            beta (float): Tanh scaling for regularization.
            user_scale_fact (float): User scaling factor.

        Returns:
            loss (Tensor): The loss value.
        """
        x_pilot_R_scaled = user_scale_fact * x_pilot_R
        x_R_scaled = user_scale_fact * x_R
        loss = 1 / (lam + 1) * (self.loss(x_R_scaled, x_pilot_R_scaled) + lam * self.user_scale_tanh_loss(x_R, x_pilot_R, user_scale_fact, beta=beta))
        return loss

    def tanh_quant(self, x_in, beta=2):
        """
        Apply tanh quantization.

        Args:
            x_in (Tensor): Input data.
            beta (float): Tanh scaling.

        Returns:
            x_quant (Tensor): Quantized data.
        """
        x_quant = torch.tanh(beta * x_in)
        return x_quant
    
    def tanh_loss(self, x_in, x_pilot_R, beta=2):
        """
        Calculate the loss for tanh quantization.

        Args:
            x_in (Tensor): Input data.
            x_pilot_R (Tensor): The pilot data.
            beta (float): Tanh scaling.

        Returns:
            loss (Tensor): The loss value.
        """
        return self.loss(self.tanh_quant(x_in, beta=beta), x_pilot_R)
    
    def user_scale_tanh_loss(self, x_in, x_pilot_R, user_scale_fact, beta=10):
        """
        Calculate the loss for user-scaled tanh quantization.

        Args:
            x_in (Tensor): Input data.
            x_pilot_R (Tensor): The pilot data.
            user_scale_fact (float): User scaling factor.
            beta (float): Tanh scaling.

        Returns:
            loss (Tensor): The loss value.
        """
        x_pilot_scaled = user_scale_fact * x_pilot_R
        x_in_scaled = user_scale_fact * self.tanh_quant(x_in, beta=beta)
        return self.loss(x_in_scaled, x_pilot_scaled)
    
    def normalize(self, x_in):
        """
        Normalize the input.

        Args:
            x_in (Tensor): Input data.

        Returns:
            x_in (Tensor): Normalized data.
        """
        K = x_in.shape[0]
        x_in = torch.sqrt(torch.tensor(K, dtype=torch.float32, device=device)) * F.normalize(x_in, p=2, dim=0)
        return x_in
        
    def forward(self, x_R, G_R, snr_lin, scale_fact, tau=5):
        scale_fact = torch.diag_embed(scale_fact)
        scale_mat_1 = torch.cat((scale_fact.real, -1 * scale_fact.imag), dim=1)
        scale_mat_2 = torch.cat((scale_fact.imag, scale_fact.real), dim=1)
        scale_mat = torch.cat((scale_mat_1, scale_mat_2), dim=0)
        
        x_in = x_R.T
        x_in = x_in[:, :, None]
        G_R_T = torch.transpose(G_R, 1, 2)
        del_x_obmnet = self.sigmoid(torch.matmul(-G_R, x_in))
        del_x_obmnet = torch.matmul(G_R_T, del_x_obmnet)
        x_in = x_in + self.alpha * (scale_mat @ del_x_obmnet)
        x_obmnet = x_in.squeeze()

        # Update gradient
        grad = del_x_obmnet.squeeze()
        grad = grad.T
        grad = F.normalize(grad, p=2, dim=0)

        in_chan = self.func_to_CNN_chan(x_R, grad, x_obmnet.T)
        x_cnn = self.bn(F.relu(self.conv1(in_chan)))
        x_cnn = x_cnn.view([self.batch_size, self.cnn_chan * self.in_dim])

        del_x = F.relu(self.bn1((self.fc1(x_cnn))))
        del_x = F.relu(self.bn2((self.fc2(del_x))))
        del_x = F.relu(self.bn3((self.fc3(del_x))))
        del_x = self.fc4(del_x)

        x_R = x_obmnet.T + del_x.T
        return x_R

# Loss scaling function
def func_create_user_loss_scaling():
    """
    Create the kappa scaling coefficient for the HieDet training.

    Returns:
        scale_fact (Tensor): User-dependent loss scaling coefficient.
    """
    y1 = torch.tensor([10, 1, 0.5, 0.1, 0], dtype=torch.float32, device=device)
    y2 = torch.arange(4, dtype=torch.float32, device=device)
    y3 = -y1[:, None] * y2
    scale_fact = torch.exp(y3)
    scale_fact = torch.cat((scale_fact, scale_fact), dim=1)
    return scale_fact

user_scale_fact = func_create_user_loss_scaling()






# Section 3: Training loop

# Initialize the network instances
net_one_bit_1 = Net_One_Bit()
net_one_bit_1.to(device)
net_one_bit_2 = Net_One_Bit()
net_one_bit_2.to(device)
net_one_bit_3 = Net_One_Bit()
net_one_bit_3.to(device)
net_one_bit_4 = Net_One_Bit()
net_one_bit_4.to(device)
net_one_bit_5 = Net_One_Bit()
net_one_bit_5.to(device)

# Combine parameters for optimization
params = list(net_one_bit_1.parameters()) + list(net_one_bit_2.parameters()) + list(
    net_one_bit_3.parameters()) + list(net_one_bit_4.parameters()) + list(net_one_bit_5.parameters())
optimizer = torch.optim.Adam(params, lr=gamma, weight_decay=1)
epoch_start = -1

# Create a dictionary to save the model and optimizer states
exp_state = {
    'Net_1': net_one_bit_1.state_dict(),
    'Net_2': net_one_bit_2.state_dict(),
    'Net_3': net_one_bit_3.state_dict(),
    'Net_4': net_one_bit_4.state_dict(),
    'Net_5': net_one_bit_5.state_dict(),
    'Optimizer': optimizer.state_dict(),
    'epochs': -1
}

# Load a previous checkpoint if available
if flag_load_prev == 1:
    exp_cp = torch.load(checkpoint_file, map_location=torch.device(device))
    net_one_bit_1.load_state_dict(exp_cp['Net_1'])
    net_one_bit_2.load_state_dict(exp_cp['Net_2'])
    net_one_bit_3.load_state_dict(exp_cp['Net_3'])
    net_one_bit_4.load_state_dict(exp_cp['Net_4'])
    net_one_bit_5.load_state_dict(exp_cp['Net_5'])
    optimizer.load_state_dict(exp_cp['Optimizer'])
    epoch_start = exp_cp['epochs']

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Training loop
for epoch_ind in range(epoch_start + 1, num_epochs):
    print('Epoch: ', epoch_ind)

    batch_order = np.random.permutation(num_batches)
    loss_epoch = 0.0
    loss_tanh_epoch = 0.0

    snr_lin_batches = np.random.choice(snr_lin_arr, num_batches)
    snr_db_batches = 10 * np.log10(snr_lin_batches)

    for batch_ind in np.arange(num_batches):
        # Measurements
        snr_lin = snr_lin_batches[batch_ind]
        [H, path_loss] = func_create_mmwave_chan_3(phi_min, phi_max, K, N)
        scale_fact = 1 / np.abs(path_loss)
        scale_fact = torch.tensor(scale_fact, dtype=torch.complex64, device=device)
        in_dict = func_create_measurements(H, num_snapshots, snr_lin)
        x_pilot = in_dict['x_pilot']
        x_pilot_R = torch.tensor(np.block([[x_pilot.real], [x_pilot.imag]]), dtype=torch.float32, device=device)
        x_zf = in_dict['x_zf']
        x_R = torch.tensor(np.block([[x_zf.real], [x_zf.imag]]), dtype=torch.float32, device=device)
        H_R = torch.tensor(in_dict['H_R'], dtype=torch.float32, device=device)
        y_R = torch.tensor(in_dict['y_R'], dtype=torch.float32, device=device)
        G_R = torch.matmul(torch.diag_embed(y_R.T), H_R)

        # Forward propagate through the network
        out_net = net_one_bit_1(0 * x_R, G_R, snr_lin, scale_fact)
        loss_1 = net_one_bit_1.loss_eval(out_net.T, x_pilot_R.T, lam=lam, beta=beta, user_scale_fact=user_scale_fact[0, :])

        out_net = net_one_bit_2(out_net, G_R, snr_lin, scale_fact)
        loss_2 = net_one_bit_2.loss_eval(out_net.T, x_pilot_R.T, lam=lam, beta=beta, user_scale_fact=user_scale_fact[1, :])

        out_net = net_one_bit_3(out_net, G_R, snr_lin, scale_fact)
        loss_3 = net_one_bit_3.loss_eval(out_net.T, x_pilot_R.T, lam=lam, beta=beta, user_scale_fact=user_scale_fact[2, :])

        out_net = net_one_bit_4(out_net, G_R, snr_lin, scale_fact)
        loss_4 = net_one_bit_4.loss_eval(out_net.T, x_pilot_R.T, lam=lam, beta=beta, user_scale_fact=user_scale_fact[3, :])

        out_net = net_one_bit_5(out_net, G_R, snr_lin, scale_fact)
        out_net = net_one_bit_5.normalize(out_net)
        loss_5 = net_one_bit_5.loss_eval(out_net.T, x_pilot_R.T, lam=lam, beta=beta, user_scale_fact=user_scale_fact[4, :])

        # Calculate loss
        loss_train_batch = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        loss_tanh_epoch += net_one_bit_5.tanh_loss(out_net.T, x_pilot_R.T, beta=10).item()

        # Backpropagation
        loss_train_batch.backward()

        # Optimizer step
        optimizer.step()

        # Aggregate loss
        loss_epoch += loss_5.item()
        if batch_ind % 200 == 0:
            x_R = out_net
            loss_tanh = net_one_bit_5.tanh_loss(out_net.T, x_pilot_R.T, beta=10).item()
            print('\t Batch ', batch_ind, ' SNR: ', int(snr_db_batches[batch_ind]),
                  ' loss: ', loss_5.item(), ' tanh loss: ', loss_tanh)

    print(loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item())
    print('Epoch ', epoch_ind, ' loss: ', loss_epoch, ' tanh loss: ', loss_tanh_epoch)

    # Save values
    exp_state['Net_1'] = net_one_bit_1.state_dict()
    exp_state['Net_2'] = net_one_bit_2.state_dict()
    exp_state['Net_3'] = net_one_bit_3.state_dict()
    exp_state['Net_4'] = net_one_bit_4.state_dict()
    exp_state['Net_5'] = net_one_bit_5.state_dict()
    exp_state['Optimizer'] = optimizer.state_dict()
    exp_state['epochs'] = epoch_ind
    torch.save(exp_state, checkpoint_file)

    print('\n\n')
