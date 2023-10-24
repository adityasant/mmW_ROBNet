import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from PIL import Image
from create_data_random_chan import func_create_mmwave_chan_3
from create_data_random_chan import func_create_measurements
from create_data_random_chan import func_create_G_R
from create_data_random_chan import func_multiply_parallel
from create_data_random_chan import func_tensor_grad_LR
from create_data_random_chan import func_to_CNN_chan
from create_data_random_chan import func_to_CNN_chan_three_input
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



# Section 1: Parameters
N = 64
K = 4
phi_min = -60*np.pi/180
phi_max = 60*np.pi/180
num_snapshots = 2000
snr_db_arr = np.array([-5,0,5,10,15,20,25,30,35])
snr_lin_arr = 10**(0.1 * snr_db_arr)
num_trials = 40000
checkpoint_file_obirim = 'Saved_Networks/mmw_robnet_matched_user_scale_mmwave_H_regularized/checkpoint.pth.tar'






# Section 2: Function and DNN class definition
def func_create_batch(H, snr_lin, num_snapshots=32, mod='qpsk'):
    """
    Create a batch of data for a communication system given channel matrix H and SNR.

    Parameters:
    - H: ndarray, shape (N, K)
        The channel matrix representing the communication channel.
    - snr_lin: float
        The signal-to-noise ratio (SNR) in linear scale.
    - num_snapshots: int, optional (default=32)
        The number of snapshots in the batch.
    - mod: str, optional (default='qpsk')
        The modulation scheme for pilot symbols. Supported values are 'qpsk' and '16qam'.

    Returns:
    - batch: list [x_pilot_R, x_R, G_R]
        A list containing three tensors:
        1. x_pilot_R: Tensor, shape (2 * K, num_snapshots), dtype float32
            The pilot symbols in real and imaginary parts.
        2. x_R: Tensor, shape (num_snapshots, 2 * K, 1), dtype float32
            The received symbols in real and imaginary parts.
        3. G_R: Tensor, shape (num_snapshots, 2 * N, 2 * K), dtype float32
            The received signal matrix.

    This function generates a batch of communication data by taking a channel matrix H, adding noise based on SNR,
    and creating pilot symbols for communication. The batch can be used for training and testing communication systems.
    """
    K = H.shape[1]
    H_R = torch.tensor(np.block([[H.real, -1*H.imag],[H.imag, H.real]]), dtype=torch.float32, device=device)
    
    if(mod=='qpsk'):
        x_pilot = np.random.choice([-1,1],[K,num_snapshots]) + 1j*np.random.choice([-1,1],[K,num_snapshots])
        
    if(mod=='16qam'):
        x_pilot = np.random.choice([-3,-1,1,3],[K,num_snapshots]) + 1j*np.random.choice([-3,-1,1,3],[K,num_snapshots])
    
    x_pilot = np.array(x_pilot, dtype=np.complex64)
    x_pilot_R = torch.tensor(np.block([[x_pilot.real],[x_pilot.imag]]), dtype=torch.float32, device=device)
    
    x_R = 1e-3 * torch.ones([num_snapshots, 2*K, 1], dtype=torch.float32, device=device)
    
    r_noiseless = H @ x_pilot

    noise_vec = np.random.randn(N, num_snapshots) + 1j * np.random.randn(N, num_snapshots)
    noise_vec = noise_vec / np.linalg.norm(noise_vec) * np.sqrt(np.linalg.norm(r_noiseless)**2 / snr_lin)
    noise_vec = np.array(noise_vec, dtype=np.complex64)
    r = r_noiseless + noise_vec
    
    y = (np.sign(r.real) + 1j * np.sign(r.imag))
    y = np.array(y, dtype=np.complex64)
    y_R = torch.tensor(np.block([[y.real],[y.imag]]), dtype=torch.float32, device=device)
    
    G_R = torch.matmul(torch.diag_embed(y_R.T), H_R)
    
    return [x_pilot_R, x_R, G_R]



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

net_one_bit_1 = Net_One_Bit(batch_size=num_snapshots)
net_one_bit_1.to(device)
net_one_bit_2 = Net_One_Bit(batch_size=num_snapshots)
net_one_bit_2.to(device)
net_one_bit_3 = Net_One_Bit(batch_size=num_snapshots)
net_one_bit_3.to(device)
net_one_bit_4 = Net_One_Bit(batch_size=num_snapshots)
net_one_bit_4.to(device)
net_one_bit_5 = Net_One_Bit(batch_size=num_snapshots)
net_one_bit_5.to(device)






# Section 3: Testing loop

BER_arr_1 = np.zeros([num_trials, len(snr_db_arr)])
BER_arr_1_user = np.zeros([K, num_trials, len(snr_db_arr)])

for trial_ind in np.arange(num_trials):
    [H, path_loss] = func_create_mmwave_chan_3(phi_min, phi_max, K, N)
    scale_fact = 1 / np.abs(path_loss)
    scale_fact = torch.tensor(scale_fact, dtype=torch.complex64, device=device)
        
    for snr_ind in np.arange(len(snr_db_arr)):
        if(trial_ind % 50 == 0):
            print('Trial: ', trial_ind+1, 'SNR (dB): ', snr_db_arr[snr_ind])
        
        measure_dict = func_create_measurements(H, num_snapshots, snr_lin_arr[snr_ind], mod='qpsk')
        x_pilot = measure_dict['x_pilot']
        x_zf = measure_dict['x_zf']
        y_R = measure_dict['y_R']
        H_R = measure_dict['H_R']
        H_R = torch.tensor(H_R, dtype=torch.float32, device=device)
        y_R = torch.tensor(y_R, dtype=torch.float32, device=device)
        G_R = torch.matmul(torch.diag_embed(y_R.T), H_R)
        x_R_tensor = torch.tensor(np.block([[x_zf.real], [x_zf.imag]]), dtype=torch.float32, device=device)
        
        out_net = net_one_bit_1(0 * x_R_tensor, G_R, snr_lin_arr[snr_ind], scale_fact)
        out_net = net_one_bit_2(out_net, G_R, snr_lin_arr[snr_ind], scale_fact)
        out_net = net_one_bit_3(out_net, G_R, snr_lin_arr[snr_ind], scale_fact)
        out_net = net_one_bit_4(out_net, G_R, snr_lin_arr[snr_ind], scale_fact)
        out_net = net_one_bit_5(out_net, G_R, snr_lin_arr[snr_ind], scale_fact)
        out_net = net_one_bit_5.normalize(out_net)
        out_net = out_net.cpu().detach().numpy()
        x_est_complex_obirim = out_net[:K, :] + 1j * out_net[K:, :]
        
        x_pilot_bits = np.reshape(x_pilot, [K * num_snapshots])
        
        x_est_bits = np.sign(x_est_complex_obirim.real) + 1j * np.sign(x_est_complex_obirim.imag)
        BER_temp = ~np.equal(x_pilot.real, x_est_bits.real) + ~np.equal(x_pilot.imag, x_est_bits.imag)
        BER_arr_1_user[:, trial_ind, snr_ind] = np.sum(BER_temp, axis=1)
        
        x_est_complex_obirim = np.reshape(x_est_complex_obirim, [K * num_snapshots])
        x_est_bits = np.sign(x_est_complex_obirim.real) + 1j * np.sign(x_est_complex_obirim.imag)
        BER_arr_1[trial_ind, snr_ind] = np.sum(~np.equal(x_pilot_bits.real, x_est_bits.real)) + np.sum(~np.equal(x_pilot_bits.imag, x_est_bits.imag))

        


# Save results
ber_arr_mmw_robnet = BER_arr_1_user_avg

np.save('BER_Results/ber_arr_mmw_robnet',ber_arr_mmw_robnet)
