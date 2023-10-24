import numpy as np
import torch

def func_create_mmwave_chan_clustered(phi_min, phi_max, K, N):
    """
    Function to create the mmwave channel matrix.

    Args:
        phi_min (float): Minimum angle of arrival.
            Specifies the minimum angle of arrival for user signals.
        phi_max (float): Maximum angle of arrival.
            Specifies the maximum angle of arrival for user signals.
        K (int): Number of users.
            The total number of users in the system.
        N (int): Number of antennas.
            The total number of antennas in the system.

    Returns:
        H (numpy.ndarray): Complex channel matrix.
            The generated mmWave channel matrix with shape (N, K).
    """
    thresh_lower = np.sqrt(0.5)
    thresh_upper = np.sqrt(1)
    theta_min = np.pi * np.sin(phi_min)
    theta_max = np.pi * np.sin(phi_max)
    theta_vals = np.random.choice(np.linspace(theta_min, theta_max, 20), K, replace=False)
    N_array = np.arange(N)
    A = np.exp(1j * np.multiply(N_array[:, None], theta_vals))
    radius_rand = np.random.uniform(low=thresh_lower, high=thresh_upper, size=K)
    angle_rand = np.random.uniform(low=0, high=2 * np.pi, size=K)
    alpha = radius_rand * np.exp(1j * angle_rand)
    
    H = A @ np.diag(alpha)
    H = np.array(H, dtype=np.complex64)
    
    return H

def func_create_mmwave_chan_clustered_sorted(phi_min, phi_max, K, N):
    """
    Function to create the mmwave channel matrix.

    Args:
        phi_min (float): Minimum angle of arrival.
            Specifies the minimum angle of arrival for user signals.
        phi_max (float): Maximum angle of arrival.
            Specifies the maximum angle of arrival for user signals.
        K (int): Number of users.
            The total number of users in the system.
        N (int): Number of antennas.
            The total number of antennas in the system.

    Returns:
        H (numpy.ndarray): Complex channel matrix.
            The generated mmWave channel matrix with shape (N, K).
        alpha (numpy.ndarray): Array of complex path gains.
            The complex path gains corresponding to each user.
    """
    thresh_lower = np.sqrt(0.5)
    thresh_upper = np.sqrt(1)
    theta_min = np.pi * np.sin(phi_min)
    theta_max = np.pi * np.sin(phi_max)
    theta_vals = np.random.choice(np.linspace(theta_min, theta_max, 20), K, replace=False)
    N_array = np.arange(N)
    A = np.exp(1j * np.multiply(N_array[:, None], theta_vals))
    radius_rand = np.random.uniform(low=thresh_lower, high=thresh_upper, size=K)
    angle_rand = np.random.uniform(low=0, high=2 * np.pi, size=K)
    alpha = radius_rand * np.exp(1j * angle_rand)
    
    alpha = np.array(sorted(alpha, key=abs, reverse=True))
    
    H = A @ np.diag(alpha)
    H = np.array(H, dtype=np.complex64)
    
    return [H, alpha]

def func_create_mmwave_chan_clustered_4_users_sorted(phi_min, phi_max, K, N):
    """
    Function to create clustered mmWave channel for 4 users.

    Args:
        phi_min (float): Minimum angle of arrival.
            Specifies the minimum angle of arrival for user signals.
        phi_max (float): Maximum angle of arrival.
            Specifies the maximum angle of arrival for user signals.
        K (int): Number of users.
            The total number of users in the system (4 users in this case).
        N (int): Number of antennas.
            The total number of antennas in the system.

    Returns:
        H (numpy.ndarray): Complex channel matrix.
            The generated mmWave channel matrix with shape (N, 4).
    """
    thresh_lower = np.sqrt(0.5)
    thresh_upper = np.sqrt(1)
    theta_min = np.pi * np.sin(phi_min)
    theta_max = np.pi * np.sin(phi_max)
    theta_vals = np.random.choice(np.linspace(theta_min, theta_max, 20), K, replace=False)
    N_array = np.arange(N)
    A = np.exp(1j * np.multiply(N_array[:, None], theta_vals))
    radius_rand = np.random.uniform(low=-0.1, high=0.1, size=K)
    radius_rand = radius_rand + 4 - np.arange(K)
    angle_rand = np.random.uniform(low=0, high=2 * np.pi, size=K)
    alpha = radius_rand * np.exp(1j * angle_rand)
    
    H = A @ np.diag(alpha)
    H = np.array(H, dtype=np.complex64)
    
    return H

def func_create_mmwave_chan(phi_min, phi_max, K, N):
    """
    Function to create the mmwave channel matrix.

    Args:
        phi_min (float): Minimum angle of arrival.
            Specifies the minimum angle of arrival for user signals.
        phi_max (float): Maximum angle of arrival.
            Specifies the maximum angle of arrival for user signals.
        K (int): Number of users.
            The total number of users in the system.
        N (int): Number of antennas.
            The total number of antennas in the system.

    Returns:
        H (numpy.ndarray): Complex channel matrix.
            The generated mmWave channel matrix with shape (N, K).
    """
    thresh_lower = np.sqrt(0.5)
    thresh_upper = np.sqrt(1)
    theta_min = np.pi * np.sin(phi_min)
    theta_max = np.pi * np.sin(phi_max)
    theta_vals = np.random.choice(np.linspace(theta_min, theta_max, 20), K, replace=False)
    N_array = np.arange(N)
    A = np.exp(1j * np.multiply(N_array[:, None], theta_vals))
    radius_rand = np.random.uniform(low=thresh_lower, high=thresh_upper, size=K)
    angle_rand = np.random.uniform(low=0, high=2 * np.pi, size=K)
    alpha = radius_rand * np.exp(1j * angle_rand)
    
    H = A @ np.diag(alpha)
    H = np.array(H, dtype=np.complex64)
    
    return H

def func_create_mmwave_chan_2(phi_min, phi_max, K, N):
    """
    Function to create the mmwave channel matrix and return path loss array as well.

    Args:
        phi_min (float): Minimum angle of arrival.
            Specifies the minimum angle of arrival for user signals.
        phi_max (float): Maximum angle of arrival.
            Specifies the maximum angle of arrival for user signals.
        K (int): Number of users.
            The total number of users in the system.
        N (int): Number of antennas.
            The total number of antennas in the system.

    Returns:
        H (numpy.ndarray): Complex channel matrix.
            The generated mmWave channel matrix with shape (N, K).
        alpha (numpy.ndarray): Array of complex path gains.
            The complex path gains corresponding to each user.
    """
    thresh_lower = np.sqrt(0.5)
    thresh_upper = np.sqrt(1)
    theta_min = np.pi * np.sin(phi_min)
    theta_max = np.pi * np.sin(phi_max)
    theta_vals = np.random.choice(np.linspace(theta_min, theta_max, 20), K, replace=False)
    N_array = np.arange(N)
    A = np.exp(1j * np.multiply(N_array[:, None], theta_vals))
    radius_rand = np.random.uniform(low=thresh_lower, high=thresh_upper, size=K)
    angle_rand = np.random.uniform(low=0, high=2 * np.pi, size=K)
    alpha = radius_rand * np.exp(1j * angle_rand)
    
    H = A @ np.diag(alpha)
    H = np.array(H, dtype=np.complex64)
    
    return [H, alpha]

def func_create_mmwave_chan_3(phi_min, phi_max, K, N):
    """
    Function to create the mmwave channel matrix and return path loss array as well.
    Sort path gains in descending order.

    Args:
        phi_min (float): Minimum angle of arrival.
            Specifies the minimum angle of arrival for user signals.
        phi_max (float): Maximum angle of arrival.
            Specifies the maximum angle of arrival for user signals.
        K (int): Number of users.
            The total number of users in the system.
        N (int): Number of antennas.
            The total number of antennas in the system.

    Returns:
        H (numpy.ndarray): Complex channel matrix.
            The generated mmWave channel matrix with shape (N, K).
        alpha (numpy.ndarray): Array of complex path gains.
            The complex path gains corresponding to each user, sorted in descending order.
    """
    thresh_lower = np.sqrt(0.5)
    thresh_upper = np.sqrt(1)
    theta_min = np.pi * np.sin(phi_min)
    theta_max = np.pi * np.sin(phi_max)
    theta_vals = np.random.choice(np.linspace(theta_min, theta_max, 20), K, replace=False)
    N_array = np.arange(N)
    A = np.exp(1j * np.multiply(N_array[:, None], theta_vals))
    radius_rand = np.random.uniform(low=thresh_lower, high=thresh_upper, size=K)
    angle_rand = np.random.uniform(low=0, high=2 * np.pi, size=K)
    alpha = radius_rand * np.exp(1j * angle_rand)
    
    alpha = np.array(sorted(alpha, key=abs, reverse=True))
    
    H = A @ np.diag(alpha)
    H = np.array(H, dtype=np.complex64)
    
    return [H, alpha]

def func_create_measurements(H, num_snapshots, snr_lin, mod='qpsk'):
    """
    Function to create the noisy measurements and the one-bit received signal from the channel matrix.

    Args:
        H (numpy.ndarray): Complex channel matrix.
            The mmWave channel matrix with shape (N, K).
        num_snapshots (int): Number of snapshots.
            The total number of snapshots.
        snr_lin (float): Linear SNR value.
            The signal-to-noise ratio in linear scale.
        mod (str, optional): Modulation type (default is 'qpsk').
            The modulation type used for creating the pilot data.

    Returns:
        out_dict (dict): Dictionary of numpy arrays.
            A dictionary containing various arrays, including pilot data, received signals, noisy signals, and more.
    """
    N = H.shape[0]
    K = H.shape[1]

    H = np.array(H,dtype=np.complex64)
    
    if(mod=='qpsk'):
        x_pilot = np.random.choice([-1,1],[K,num_snapshots]) + 1j*np.random.choice([-1,1],[K,num_snapshots])
        
    if(mod=='16qam'):
        x_pilot = np.random.choice([-3,-1,1,3],[K,num_snapshots]) + 1j*np.random.choice([-3,-1,1,3],[K,num_snapshots])
        
    x_pilot = np.array(x_pilot,dtype=np.complex64)
    # x_pilot = np.random.choice([-3,-1,1,3],[K,num_snapshots]) + 1j*np.random.choice([-3,-1,1,3],[K,num_snapshots])

    # Received unquantized noisy signal
    r_noiseless = H@x_pilot

    noise_vec = np.random.randn(N,num_snapshots) + 1j*np.random.randn(N,num_snapshots)
    noise_vec = noise_vec / np.linalg.norm(noise_vec)  * np.sqrt(np.linalg.norm(r_noiseless)**2 / snr_lin)
    noise_vec = np.array(noise_vec,dtype=np.complex64)

    r = r_noiseless + noise_vec
    
    # One-bit data
    y = ( np.sign(r.real) + 1j*np.sign(r.imag) )
    y = np.array(y,dtype=np.complex64)

    # Real-data
    H_R = np.block([[H.real, -1*H.imag],[H.imag, H.real]])
    y_R = np.block([[y.real],[y.imag]])
    
    # Initial estimate
    H_pinv = np.linalg.pinv(H)
    x_zf = H_pinv @ y
    
    x_est = np.random.randn(K,num_snapshots) + 1j*np.random.randn(K,num_snapshots)

    x_zf = x_zf / np.mean(np.abs(x_zf)) * np.sqrt(2)
    
    out_dict = {}
    out_dict.update({'x_pilot':x_pilot})
    out_dict.update({'r_noiseless':r_noiseless})
    out_dict.update({'r':r})
    out_dict.update({'y':y})
    out_dict.update({'y_R':y_R})
    out_dict.update({'H_R':H_R})
    out_dict.update({'H_pinv':H_pinv})
    out_dict.update({'x_est':x_est})
    out_dict.update({'x_zf':x_zf})
    
    return out_dict
    
    return out_dict

def func_create_G_R(H, y):
    """
    Function to create the matrix G_R with each slice as the unique G_R matrix for each snapshot.

    Args:
        H (numpy.ndarray): 2-D channel matrix.
            The complex-valued channel matrix with shape (N, K).
        y (numpy.ndarray): 2-D matrix of one-bit measurements per column.
            The one-bit measurements represented as a complex matrix.

    Returns:
        G_R (numpy.ndarray): 3-D numpy array.
            A 3-D numpy array with each slice as a unique G_R matrix.
    """
    G_R = H @ np.linalg.pinv(H.T @ H) @ H.T @ y
    
    return G_R

def func_multiply_parallel(A, x):
    """
    Function to efficiently multiply each slice of the 3-D matrix A separately with each column of x.

    Args:
        A (numpy.ndarray): Input 3-D matrix with each slice as a unique snapshot.
        x (numpy.ndarray): Input 2-D matrix with each column as a unique snapshot.

    Returns:
        y_out (numpy.ndarray): 2-D numpy array.
            A 2-D numpy array with each column as the result of multiplication.
    """
    y_out = np.einsum('ijk,jk->ik', A, x)
    
    return y_out

def func_grad(rho, N_ant, G_R, x_temp):
    """
    Function to evaluate the gradient for a known random channel matrix.

    Args:
        rho (float): Linear SNR value.
            The signal-to-noise ratio in linear scale.
        N_ant (int): Number of antennas.
            The total number of antennas.
        G_R (numpy.ndarray): Matrix containing the one-bit and channel information.
            A 3-D numpy array with each slice as a unique G_R matrix.
        x_temp (numpy.ndarray): Current estimate of symbol estimate.
            A 2-D numpy array representing the current symbol estimate.
    """
    P_k = rho
    a = torch.tensor(G_R)[:, :, 0] * torch.tensor(x_temp).view(N_ant, -1)
    x_term = torch.mean(torch.relu(1 - P_k * a))
    grad_term = P_k * a
    grad_term[a * P_k > 1] = 0
    grad_term = -torch.mean(grad_term, 1)
    grad = x_term * grad_term
    
    return grad

def func_to_CNN_chan(x_in, grad_in):
    """
    Function to convert the input data to the form accepted by the CNN channels.

    Args:
        x_in (Tensor): Input data with concatenated real and imaginary parts.
        grad_in (Tensor): Input gradient with concatenated real and imaginary parts.

    Returns:
        out_chan (Tensor): 3-D torch tensor.
    """
    K = int(x_in.shape[0] / 2)
    out_chan = torch.stack([x_in[:K,:],x_in[K:,:],grad_in[:K,:],grad_in[K:,:]])
    out_chan = torch.moveaxis(out_chan, [0, 1, 2], [1, 2, 0])
    return out_chan

def func_to_CNN_chan_three_input(x_in, grad_in, x_obmnet):
    """
    Function converts the input data to the form accepted by the CNN channels.

    Args:
        x_in (Tensor): Input data with concatenated real and imaginary parts.
        grad_in (Tensor): Input gradient with concatenated real and imaginary parts.
        x_obmnet (Tensor): obmnet output with concatenated real and imaginary parts.

    Returns:
        out_chan (Tensor): 3-D torch tensor.
    """
    K = int(x_in.shape[0]/2)
    
    out_chan = torch.stack([x_in[:K,:],x_in[K:,:],grad_in[:K,:],grad_in[K:,:],
                            x_obmnet[:K,:],x_obmnet[K:,:]])
    out_chan = torch.moveaxis(out_chan,[0,1,2],[1,2,0])
    
    return out_chan

def func_from_CNN_chan(in_chan):
    """
    Function converts the output data of the CNN to other types for gradient evaluation.

    Args:
        in_chan (Tensor): CNN output.

    Returns:
        x_out (Tensor): 3-D torch tensor.
    """
    x_real = in_chan[:, 0, :].T
    x_imag = in_chan[:, 1, :].T
    x_out = torch.cat([x_real, x_imag], dim=0)
    return x_out

def func_tensor_grad_LR(rho, N_ant, G_R, x_temp, func_sigmoid):
    """
    Function to evaluate the gradient for a known random channel matrix using logistic regression.

    Args:
        rho (float): Linear SNR value.
        N_ant (int): Number of antennas.
        G_R (Tensor): Matrix containing the one-bit and channel information.
        x_temp (Tensor): Current estimate of symbol estimate.
        func_sigmoid (function): Function class with logistic sigmoid.

    Returns:
        list: A list with nan_flag and gradient.
    """
    z = -1.702 * torch.sqrt(torch.tensor(2 * rho)) * func_multiply_parallel(G_R, x_temp)
    nan_flag = 0
    G_R_T = torch.moveaxis(G_R, [0, 1, 2], [0, 2, 1])
    sig_val = func_sigmoid(z)
    return [nan_flag, -1.702 * np.sqrt(2 * rho) * func_multiply_parallel(G_R_T, sig_val)]

def func_16qam_demapper(symbols_I, symbols_Q, packetSize, threshold=2.0):
    """
    Function to evaluate the 16-QAM demodulation.

    Args:
        symbols_I (numpy.ndarray): In-phase symbols.
        symbols_Q (numpy.ndarray): Quadrature symbols.
        packetSize (int): Number of bits.
        threshold (float): Threshold value for demodulation.

    Returns:
        numpy.ndarray: Bit stream.
    """
    Ns = int(packetSize / 4)
    bits_I = []
    bits_Q = []
    for i in range(Ns):
        if 0 <= symbols_I[i] <= threshold:
            bits_I.append(1)
            bits_I.append(0)
        if threshold < symbols_I[i]:
            bits_I.append(1)
            bits_I.append(1)
        if -threshold <= symbols_I[i] < 0:
            bits_I.append(0)
            bits_I.append(1)
        if symbols_I[i] < -threshold:
            bits_I.append(0)
            bits_I.append(0)

        if 0 <= symbols_Q[i] <= threshold:
            bits_Q.append(1)
            bits_Q.append(0)
        if threshold < symbols_Q[i]:
            bits_Q.append(1)
            bits_Q.append(1)
        if -threshold <= symbols_Q[i] < 0:
            bits_Q.append(0)
            bits_Q.append(1)
        if symbols_Q[i] < -threshold:
            bits_Q.append(0)
            bits_Q.append(0)

    bits_I = list(map(int, bits_I))
    bits_Q = list(map(int, bits_Q))

    bitStream = np.zeros(packetSize)

    for i in range(len(bits_I)):
        bitStream[2 * i] = bits_I[i]
        bitStream[2 * i - 1] = bits_Q[i - 1]

    return bitStream