import numpy as np
from zernike import RZern
from scipy.sparse import coo_matrix, vstack, diags
from scipy.sparse.linalg import lsqr
from scipy.linalg import solve, lstsq
from skimage.restoration import unwrap_phase
import time
import logging

logger = logging.getLogger(__name__)

def check_full_column_rank(A):
    rank = np.linalg.matrix_rank(A)
    
    # Check if the computed rank equals the number of columns in A
    return rank == A.shape[1]


def build_row_map(K):
    # Build once and reuse outside
    return {tuple(row): i for i, row in enumerate(K)}

def form_diff_and_offset_operator(K, idx_overlap, ki, kl, row_map):
    #
    # 1) Build D in coordinate form
    #
    # We'll do D(i1, row_map[K[idx_overlap[i1]] - ki]) = +1 and
    #              D(i1, row_map[K[idx_overlap[i1]] - kl]) = -1
    # if they exist in row_map.
    #
    rows_D = []
    cols_D = []
    vals_D = []
    
    # We'll need these a lot:
    K_sub = K[idx_overlap]          # shape: (len(idx_overlap), dimension)
    
    for i1, k_orig in enumerate(K_sub):
        # differences
        diff_ki = tuple(k_orig - ki)
        diff_kl = tuple(k_orig - kl)
        
        c_ki = row_map.get(diff_ki, None)
        if c_ki is not None:
            rows_D.append(i1)
            cols_D.append(c_ki)
            vals_D.append(+1)
        
        c_kl = row_map.get(diff_kl, None)
        if c_kl is not None:
            rows_D.append(i1)
            cols_D.append(c_kl)
            vals_D.append(-1)
    
    D = coo_matrix((vals_D, (rows_D, cols_D)), shape=(len(idx_overlap), len(K))).tocsr()
    
    #
    # 2) Build D0 in coordinate form
    #    It's the same offset for every row i1 if row_map has ki, kl
    #
    rows_D0 = []
    cols_D0 = []
    vals_D0 = []
    
    c_ki = row_map.get(tuple(ki), None)
    c_kl = row_map.get(tuple(kl), None)
    
    if c_ki is not None:
        rows_D0.extend(range(len(idx_overlap)))
        cols_D0.extend([c_ki]*len(idx_overlap))
        vals_D0.extend([+1]*len(idx_overlap))
    
    if c_kl is not None:
        rows_D0.extend(range(len(idx_overlap)))
        cols_D0.extend([c_kl]*len(idx_overlap))
        vals_D0.extend([-1]*len(idx_overlap))
    
    D0 = coo_matrix((vals_D0, (rows_D0, cols_D0)), shape=(len(idx_overlap), len(K))).tocsr()
    
    return D, D0



def form_Zernike_operator(zernikeModes, size):
    """
    Calculates the Zernike modes at given x and y coordinates.
    Input:
        zernikeModes: Zernike modes 
        x: x coordinates vector
        y: y coordinates vector
    Output:
        Zernike matrix with len(zernikeModes) columns and len(x) rows
    """
    temp = np.linspace(-1, 1, size)  # Example: temp is a linear space across the grid
    
    # 2. Create the coordinate grid (Xz, Yz)
    Yz, Xz = np.meshgrid(temp, temp)

    # 3. Convert Cartesian coordinates to polar coordinates (r, theta)
    theta = np.arctan2(Yz, Xz)
    r = np.sqrt(Xz**2 + Yz**2)

    # 4. Define the mask for points inside the unit circle (r <= 1)
    idx2use = (r <= 1)

    nZernike = len(zernikeModes)  # Number of Zernike modes
    Hz = np.zeros((size**2, nZernike))  # Initialize Hz matrix
    rzern = RZern(9)  

    # zernikeTemp will hold the Zernike values for all modes
    zernikeTemp = np.zeros((np.sum(idx2use), nZernike))

    # Loop over the Zernike modes
    for i, p in enumerate(zernikeModes):
        Z = rzern.Zk(p, r[idx2use], theta[idx2use])  # Compute the Zernike polynomial for mode idx
        zernikeTemp[:, i] = Z
    Hz[idx2use.flatten(), :] = zernikeTemp

    return Hz



def get_ctf(recFTframes, shifts, CTF_radius=None, CTF=None, useWeights=False, useZernike=True):
    size_x = recFTframes[0].shape[0]
    size_y = recFTframes[0].shape[1]
    u = np.linspace(-size_x/2, size_x/2, size_x).astype(int)
    v = np.linspace(-size_y/2, size_y/2, size_y).astype(int)
    U, V = np.meshgrid(u, v)
    R = np.sqrt((U)**2 + (V)**2)


    # sort shifts and frames by angle
    phis = [np.arctan2(k[1], k[0]) for k in shifts]
    shifts = [k for _, k in sorted(zip(phis, shifts))]
    recFTframes = [k for _, k in sorted(zip(phis, recFTframes))]

    # Test if CTF_radius or CTF is None if both are None raise an error
    if CTF_radius is None and CTF is None:
        raise ValueError("CTF_radius or CTF must be provided")
    
    
    # Create the CTF matrix
    if CTF is None:
        CTF = np.zeros((size_x, size_y))
        CTF[R < CTF_radius] = 1


    CTFs = [np.roll(CTF, k, axis=(1, 0)) for k in shifts]
    idx_overlap_l = [np.nonzero(CTFs[i].flatten() * CTFs[(i+1) % len(CTFs)].flatten())[0] for i in range(len(CTFs))]

    # Unwrap the phase differences
    logger.info("Unwrapping phase differences")
    phase_diffs = []
    weights = []
    for i in range(len(recFTframes)):
        # find the max x and max y coordinate where CTF[i] is 1
        recFTframes_it = [recFTframes[i], recFTframes[(i+1) % len(recFTframes)]]
        CTFs_it = [CTFs[i], CTFs[(i+1) % len(CTFs)]]
        phaseTemp = unwrap_phase_differences(recFTframes_it, size_x, size_y, CTFs_it, i)

        if useWeights:
            weight= np.log10(np.abs(recFTframes_it[0] * recFTframes_it[1]) + 1)
            phaseTemp *= weight
            weights.append(diags(weight.flatten()[idx_overlap_l[i]])) 

        phase_diffs.append(phaseTemp.flatten()[idx_overlap_l[i]])
            

    phase_diffs = np.concatenate(phase_diffs)

    # Form Operators
    K = np.array(list(zip(U.flatten(), V.flatten())))
    idx_ctf = np.nonzero(CTF.flatten())[0]

    logger.info("Forming operators")
    start_time = time.time()
    Ds = []
    row_map = build_row_map(K)
    for i in range(len(shifts)):
        ki = shifts[i]
        kl = shifts[(i+1) % len(shifts)]
        idx_overlap = idx_overlap_l[i]
        A, O = form_diff_and_offset_operator(K, idx_overlap, ki, kl, row_map)

        A = A[:, idx_ctf] # could be optimized
        O = O[:, idx_ctf] # could be optimized

        if useWeights:
            D = weights[i] @ (A - O)
        else:
            D = A - O
        Ds.append(D)

    logger.info(
        "Time to form DSs and D0s: %.2f seconds",
        time.time() - start_time,
    )

    Ds = vstack(Ds)

    Dz = form_Zernike_operator(range(3,50), size_x)
    Dz= Dz[idx_ctf, :]


    if useZernike:
        logger.info("Solving for Zernike coefficients")
        # Solve for the Zernike coefficients
        Ds = Ds@Dz

        # check if Ds is full column rank

        # Solve for the Zernike coefficients
        zernike_solved = solve((Ds.T@Ds) + np.eye(Ds.shape[1]), Ds.T@phase_diffs)             # Regularized least squares scipy.linalg.solve (10 calls = 11.61)
        # zernike_solved = np.linalg.solve((Ds.T@Ds) + np.eye(Ds.shape[1]), Ds.T@phase_diffs)     # Regularized least squares numpy.linalg.solve (10 calls = 11.27)
        # zernike_solved = solve(Ds.T@Ds, Ds.T@phase_diffs)                                       # least squares scipy.linalg.solve (10 calls = 11.11)
        # zernike_solved = np.linalg.solve(Ds.T@Ds, Ds.T@phase_diffs)                              # least squares numpy.linalg.solve (10 calls = 11.21)

        # zernike_solved = lstsq(Ds, phase_diffs)[0]                                              # least squares scipy.linalg.lstsq (10 calls = 11.49)
        # zernike_solved = np.linalg.lstsq(Ds, phase_diffs, rcond=None)[0]                        # least squares numpy.linalg.lstsq (10 calls = 11.75)
        # zernike_solved = lsqr(Ds, phase_diffs)[0]                                                # least squares scipy.sparse.linalg.lsqr (10 calls = 12.65)
        
        logger.info("Zernike coefficients solved")

        n_pupil_px = np.sum(CTF)
        CTF_abe = (Dz@zernike_solved)
        # 
        CTF_abe_full = np.zeros((size_x*size_y)) 
        CTF_abe_full[idx_ctf] = CTF_abe
        CTF_abe_full = CTF_abe_full.reshape((size_x, size_y))
        CTF_abe_full -= (np.sum(CTF_abe_full) / n_pupil_px) # remove global phase
        CTF_abe_full *= CTF
    else:
        # solve without the Zernike coefficients
        # CTF_abe = solve((Ds.T@Ds + np.eye(Ds.shape[1])), Ds.T@phase_diffs)
        CTF_abe = lsqr(Ds, phase_diffs)[0]
        
        CTF_abe_full = np.zeros((size_x*size_y))
        CTF_abe_full[idx_ctf] = CTF_abe
        CTF_abe_full = CTF_abe_full.reshape((size_x, size_y))
        CTF_abe_full -= (np.sum(CTF_abe_full) / np.sum(CTF)) # remove global phase
        CTF_abe_full *= CTF

    return CTF * np.exp(1j * CTF_abe_full)

def unwrap_phase_differences(recFTframes, size_x, size_y, CTFs, i):
    overlapCTF = CTFs[0] * CTFs[1]
    idx_x1, idx_y1 = np.where(overlapCTF == 1)
    xbd_max = idx_x1.max() + 5
    ybd_max = idx_y1.max() + 5
    xbd_min = idx_x1.min() - 5
    ybd_min = idx_y1.min() - 5


    phaseTemp = recFTframes[0] * CTFs[0] * np.conj(recFTframes[1] * CTFs[1])
        # tempWeights = np.log10(np.abs(phaseTemp[xbd_min:xbd_max,ybd_min:ybd_max]) + 1)
        # phaseUnwrapTemp = phase_unwrap_cg(np.angle(phaseTemp)[xbd_min:xbd_max,ybd_min:ybd_max], tempWeights/np.max(tempWeights))
    phaseUnwrapTemp = unwrap_phase(np.angle(phaseTemp[xbd_min:xbd_max,ybd_min:ybd_max]))
    phaseRaw = np.angle(phaseTemp[xbd_min:xbd_max,ybd_min:ybd_max])
        # phase_diffs[i] = phaseUnwrapTemp
    wrappedPhase = phaseUnwrapTemp - phaseRaw
        
        # Histogram of the phase differences
    N = np.histogram(wrappedPhase, bins=32, range=(-np.pi * 2, np.pi * 2))[0]
    N[:10] = 0
    N[-10:] = 0
    idxPk = np.argmax(N)
    x2use = np.arange(-np.pi * 2, np.pi * 2, np.pi / 8)
    offsetPk = np.mean(wrappedPhase[(wrappedPhase > x2use[idxPk-1]) & (wrappedPhase < x2use[idxPk + 1])])
    phaseTemp = phaseRaw + np.round((wrappedPhase - offsetPk) / (2 * np.pi)) * 2 * np.pi + offsetPk
        
        # pad phaseTemp with zeros to the size of the original frame
    phaseTemp = np.pad(phaseTemp, ((xbd_min, size_x - xbd_max), (ybd_min, size_y - ybd_max)))
    return phaseTemp

