
TIME_STEP = 5 # seconds.

"""Annotate the tension states based on the centromere separation and spindle length."""
import os
import numpy as np
import pandas as pd

from glob import glob
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import PowerTransformer
from hmmlearn import hmm

# Where the per-cell CSVs are read from and where the annotated ones are written.
INPUT_DIRECTORY = 'high_speed_output'
OUTPUT_DIRECTORY = 'high_speed_trajectories'
GMM_SUMMARY_PATH = 'high_speed_trajectories/gmm_components.csv'
COMBINED_TRAJECTORY_PATH = 'high_speed_trajectories/all_trajectories.csv'

# ── Batch definitions ─────────────────────────────────────────────────────
# Each key is a phenotype name and each value is the list of YV strain names
# belonging to that phenotype. Every cell from every listed YV strain is pooled
# together, a single 3 component GMM is fit to that pool, and the fitted
# components become the HMM emission distributions for those cells.
#
# The batch name is the mutant followed by the centromere, for example
# 'Ase1-7A CENIV'. The combined output splits the name back apart into a
# separate mutant column and centromere column.
BATCHES = {
    'WT CENIV': ['YV3640', 'YV3670', 'YV3701', 'YV3702'],
    'Cin8-RK kip1∆ CENIV': ['YV3663', 'YV3673', 'YV3682', 'YV3700'],
    'Ase1-7A CENIV': ['YV3744', 'YV3748', 'YV3841', 'YV3844'],
    'WT CENXV': ['YV4228', 'YV4232'],
    'Cin8-RK kip1∆ CENXV': ['YV4235', 'YV4240'],
    'Ase1-7A CENXV': ['YV4243', 'YV4305'],
}

# The centromere names that may appear at the end of a batch name.
CENTROMERE_NAMES = ['CENIV', 'CENXV']

# 3 hidden states corresponding to tension levels.
# State order: 0 = No Tension, 1 = Low Tension, 2 = High Tension
N_TENSION_STATES = 3

# ── Single tunable parameter ──────────────────────────────────────────────
# Self-transition probability p. Everything else in the transition matrix
# is derived from this one value.
#
# For a self-transition probability p and adjacent transition q, the Viterbi
# algorithm switches states when the emission likelihood ratio R = b_j(x)/b_i(x)
# exceeds p/q for a single observation, or sqrt(p/q) for two consecutive ones.
#
#   1 obs stays  when: R < p/q
#   2 obs switch when: R > sqrt(p/q)
#
# p = 0.70 → p/q ≈ 2.4,  sqrt(p/q) ≈ 1.55
# → A single timepoint up to ~2.4× more likely in another state won't switch,
#   but 2 consecutive observations at ~1.6× or more will switch.
SELF_TRANSITION_P = 0.50

def build_transmat(p, n_states=3, jump_prob=0.1):
    """Build a row-stochastic transition matrix from a single self-transition
    probability *p*.

    Each row gives probability *p* to staying, *jump_prob* to each non-adjacent
    state, and distributes the remainder equally among adjacent neighbours.
    """
    mat = np.full((n_states, n_states), 0.0)
    for i in range(n_states):
        mat[i, i] = p
        # Identify adjacent and non-adjacent states
        adjacent = [j for j in range(n_states) if j != i and abs(j - i) == 1]
        non_adjacent = [j for j in range(n_states) if j != i and abs(j - i) > 1]
        # Assign jump_prob to each non-adjacent state
        for j in non_adjacent:
            mat[i, j] = jump_prob
        # Remainder split equally among adjacent neighbours
        remainder = 1.0 - p - jump_prob * len(non_adjacent)
        q = remainder / len(adjacent) if adjacent else 0.0
        for j in adjacent:
            mat[i, j] = q
    return mat

DEFAULT_TRANSMAT = build_transmat(SELF_TRANSITION_P)
DEFAULT_STARTPROB = np.array([0.6, 0.3, 0.1])  # Initial state probabilities

STATE_LABELS = ['No Tension', 'Low Tension', 'High Tension']

def find_batch_files(yv_strains):
    """Return the paths of every cell CSV belonging to the given YV strains."""
    input_file_paths = []
    for yv_strain in yv_strains:
        search_pattern = os.path.join(INPUT_DIRECTORY, yv_strain, '*.csv')
        input_file_paths.extend(sorted(glob(search_pattern)))
    return input_file_paths

def load_cell(input_file_path):
    """Read one cell's CSV and return it as a cleaned DataFrame.

    The raw files use one of two different names for the separation column, so
    both are accepted here and renamed to a single consistent column.
    """
    df = pd.read_csv(input_file_path)

    if 'Centromere Separation' in df.columns:
        separation = df['Centromere Separation']
    elif 'Centromere Separation (Gaussian)' in df.columns:
        separation = df['Centromere Separation (Gaussian)']
    else:
        raise ValueError(
            f"Required columns not found in {input_file_path}. Expected "
            "'Centromere Separation' or 'Centromere Separation (Gaussian)'."
        )

    if 'Spindle Length' not in df.columns:
        raise ValueError(f"Required column 'Spindle Length' not found in {input_file_path}.")

    df_cleaned = pd.DataFrame({
        'Centromere Separation (um)': separation,
        'Spindle Length (um)': df['Spindle Length'],
        'Time (sec)': df.index * TIME_STEP,
    })
    return df_cleaned

def pool_centromere_separations(input_file_paths):
    """Collect the centromere separation of every timepoint of every given cell.

    Returns a single 1D array with all of the NaN measurements removed.
    """
    separations_per_cell = []
    for input_file_path in input_file_paths:
        df_cleaned = load_cell(input_file_path)
        separations_per_cell.append(df_cleaned['Centromere Separation (um)'])

    pooled_separations = pd.concat(separations_per_cell, ignore_index=True)
    pooled_separations = pooled_separations.dropna()
    return pooled_separations.values

def fit_emission_gmm(separations, n_states=N_TENSION_STATES, random_state=42):
    """Fit a Gaussian mixture model to a pooled array of centromere separations.

    Parameters
    ----------
    separations : array-like, shape (n_observations,)
        Every centromere separation measurement in the batch.
    n_states : int
        Number of mixture components to fit.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    transformer : PowerTransformer
        The fitted transform. Every trajectory in the batch must be passed
        through this same object before it can be decoded.
    means : np.ndarray, shape (n_states,)
        Component means in the transformed space, sorted lowest to highest.
    variances : np.ndarray, shape (n_states,)
        Component variances in the transformed space, ordered like *means*.
    weights : np.ndarray, shape (n_states,)
        Component mixing weights, ordered like *means*.

    Notes
    -----
    Centromere separation cannot be negative and piles up against zero, which
    a Gaussian cannot represent - it would have to put half its density below
    zero. Fitting in a power transformed space removes that boundary, so the
    mixture can place a component on the unseparated population.
    """
    observations = np.asarray(separations, dtype=float).reshape(-1, 1)

    # Yeo-Johnson rather than Box-Cox because it is defined at zero. Leaving
    # the result unstandardised means the single fitted exponent (lambdas_)
    # fully describes the transform, so it can be reconstructed from the
    # summary CSV alone.
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed = transformer.fit_transform(observations)

    # GaussianMixture runs the whole EM fit internally and exposes the fitted
    # means, variances and mixing weights of each component afterwards.
    model = BayesianGaussianMixture(
        n_components=n_states,
        covariance_type='diag',
        random_state=random_state,
        n_init=50,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=100.0,  # larger -> weights pushed toward 1/K
    )
    model.fit(transformed)

    means = model.means_.flatten()
    variances = model.covariances_.flatten()
    weights = model.weights_.flatten()

    # Sort the components by mean so they line up with the tension labels:
    # the lowest mean component is No Tension and the highest is High Tension.
    # The transform is monotonic, so this order also holds in microns.
    sort_order = np.argsort(means)
    means = means[sort_order]
    variances = variances[sort_order]
    weights = weights[sort_order]

    return transformer, means, variances, weights

def yeo_johnson_forward(separations, lam):
    """Apply the Yeo-Johnson power transform to non-negative separations.

    This reproduces PowerTransformer(standardize=False) from the exponent
    alone, so a saved lambda is enough to redo the transform elsewhere.
    """
    separations = np.asarray(separations, dtype=float)

    if np.isclose(lam, 0.0):
        return np.log(separations + 1.0)

    return ((separations + 1.0) ** lam - 1.0) / lam

def mixture_density_in_microns(separations, means, variances, weights, lam):
    """Evaluate the fitted mixture as a density over separation in microns.

    The components are Gaussian in the transformed space, so each separation
    is transformed first and the result is multiplied by the derivative of the
    transform (the change of variables factor). Use this to draw the fit on
    top of a histogram of the raw separations.
    """
    separations = np.asarray(separations, dtype=float)
    transformed = yeo_johnson_forward(separations, lam)

    density = np.zeros_like(separations)
    for mean, variance, weight in zip(means, variances, weights):
        gaussian = np.exp(-0.5 * (transformed - mean) ** 2 / variance)
        density += weight * gaussian / np.sqrt(2 * np.pi * variance)

    # Change of variables: multiply by |d(transformed)/d(separation)|.
    derivative = (separations + 1.0) ** (lam - 1.0)
    return density * derivative

def component_density_in_microns(separations, mean, variance, weight, lam):
    """Evaluate one mixture component as a density over separation in microns."""
    return mixture_density_in_microns(separations, [mean], [variance], [weight], lam)

def annotate_data_trajectory(
    df,
    transformer,
    means,
    covars,
    column='Centromere Separation (um)',
    transmat=DEFAULT_TRANSMAT,
    startprob=DEFAULT_STARTPROB,
    n_iter=100,
    random_state=42,
):
    """Annotate a trajectory with tension-state labels using a Gaussian HMM.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory data containing the separation column.
    transformer : PowerTransformer
        The transform the emissions were fit in, from ``fit_emission_gmm``.
    means : array-like, shape (n_states,) or (n_states, 1)
        Means of the Gaussian emission distributions, in the transformed space.
    covars : array-like, shape (n_states,) or (n_states, 1)
        Variances of the Gaussian emission distributions, transformed space.
    column : str
        Column to use as the HMM emission observation.
    transmat : array-like, shape (n_states, n_states)
        Transition probability matrix (row-stochastic).
    startprob : array-like, shape (n_states,)
        Initial state probabilities.
    n_iter : int
        Maximum number of EM iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``Tension Classification`` column of state labels
        and a ``HMM State`` column of integer state indices.
    """
    n_states = len(means)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=n_iter,
        random_state=random_state,
        init_params='',     # Don't auto-initialize anything — we set all params manually
        params='',           # Don't update any params — just decode states with the given model
    )
    model.means_ = np.array(means, dtype=float).reshape(n_states, 1)
    model.covars_ = np.array(covars, dtype=float).reshape(n_states, 1)
    model.transmat_ = np.array(transmat, dtype=float)
    model.startprob_ = np.array(startprob, dtype=float)

    # Drop rows with NaN in the observation column so the HMM can decode.
    valid_mask = df[column].notna()
    raw_observations = df.loc[valid_mask, [column]].values.astype(float)
    if raw_observations.shape[0] == 0:
        df['Tension Classification'] = [np.nan for _ in range(len(df))]
        return df

    # Decode in the same transformed space the emissions were fit in.
    X = transformer.transform(raw_observations)
    states = model.predict(X)

    new_df = df.copy()
    new_df['HMM State'] = np.nan
    new_df.loc[valid_mask, 'HMM State'] = states
    new_df['Tension Classification'] = new_df['HMM State'].apply(
        lambda s: STATE_LABELS[int(s)] if pd.notna(s) else np.nan
    )
    new_df.drop(columns=['HMM State'], inplace=True)
    return new_df

def split_batch_name(phenotype):
    """Split a batch name into its mutant part and its centromere part.

    The centromere comes last, so 'Ase1-7A CENIV' gives the mutant 'Ase1-7A'
    and the centromere 'CENIV'. If the name does not end in a known centromere
    the whole name is treated as the mutant and the centromere is left blank.
    """
    for centromere in CENTROMERE_NAMES:
        if phenotype.endswith(centromere):
            mutant = phenotype[:-len(centromere)].strip()
            return mutant, centromere

    return phenotype, ''

def build_output_path(input_file_path):
    """Map an input path onto the matching path in the output directory.

    For example 'output/YV3640/cell.csv' becomes 'trajectories/YV3640/cell.csv'.
    """
    relative_path = os.path.relpath(input_file_path, INPUT_DIRECTORY)
    return os.path.join(OUTPUT_DIRECTORY, relative_path)

def process_file(input_file_path, transformer, means, covars):
    """Process and clean the CSV file and save it in the trjactories directory using the same overall structure.
    (Creating necicary directories if they don't exist)"""
    # Read and clean the data.
    df_cleaned = load_cell(input_file_path)

    # Label each timepoint using this batch's emission distributions.
    df_cleaned = annotate_data_trajectory(df_cleaned, transformer, means, covars)

    # Create the output directory structure
    output_path = build_output_path(input_file_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the cleaned DataFrame to the new location
    df_cleaned.to_csv(output_path, index=False)

    return df_cleaned


def identify_time_of_anaphase(df):
    """Identify the time of anaphase onset in a trajectory.

    Anaphase is defined as the first timepoint where the spindle length exceeds
    2.5 microns for more than three timepoints in a row. If no such timepoint exists, return NaN.
    """
    spindle_length_threshold = 2.5
    consecutive_timepoints_required = 3

    # Create a boolean mask where spindle length exceeds the threshold
    above_threshold = df['Spindle Length (um)'] > spindle_length_threshold

    # Find the indices where the condition is met for the required number of consecutive timepoints
    count = 0
    for index, is_above in enumerate(above_threshold):
        if is_above:
            count += 1
            if count >= consecutive_timepoints_required:
                return df['Time (sec)'].iloc[index - consecutive_timepoints_required + 1]
        else:
            count = 0

    return np.nan  # Return NaN if no anaphase onset is found


def build_combined_trajectory(df_cleaned, phenotype, input_file_path):
    """Build the combined-output version of one cell's annotated trajectory.

    The mutant and centromere come from the batch name and the cell ID is the
    name of the cell's file, so every row can be traced back to its source.
    """
    mutant, centromere = split_batch_name(phenotype)
    cell_id = os.path.splitext(os.path.basename(input_file_path))[0]

    combined_df = pd.DataFrame({
        'Mutant': mutant,
        'CEN': centromere,
        'Cell ID': cell_id,
        'Time (sec)': df_cleaned['Time (sec)'],
        'Spindle Length (um)': df_cleaned['Spindle Length (um)'],
        'Centromere Separation (um)': df_cleaned['Centromere Separation (um)'],
        'Tension Classification': df_cleaned['Tension Classification'],
        'Time Until Anaphase (sec)': df_cleaned['Time (sec)'] - identify_time_of_anaphase(df_cleaned),
    })
    return combined_df

def build_gmm_summary_rows(phenotype, yv_strains, transformer, means, variances, weights):
    """Build one summary row per GMM component for a single batch.

    The mean and variance are in the transformed space, because that is where
    the components are Gaussian and that is what the HMM consumes. The
    separation columns read each component back in microns as a centre and the
    range one standard deviation either side of it.

    A variance cannot be converted to microns on its own: the transform is
    non-linear, so a symmetric interval in the transformed space maps to an
    asymmetric one in microns. That is why the bounds are given explicitly.
    Note also that the centre is the separation that maps to the mean, which
    behaves like a median rather than an average.
    """
    standard_deviations = np.sqrt(variances)
    centres = transformer.inverse_transform(means.reshape(-1, 1)).flatten()
    lower = transformer.inverse_transform((means - standard_deviations).reshape(-1, 1)).flatten()
    upper = transformer.inverse_transform((means + standard_deviations).reshape(-1, 1)).flatten()

    summary_rows = []
    for state_index in range(len(means)):
        summary_rows.append({
            'Strain': phenotype,
            'YV Strains': ', '.join(yv_strains),
            'Component': state_index,
            'Tension State': STATE_LABELS[state_index],
            'Mean (transformed)': means[state_index],
            'Variance (transformed)': variances[state_index],
            'Weight': weights[state_index],
            'Separation (um)': centres[state_index],
            'Separation Low (um)': lower[state_index],
            'Separation High (um)': upper[state_index],
            'Lambda': transformer.lambdas_[0],
        })
    return summary_rows


if __name__ == '__main__':
    from tqdm import tqdm

    gmm_summary_rows = []
    combined_trajectories = []

    for phenotype, yv_strains in BATCHES.items():
        print(f"\nProcessing {phenotype}: {', '.join(yv_strains)}")

        input_files = find_batch_files(yv_strains)
        if len(input_files) == 0:
            print(f"  No CSV files found for {phenotype}. Skipping.")
            continue

        # Fit this batch's emission distributions to every timepoint of every
        # cell in the batch at once.
        pooled_separations = pool_centromere_separations(input_files)
        transformer, means, variances, weights = fit_emission_gmm(pooled_separations)

        # Read the components back in microns just for the printout.
        standard_deviations = np.sqrt(variances)
        centres = transformer.inverse_transform(means.reshape(-1, 1)).flatten()
        lower = transformer.inverse_transform((means - standard_deviations).reshape(-1, 1)).flatten()
        upper = transformer.inverse_transform((means + standard_deviations).reshape(-1, 1)).flatten()

        print(f"  {len(input_files)} cells, {len(pooled_separations)} timepoints, "
              f"lambda={transformer.lambdas_[0]:.3f}")
        for state_index in range(len(means)):
            print(
                f"  {STATE_LABELS[state_index]:>12}: "
                f"separation={centres[state_index]:.3f} um "
                f"[{lower[state_index]:.3f}, {upper[state_index]:.3f}], "
                f"weight={weights[state_index]:.3f}"
            )

        gmm_summary_rows.extend(
            build_gmm_summary_rows(phenotype, yv_strains, transformer, means, variances, weights)
        )

        # Annotate every cell in the batch using the batch's own emissions.
        for file in tqdm(input_files, desc=f"  {phenotype}"):
            try:
                df_cleaned = process_file(file, transformer, means, variances)
                combined_trajectories.append(
                    build_combined_trajectory(df_cleaned, phenotype, file)
                )
            except Exception as e:
                print(f"Error processing file {file}:")
                raise e

    # Save the fitted GMM components, one row per component per batch.
    gmm_summary_df = pd.DataFrame(gmm_summary_rows)
    os.makedirs(os.path.dirname(GMM_SUMMARY_PATH), exist_ok=True)
    gmm_summary_df.to_csv(GMM_SUMMARY_PATH, index=False)
    print(f"\nSaved GMM components for {len(BATCHES)} batches to {GMM_SUMMARY_PATH}.")

    # Save every cell's trajectory stacked together into one table.
    combined_df = pd.concat(combined_trajectories, ignore_index=True)
    combined_df.to_csv(COMBINED_TRAJECTORY_PATH, index=False)
    print(f"Saved {len(combined_df)} combined timepoints to {COMBINED_TRAJECTORY_PATH}.")

    print('Done processing all files.')
