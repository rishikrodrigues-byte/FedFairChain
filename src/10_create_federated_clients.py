from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = Path("data/processed/train.csv")
OUTPUT_DIR = Path("data/processed/federated_clients")

NUM_CLIENTS = 4

# Smaller alpha = stronger heterogeneity.
# 0.7 gives useful non-IID behaviour without making
# the hospitals unrealistically tiny or extreme.
DIRICHLET_ALPHA = 0.7

RANDOM_STATE = 42
MAX_ATTEMPTS = 1000

TARGETS = [
    "DIABETES",
    "HYPERTENSION",
    "CVD",
]


# ============================================================
# LOAD TRAINING DATA ONLY
# ============================================================

print("=" * 84)
print("FEDFAIRCHAIN - NON-IID FEDERATED CLIENT CONSTRUCTION")
print("=" * 84)

df = pd.read_csv(TRAIN_PATH)

print(f"\nTraining patients available : {len(df)}")
print(f"Number of simulated clients : {NUM_CLIENTS}")
print(f"Dirichlet alpha             : {DIRICHLET_ALPHA}")

print("\n[PASS] Validation dataset is not loaded.")
print("[PASS] Test dataset is not loaded.")


# ============================================================
# VALIDATE DATA
# ============================================================

if df["SEQN"].duplicated().any():
    raise ValueError("Duplicate SEQN values detected.")

for target in TARGETS:
    if not df[target].isin([0, 1]).all():
        raise ValueError(
            f"Invalid binary values detected in {target}"
        )


# ============================================================
# CREATE MULTI-LABEL PATTERN
# ============================================================

df = df.copy()

df["LABEL_PATTERN"] = (
    df[TARGETS]
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

print("\nTraining-set multi-label patterns:")

pattern_counts = (
    df["LABEL_PATTERN"]
    .value_counts()
    .sort_index()
)

for pattern, count in pattern_counts.items():
    print(
        f"{pattern}: "
        f"{count:<5} "
        f"({count / len(df) * 100:.2f}%)"
    )


# ============================================================
# NON-IID PARTITION FUNCTION
# ============================================================

def create_partition(seed):

    rng = np.random.default_rng(seed)

    client_indices = [
        []
        for _ in range(NUM_CLIENTS)
    ]

    for pattern in sorted(
        df["LABEL_PATTERN"].unique()
    ):

        pattern_indices = (
    df.index[
        df["LABEL_PATTERN"] == pattern
    ]
    .to_numpy()
    .copy()
)

        rng.shuffle(pattern_indices)

        proportions = rng.dirichlet(
            np.full(
                NUM_CLIENTS,
                DIRICHLET_ALPHA
            )
        )

        allocation = rng.multinomial(
            len(pattern_indices),
            proportions
        )

        start = 0

        for client_id, count in enumerate(
            allocation
        ):

            end = start + count

            client_indices[
                client_id
            ].extend(
                pattern_indices[
                    start:end
                ]
            )

            start = end

    clients = []

    for indices in client_indices:

        client_df = (
            df.loc[indices]
            .sample(
                frac=1,
                random_state=seed
            )
            .reset_index(drop=True)
        )

        clients.append(client_df)

    return clients


# ============================================================
# PARTITION QUALITY CHECK
# ============================================================

def partition_is_acceptable(clients):

    # Avoid unusably tiny clients.
    if min(len(client) for client in clients) < 700:
        return False

    # Every client should contain both classes for
    # every disease, with enough positive examples.
    for client in clients:

        for target in TARGETS:

            positives = int(
                client[target].sum()
            )

            negatives = (
                len(client) - positives
            )

            if positives < 30:
                return False

            if negatives < 30:
                return False

    # Require measurable heterogeneity.
    prevalence_gaps = []

    for target in TARGETS:

        prevalences = [
            client[target].mean()
            for client in clients
        ]

        gap = (
            max(prevalences)
            - min(prevalences)
        )

        prevalence_gaps.append(gap)

    # At least one disease should vary by
    # 10 percentage points or more across clients.
    if max(prevalence_gaps) < 0.10:
        return False

    return True


# ============================================================
# SEARCH FOR VALID NON-IID PARTITION
# ============================================================

clients = None
selected_seed = None

for attempt in range(MAX_ATTEMPTS):

    candidate_seed = (
        RANDOM_STATE + attempt
    )

    candidate_clients = (
        create_partition(
            candidate_seed
        )
    )

    if partition_is_acceptable(
        candidate_clients
    ):

        clients = candidate_clients
        selected_seed = candidate_seed
        break


if clients is None:
    raise RuntimeError(
        "Could not construct an acceptable "
        "non-IID partition."
    )


print(
    f"\nSelected partition seed: "
    f"{selected_seed}"
)


# ============================================================
# GLOBAL TRAINING PREVALENCE
# ============================================================

print("\n" + "=" * 84)
print("GLOBAL TRAINING DISTRIBUTION")
print("=" * 84)

print(f"Rows: {len(df)}")

for target in TARGETS:

    prevalence = (
        df[target].mean() * 100
    )

    print(
        f"{target:<13}: "
        f"{prevalence:.2f}%"
    )


# ============================================================
# CLIENT REPORT
# ============================================================

summary_rows = []

for client_id, client in enumerate(
    clients,
    start=1
):

    print("\n" + "=" * 84)
    print(f"SIMULATED HOSPITAL NODE {client_id}")
    print("=" * 84)

    print(
        f"Patients: {len(client)} "
        f"({len(client) / len(df) * 100:.2f}% "
        f"of training set)"
    )

    print("\nDisease prevalence:")

    summary = {
        "client":
            f"Hospital_{client_id}",

        "patients":
            len(client),
    }

    for target in TARGETS:

        positives = int(
            client[target].sum()
        )

        prevalence = (
            client[target].mean() * 100
        )

        summary[
            f"{target}_positive"
        ] = positives

        summary[
            f"{target}_prevalence"
        ] = prevalence

        print(
            f"{target:<13} "
            f"positive={positives:<5} "
            f"prevalence={prevalence:.2f}%"
        )

    print("\nMulti-label patterns:")

    client_patterns = (
        client["LABEL_PATTERN"]
        .value_counts()
        .sort_index()
    )

    for pattern in [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    ]:

        count = int(
            client_patterns.get(
                pattern,
                0
            )
        )

        percentage = (
            count / len(client) * 100
        )

        print(
            f"{pattern}: "
            f"{count:<5} "
            f"({percentage:.2f}%)"
        )

        summary[
            f"pattern_{pattern}"
        ] = count

    print("\nSex distribution:")

    male = int(
        (client["SEX"] == 1).sum()
    )

    female = int(
        (client["SEX"] == 2).sum()
    )

    print(
        f"Male   : {male} "
        f"({male / len(client) * 100:.2f}%)"
    )

    print(
        f"Female : {female} "
        f"({female / len(client) * 100:.2f}%)"
    )

    summary["male"] = male
    summary["female"] = female

    summary_rows.append(
        summary
    )


# ============================================================
# HETEROGENEITY AUDIT
# ============================================================

print("\n" + "=" * 84)
print("NON-IID HETEROGENEITY AUDIT")
print("=" * 84)

heterogeneity_passed = True

for target in TARGETS:

    prevalences = np.array(
        [
            client[target].mean() * 100
            for client in clients
        ]
    )

    gap = (
        prevalences.max()
        - prevalences.min()
    )

    print(
        f"{target:<13} "
        f"min={prevalences.min():.2f}%  "
        f"max={prevalences.max():.2f}%  "
        f"gap={gap:.2f} percentage points"
    )

    if gap >= 10.0:
        print(
            "  -> Strong heterogeneity "
            "for this target"
        )
    elif gap >= 5.0:
        print(
            "  -> Moderate heterogeneity "
            "for this target"
        )
    else:
        print(
            "  -> Mild heterogeneity "
            "for this target"
        )


# ============================================================
# SPLIT INTEGRITY
# ============================================================

print("\n" + "=" * 84)
print("FEDERATED PARTITION INTEGRITY")
print("=" * 84)

all_ids = []

for client in clients:
    all_ids.extend(
        client["SEQN"].tolist()
    )

unique_ids = set(all_ids)

print(
    "Total rows across clients :",
    len(all_ids)
)

print(
    "Unique patient IDs        :",
    len(unique_ids)
)

print(
    "Original training rows    :",
    len(df)
)

all_preserved = (
    len(all_ids) == len(df)
)

no_overlap = (
    len(unique_ids) == len(df)
)

if all_preserved:
    print("[PASS] All training patients preserved")
else:
    print("[FAIL] Training patients missing")

if no_overlap:
    print("[PASS] No patient appears in multiple clients")
else:
    print("[FAIL] Patient overlap detected")


# ============================================================
# SAVE CLIENT DATA
# ============================================================

if not (
    all_preserved
    and no_overlap
):
    raise RuntimeError(
        "Federated partition integrity failed."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for client_id, client in enumerate(
    clients,
    start=1
):

    save_df = client.drop(
        columns=["LABEL_PATTERN"]
    )

    path = (
        OUTPUT_DIR /
        f"hospital_{client_id}.csv"
    )

    save_df.to_csv(
        path,
        index=False
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    Path("results") /
    "federated_client_distribution.csv"
)

Path("results").mkdir(
    parents=True,
    exist_ok=True
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("\nSaved simulated client datasets:")

for client_id in range(
    1,
    NUM_CLIENTS + 1
):

    print(
        OUTPUT_DIR /
        f"hospital_{client_id}.csv"
    )

print(
    "\nSaved client distribution report:"
)
print(summary_path)


print("\nIMPORTANT:")
print(
    "These are simulated hospital nodes "
    "constructed only from the TRAINING set."
)

print(
    "They represent non-IID federated partitions, "
    "not four real hospitals."
)


print("\n" + "=" * 84)
print("STAGE 4.1 NON-IID FEDERATED CLIENT CREATION COMPLETE.")
print("=" * 84)