import pandas as pd
import re
import unicodedata


# ============================================================
# STEP 17 - FAST BLOCKING EVALUATION
# ============================================================

print("\n===== STEP 17: FAST BLOCKING EVALUATION =====")


# ============================================================
# CONFIGURATION
# ============================================================

# Keep this deliberately small for the first experiment.
# We can increase it later if necessary.
GROUND_TRUTH_SAMPLE = 10000

# Maximum number of external records loaded into memory
# for the blocking experiment.
EXTERNAL_SAMPLE = 500000


# ============================================================
# FAST NORMALIZATION
# ============================================================

def normalize_text(series):
    """
    Vectorized-ish Pandas normalization.

    Keeps Unicode characters.
    """

    series = series.fillna("").astype(str)

    series = series.str.strip()

    # Unicode normalization
    series = series.map(
        lambda x: unicodedata.normalize("NFKC", x)
    )

    # Lowercase / casefold
    series = series.str.casefold()

    # Replace punctuation with spaces.
    series = series.str.replace(
        r"[^\w\s]",
        " ",
        regex=True
    )

    # Collapse whitespace
    series = series.str.replace(
        r"\s+",
        " ",
        regex=True
    )

    return series.str.strip()


def compact_text(series):
    """
    Remove spaces from normalized text.
    """

    return series.str.replace(
        r"\s+",
        "",
        regex=True
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


print(
    "S1 records:",
    len(s1)
)

print(
    "S2 records:",
    len(s2)
)

print(
    "S3 records:",
    len(s3)
)


# ============================================================
# SAMPLE GROUND TRUTH
# ============================================================

gt_sample = gt.sample(
    n=min(
        GROUND_TRUTH_SAMPLE,
        len(gt)
    ),
    random_state=42
).copy()


print(
    "\nGround truth rows sampled:",
    len(gt_sample)
)


# ============================================================
# GET REQUIRED EXTERNAL IDS
# ============================================================

print(
    "\nExtracting true matched IDs..."
)


def extract_ids(value):
    if pd.isna(value):
        return []

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


required_s2_ids = set()
required_s3_ids = set()

for value in gt_sample["matched_entity_ids"]:

    for entity_id in extract_ids(value):

        if entity_id.startswith("S2-"):
            required_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            required_s3_ids.add(entity_id)


print(
    "True S2 IDs required:",
    len(required_s2_ids)
)

print(
    "True S3 IDs required:",
    len(required_s3_ids)
)


# ============================================================
# CREATE SMALL EXTERNAL DATASET
# ============================================================

print(
    "\nPreparing external records..."
)


# Always include all true matched records.
s2_true = s2[
    s2["entity_id"].isin(
        required_s2_ids
    )
].copy()

s3_true = s3[
    s3["entity_id"].isin(
        required_s3_ids
    )
].copy()


# Add random negative/background records.
s2_background = s2[
    ~s2["entity_id"].isin(
        required_s2_ids
    )
].sample(
    n=min(
        EXTERNAL_SAMPLE,
        len(s2)
    ),
    random_state=42
)

s3_background = s3[
    ~s3["entity_id"].isin(
        required_s3_ids
    )
].sample(
    n=min(
        EXTERNAL_SAMPLE,
        len(s3)
    ),
    random_state=42
)


s2_eval = pd.concat(
    [
        s2_true,
        s2_background
    ],
    ignore_index=True
)

s3_eval = pd.concat(
    [
        s3_true,
        s3_background
    ],
    ignore_index=True
)


print(
    "S2 evaluation records:",
    len(s2_eval)
)

print(
    "S3 evaluation records:",
    len(s3_eval)
)


# ============================================================
# NORMALIZE EXTERNAL DATA
# ============================================================

print(
    "\nNormalizing S2..."
)

s2_eval["name_norm"] = normalize_text(
    s2_eval["business_name"]
)

s2_eval["name_compact"] = compact_text(
    s2_eval["name_norm"]
)

s2_eval["country_norm"] = normalize_text(
    s2_eval["country"]
)

s2_eval["address_norm"] = normalize_text(
    s2_eval["business_address"]
)


print(
    "Normalizing S3..."
)

s3_eval["name_norm"] = normalize_text(
    s3_eval["business_name"]
)

s3_eval["name_compact"] = compact_text(
    s3_eval["name_norm"]
)

s3_eval["country_norm"] = normalize_text(
    s3_eval["country"]
)

s3_eval["address_norm"] = normalize_text(
    s3_eval["business_address"]
)


# ============================================================
# CREATE BLOCKING INDEXES
# ============================================================

print(
    "\nCreating blocking indexes..."
)


def build_index(df, column):

    temp = df[
        [
            "entity_id",
            "country_norm",
            column
        ]
    ].copy()

    # Remove empty keys
    temp = temp[
        (temp["country_norm"] != "")
        &
        (temp[column] != "")
    ]

    # Multi-column key
    temp["block_key"] = (
        temp["country_norm"]
        + "||"
        + temp[column]
    )

    return (
        temp.groupby("block_key")[
            "entity_id"
        ]
        .apply(set)
        .to_dict()
    )


# ------------------------------------------------------------
# S2 indexes
# ------------------------------------------------------------

s2_exact_name = build_index(
    s2_eval,
    "name_norm"
)

s2_compact_name = build_index(
    s2_eval,
    "name_compact"
)

s2_address = build_index(
    s2_eval,
    "address_norm"
)


# ------------------------------------------------------------
# S3 indexes
# ------------------------------------------------------------

s3_exact_name = build_index(
    s3_eval,
    "name_norm"
)

s3_compact_name = build_index(
    s3_eval,
    "name_compact"
)

s3_address = build_index(
    s3_eval,
    "address_norm"
)


# ============================================================
# PREPARE S1 SAMPLE
# ============================================================

s1_sample = s1[
    s1["entity_id"].isin(
        gt_sample["source1_entity_id"]
    )
].copy()


s1_sample["name_norm"] = normalize_text(
    s1_sample["business_name"]
)

s1_sample["name_compact"] = compact_text(
    s1_sample["name_norm"]
)

s1_sample["country_norm"] = normalize_text(
    s1_sample["country"]
)

s1_sample["address_norm"] = normalize_text(
    s1_sample["business_address"]
)


# ============================================================
# GROUND TRUTH LOOKUP
# ============================================================

gt_lookup = gt_sample.set_index(
    "source1_entity_id"
)


# ============================================================
# BLOCKING TEST
# ============================================================

strategies = {
    "exact_name": [],
    "compact_name": [],
    "address": [],
    "name_union": []
}


total_true_matches = 0


# Statistics
candidate_counts = {
    key: []
    for key in strategies
}

recovered_counts = {
    key: 0
    for key in strategies
}


# ============================================================
# PROCESS S1 SAMPLE
# ============================================================

print(
    "\nEvaluating blocking..."
)


for _, row in s1_sample.iterrows():

    s1_id = row["entity_id"]

    gt_row = gt_lookup.loc[s1_id]

    true_ids = set(
        extract_ids(
            gt_row["matched_entity_ids"]
        )
    )

    # --------------------------------------------------------
    # Process S2 and S3 independently
    # --------------------------------------------------------

    for source in ["S2", "S3"]:

        true_source_ids = {
            x
            for x in true_ids
            if x.startswith(
                source + "-"
            )
        }

        total_true_matches += (
            len(true_source_ids)
        )

        # ----------------------------------------------------
        # Select index
        # ----------------------------------------------------

        if source == "S2":

            exact_index = s2_exact_name
            compact_index = s2_compact_name
            address_index = s2_address

        else:

            exact_index = s3_exact_name
            compact_index = s3_compact_name
            address_index = s3_address

        country = row["country_norm"]

        name = row["name_norm"]

        compact = row["name_compact"]

        address = row["address_norm"]

        # ----------------------------------------------------
        # Exact name
        # ----------------------------------------------------

        key = (
            country
            + "||"
            + name
        )

        exact_candidates = (
            exact_index.get(
                key,
                set()
            )
        )

        candidate_counts[
            "exact_name"
        ].append(
            len(exact_candidates)
        )

        recovered_counts[
            "exact_name"
        ] += len(
            true_source_ids
            & exact_candidates
        )

        # ----------------------------------------------------
        # Compact name
        # ----------------------------------------------------

        key = (
            country
            + "||"
            + compact
        )

        compact_candidates = (
            compact_index.get(
                key,
                set()
            )
        )

        candidate_counts[
            "compact_name"
        ].append(
            len(compact_candidates)
        )

        recovered_counts[
            "compact_name"
        ] += len(
            true_source_ids
            & compact_candidates
        )

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        key = (
            country
            + "||"
            + address
        )

        address_candidates = (
            address_index.get(
                key,
                set()
            )
        )

        candidate_counts[
            "address"
        ].append(
            len(address_candidates)
        )

        recovered_counts[
            "address"
        ] += len(
            true_source_ids
            & address_candidates
        )

        # ----------------------------------------------------
        # Union of name strategies
        # ----------------------------------------------------

        union_candidates = (
            exact_candidates
            |
            compact_candidates
        )

        candidate_counts[
            "name_union"
        ].append(
            len(union_candidates)
        )

        recovered_counts[
            "name_union"
        ] += len(
            true_source_ids
            & union_candidates
        )


# ============================================================
# RESULTS
# ============================================================

print(
    "\n===== BLOCKING RESULTS ====="
)

result_rows = []


for strategy in strategies:

    if total_true_matches > 0:

        recall = (
            recovered_counts[strategy]
            /
            total_true_matches
            *
            100
        )

    else:

        recall = 0

    counts = candidate_counts[
        strategy
    ]

    average_candidates = (
        sum(counts)
        /
        len(counts)
        if counts
        else 0
    )

    max_candidates = (
        max(counts)
        if counts
        else 0
    )

    result_rows.append(
        {
            "strategy": strategy,

            "true_matches":
                total_true_matches,

            "recovered_matches":
                recovered_counts[
                    strategy
                ],

            "candidate_recall_%":
                round(
                    recall,
                    2
                ),

            "avg_candidates":
                round(
                    average_candidates,
                    2
                ),

            "max_candidates":
                max_candidates
        }
    )


results = pd.DataFrame(
    result_rows
)


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "output/blocking_results.tsv",
    sep="\t",
    index=False
)


print(
    "\nSaved:"
)

print(
    "output/blocking_results.tsv"
)


print(
    "\n===== BLOCKING EVALUATION COMPLETE ====="
)