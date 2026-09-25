import pandas as pd
import numpy as np
import re
import unicodedata
from collections import defaultdict


print("\n===== STEP 18: IMPROVED BLOCKING EVALUATION =====\n")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset/train"
OUTPUT_FILE = "output/blocking_results_v2.tsv"

GT_SAMPLE_SIZE = 10000
BACKGROUND_SIZE = 500000
RANDOM_STATE = 42


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text while preserving Unicode characters.
    """

    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    value = value.casefold()

    # Replace punctuation with spaces.
    # Unicode letters and numbers are preserved.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Collapse multiple spaces.
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def compact_text(value):
    """
    Remove spaces from normalized text.
    """

    if not value:
        return ""

    return value.replace(" ", "")


def first_token(value):
    """
    Return the first token of a normalized string.
    """

    if not value:
        return ""

    parts = value.split()

    if not parts:
        return ""

    return parts[0]


def prefix(value, n):
    """
    Return first n characters.
    """

    if not value:
        return ""

    return value[:n]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

s1 = pd.read_csv(
    f"{TRAIN_DIR}/train_source1.tsv",
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    f"{TRAIN_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    f"{TRAIN_DIR}/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    f"{TRAIN_DIR}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print(f"S1 records: {len(s1)}")
print(f"S2 records: {len(s2)}")
print(f"S3 records: {len(s3)}")


# ============================================================
# SAMPLE GROUND TRUTH
# ============================================================

gt_sample = gt.sample(
    n=min(
        GT_SAMPLE_SIZE,
        len(gt)
    ),
    random_state=RANDOM_STATE
).copy()

print(
    f"\nGround truth rows sampled: {len(gt_sample)}"
)


# ============================================================
# PARSE MATCHED ENTITY IDS
# ============================================================

def extract_ids(value):
    """
    Convert matched_entity_ids into a set of IDs.

    Supports:
    - comma
    - semicolon
    - pipe

    Empty values produce an empty set.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    parts = re.split(
        r"[,;|]",
        value
    )

    return {
        x.strip()
        for x in parts
        if x.strip()
    }


# ============================================================
# BUILD GROUND-TRUTH MAP
# ============================================================

true_map = {}

for _, row in gt_sample.iterrows():

    s1_id = row["source1_entity_id"]

    true_map[s1_id] = extract_ids(
        row["matched_entity_ids"]
    )


# ============================================================
# SEPARATE S2 AND S3 TRUE IDS
# ============================================================

true_s2_ids = set()
true_s3_ids = set()

for ids in true_map.values():

    for entity_id in ids:

        if entity_id.startswith("S2-"):

            true_s2_ids.add(
                entity_id
            )

        elif entity_id.startswith("S3-"):

            true_s3_ids.add(
                entity_id
            )


print(
    f"True S2 IDs required: {len(true_s2_ids)}"
)

print(
    f"True S3 IDs required: {len(true_s3_ids)}"
)


# ============================================================
# PREPARE EXTERNAL EVALUATION DATA
# ============================================================

def prepare_external(
    source_df,
    true_ids,
    background_size
):

    true_ids = set(true_ids)

    # Keep every true match.
    true_rows = source_df[
        source_df["entity_id"].isin(true_ids)
    ].copy()

    # Remaining records can be used as background.
    remaining = source_df[
        ~source_df["entity_id"].isin(true_ids)
    ]

    bg_size = min(
        background_size,
        len(remaining)
    )

    background = remaining.sample(
        n=bg_size,
        random_state=RANDOM_STATE
    )

    result = pd.concat(
        [
            true_rows,
            background
        ],
        ignore_index=True
    )

    # Safety against accidental duplicate IDs.
    result = result.drop_duplicates(
        subset=["entity_id"]
    ).reset_index(
        drop=True
    )

    return result


print("\nPreparing external records...")


s2_eval = prepare_external(
    s2,
    true_s2_ids,
    BACKGROUND_SIZE
)

s3_eval = prepare_external(
    s3,
    true_s3_ids,
    BACKGROUND_SIZE
)


print(
    f"S2 evaluation records: {len(s2_eval)}"
)

print(
    f"S3 evaluation records: {len(s3_eval)}"
)


# ============================================================
# PREPARE S1
# ============================================================

print("\nNormalizing S1...")


s1_eval = s1[
    s1["entity_id"].isin(
        true_map.keys()
    )
].copy()


s1_eval["country_norm"] = (
    s1_eval["country"]
    .map(normalize_text)
)


s1_eval["name_norm"] = (
    s1_eval["business_name"]
    .map(normalize_text)
)


s1_eval["address_norm"] = (
    s1_eval["business_address"]
    .map(normalize_text)
)


s1_eval["name_compact"] = (
    s1_eval["name_norm"]
    .map(compact_text)
)


s1_eval["address_compact"] = (
    s1_eval["address_norm"]
    .map(compact_text)
)


# ============================================================
# NORMALIZE EXTERNAL SOURCES
# ============================================================

def normalize_external(df):

    df = df.copy()

    df["country_norm"] = (
        df["country"]
        .map(normalize_text)
    )

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_text)
    )

    df["name_compact"] = (
        df["name_norm"]
        .map(compact_text)
    )

    df["address_compact"] = (
        df["address_norm"]
        .map(compact_text)
    )

    return df


print("Normalizing S2...")

s2_eval = normalize_external(
    s2_eval
)


print("Normalizing S3...")

s3_eval = normalize_external(
    s3_eval
)


# ============================================================
# CREATE BLOCKING FEATURES
# ============================================================

def create_blocking_keys(df):

    df = df.copy()

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    df["name_prefix_3"] = (
        df["name_compact"]
        .map(
            lambda x: prefix(x, 3)
        )
    )

    df["name_prefix_5"] = (
        df["name_compact"]
        .map(
            lambda x: prefix(x, 5)
        )
    )

    df["name_prefix_7"] = (
        df["name_compact"]
        .map(
            lambda x: prefix(x, 7)
        )
    )

    df["name_first_token"] = (
        df["name_norm"]
        .map(first_token)
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    df["address_prefix_5"] = (
        df["address_compact"]
        .map(
            lambda x: prefix(x, 5)
        )
    )

    df["address_prefix_8"] = (
        df["address_compact"]
        .map(
            lambda x: prefix(x, 8)
        )
    )

    df["address_first_token"] = (
        df["address_norm"]
        .map(first_token)
    )

    return df


print("\nCreating blocking keys...")


s1_eval = create_blocking_keys(
    s1_eval
)

s2_eval = create_blocking_keys(
    s2_eval
)

s3_eval = create_blocking_keys(
    s3_eval
)


# ============================================================
# BUILD COUNTRY + KEY INDEX
# ============================================================

def build_country_index(
    df,
    key_column
):

    index = defaultdict(set)

    for country, entity_id, key in zip(
        df["country_norm"],
        df["entity_id"],
        df[key_column]
    ):

        if not country:
            continue

        if not key:
            continue

        compound_key = (
            country,
            key
        )

        index[
            compound_key
        ].add(entity_id)

    return index


# ============================================================
# BUILD ALL INDEXES
# ============================================================

def build_all_indexes(df):

    print("Building indexes...")

    indexes = {}

    columns = [

        "name_norm",

        "name_compact",

        "name_prefix_3",

        "name_prefix_5",

        "name_prefix_7",

        "name_first_token",

        "address_prefix_5",

        "address_prefix_8",

        "address_first_token",
    ]

    for column in columns:

        indexes[column] = (
            build_country_index(
                df,
                column
            )
        )

    return indexes


print("\nBuilding S2 indexes...")

s2_indexes = build_all_indexes(
    s2_eval
)


print("Building S3 indexes...")

s3_indexes = build_all_indexes(
    s3_eval
)


# ============================================================
# BLOCKING STRATEGIES
# ============================================================

strategies = [

    "name_norm",

    "name_compact",

    "name_prefix_3",

    "name_prefix_5",

    "name_prefix_7",

    "name_first_token",

    "address_prefix_5",

    "address_prefix_8",

    "address_first_token",
]


# ============================================================
# GET CANDIDATES
# ============================================================

def get_candidates(
    row,
    index,
    strategy
):

    country = row["country_norm"]

    if not country:
        return set()


    # --------------------------------------------------------
    # NAME BLOCKING
    # --------------------------------------------------------

    if strategy == "name_norm":

        key = row["name_norm"]


    elif strategy == "name_compact":

        key = row["name_compact"]


    elif strategy == "name_prefix_3":

        key = row["name_prefix_3"]


    elif strategy == "name_prefix_5":

        key = row["name_prefix_5"]


    elif strategy == "name_prefix_7":

        key = row["name_prefix_7"]


    elif strategy == "name_first_token":

        key = row["name_first_token"]


    # --------------------------------------------------------
    # ADDRESS BLOCKING
    # --------------------------------------------------------

    elif strategy == "address_prefix_5":

        key = row["address_prefix_5"]


    elif strategy == "address_prefix_8":

        key = row["address_prefix_8"]


    elif strategy == "address_first_token":

        key = row["address_first_token"]


    else:

        return set()


    if not key:

        return set()


    return index.get(
        (
            country,
            key
        ),
        set()
    )


# ============================================================
# EVALUATE ONE SOURCE
# ============================================================

def evaluate_source(
    s1_df,
    indexes,
    source_name
):

    print(
        f"\nEvaluating {source_name}..."
    )

    results = []


    for strategy in strategies:

        recovered = 0

        total_true = 0

        candidate_counts = []


        index = indexes[
            strategy
        ]


        for _, row in s1_df.iterrows():

            s1_id = row[
                "entity_id"
            ]


            # ------------------------------------------------
            # True matches for this source
            # ------------------------------------------------

            true_ids = {

                entity_id

                for entity_id in
                true_map.get(
                    s1_id,
                    set()
                )

                if entity_id.startswith(
                    source_name + "-"
                )
            }


            total_true += len(
                true_ids
            )


            # ------------------------------------------------
            # Generate candidates
            # ------------------------------------------------

            candidates = get_candidates(
                row,
                index,
                strategy
            )


            candidate_counts.append(
                len(candidates)
            )


            # ------------------------------------------------
            # Count recovered true matches
            # ------------------------------------------------

            recovered += len(
                true_ids.intersection(
                    candidates
                )
            )


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        if total_true > 0:

            recall = (
                recovered
                / total_true
                * 100
            )

        else:

            recall = 0


        # ----------------------------------------------------
        # Candidate statistics
        # ----------------------------------------------------

        if candidate_counts:

            avg_candidates = (
                np.mean(
                    candidate_counts
                )
            )

            median_candidates = (
                np.median(
                    candidate_counts
                )
            )

            max_candidates = int(
                max(
                    candidate_counts
                )
            )

        else:

            avg_candidates = 0

            median_candidates = 0

            max_candidates = 0


        results.append({

            "source": source_name,

            "strategy": strategy,

            "true_matches": total_true,

            "recovered_matches": recovered,

            "candidate_recall_%": round(
                recall,
                2
            ),

            "avg_candidates": round(
                avg_candidates,
                2
            ),

            "median_candidates": round(
                median_candidates,
                2
            ),

            "max_candidates": (
                max_candidates
            )
        })


    return results


# ============================================================
# RUN S2 EVALUATION
# ============================================================

s2_results = evaluate_source(
    s1_eval,
    s2_indexes,
    "S2"
)


# ============================================================
# RUN S3 EVALUATION
# ============================================================

s3_results = evaluate_source(
    s1_eval,
    s3_indexes,
    "S3"
)


# ============================================================
# COMBINE RESULTS
# ============================================================

results = pd.DataFrame(
    s2_results + s3_results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n===== STEP 18 RESULTS =====\n"
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
    OUTPUT_FILE,
    sep="\t",
    index=False
)


print(
    f"\nSaved: {OUTPUT_FILE}"
)


print(
    "\n===== STEP 18 COMPLETE =====\n"
)