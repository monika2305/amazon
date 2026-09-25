import pandas as pd
import numpy as np
import re
import unicodedata
from collections import defaultdict


print("\n===== STEP 19: MULTI-BLOCK CANDIDATE EVALUATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "dataset/train"
OUTPUT_FILE = "output/multiblock_results.tsv"

GT_SAMPLE_SIZE = 10000
BACKGROUND_SIZE = 500000
RANDOM_STATE = 42


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    value = value.casefold()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def compact_text(value):

    if not value:
        return ""

    return value.replace(" ", "")


def first_token(value):

    if not value:
        return ""

    parts = value.split()

    return parts[0] if parts else ""


def prefix(value, n):

    if not value:
        return ""

    return value[:n]


# ============================================================
# LOAD DATA
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
# PARSE TRUE IDS
# ============================================================

def extract_ids(value):

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


true_map = {}

for _, row in gt_sample.iterrows():

    true_map[
        row["source1_entity_id"]
    ] = extract_ids(
        row["matched_entity_ids"]
    )


# ============================================================
# GET TRUE S2/S3 IDS
# ============================================================

true_s2_ids = set()
true_s3_ids = set()

for ids in true_map.values():

    for entity_id in ids:

        if entity_id.startswith("S2-"):

            true_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):

            true_s3_ids.add(entity_id)


print(
    f"True S2 IDs required: {len(true_s2_ids)}"
)

print(
    f"True S3 IDs required: {len(true_s3_ids)}"
)


# ============================================================
# PREPARE EXTERNAL DATA
# ============================================================

def prepare_external(
    source_df,
    true_ids
):

    true_ids = set(true_ids)

    true_rows = source_df[
        source_df["entity_id"].isin(
            true_ids
        )
    ].copy()

    remaining = source_df[
        ~source_df["entity_id"].isin(
            true_ids
        )
    ]

    background_size = min(
        BACKGROUND_SIZE,
        len(remaining)
    )

    background = remaining.sample(
        n=background_size,
        random_state=RANDOM_STATE
    )

    result = pd.concat(
        [
            true_rows,
            background
        ],
        ignore_index=True
    )

    result = result.drop_duplicates(
        subset=["entity_id"]
    ).reset_index(
        drop=True
    )

    return result


print("\nPreparing external records...")

s2_eval = prepare_external(
    s2,
    true_s2_ids
)

s3_eval = prepare_external(
    s3,
    true_s3_ids
)

print(
    f"S2 evaluation records: {len(s2_eval)}"
)

print(
    f"S3 evaluation records: {len(s3_eval)}"
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_dataframe(df):

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

    # Name blocks
    df["name_prefix_3"] = (
        df["name_compact"]
        .map(lambda x: prefix(x, 3))
    )

    df["name_prefix_5"] = (
        df["name_compact"]
        .map(lambda x: prefix(x, 5))
    )

    df["name_prefix_7"] = (
        df["name_compact"]
        .map(lambda x: prefix(x, 7))
    )

    df["name_first_token"] = (
        df["name_norm"]
        .map(first_token)
    )

    # Address blocks
    df["address_prefix_5"] = (
        df["address_compact"]
        .map(lambda x: prefix(x, 5))
    )

    df["address_prefix_8"] = (
        df["address_compact"]
        .map(lambda x: prefix(x, 8))
    )

    return df


print("\nNormalizing S1...")

s1_eval = s1[
    s1["entity_id"].isin(
        true_map.keys()
    )
].copy()

s1_eval = normalize_dataframe(
    s1_eval
)


print("Normalizing S2...")

s2_eval = normalize_dataframe(
    s2_eval
)


print("Normalizing S3...")

s3_eval = normalize_dataframe(
    s3_eval
)


# ============================================================
# BLOCKING INDEX
# ============================================================

def build_index(
    df,
    column
):

    index = defaultdict(set)

    for country, entity_id, key in zip(
        df["country_norm"],
        df["entity_id"],
        df[column]
    ):

        if not country:
            continue

        if not key:
            continue

        index[
            (
                country,
                key
            )
        ].add(entity_id)

    return index


# ============================================================
# BUILD INDEXES
# ============================================================

BLOCK_COLUMNS = [

    "name_prefix_3",

    "name_prefix_5",

    "name_prefix_7",

    "name_first_token",

    "address_prefix_5",

    "address_prefix_8",
]


def build_all_indexes(df):

    indexes = {}

    print("Building indexes...")

    for column in BLOCK_COLUMNS:

        print(
            f"  {column}"
        )

        indexes[column] = build_index(
            df,
            column
        )

    return indexes


print("\nBuilding S2 indexes...")

s2_indexes = build_all_indexes(
    s2_eval
)


print("\nBuilding S3 indexes...")

s3_indexes = build_all_indexes(
    s3_eval
)


# ============================================================
# GET SINGLE BLOCK CANDIDATES
# ============================================================

def get_block_candidates(
    row,
    index,
    block
):

    country = row["country_norm"]

    if not country:
        return set()

    key = row[block]

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
# MULTI-BLOCK STRATEGIES
# ============================================================

strategies = {

    "prefix5": [
        "name_prefix_5"
    ],

    "prefix5_firsttoken": [
        "name_prefix_5",
        "name_first_token"
    ],

    "prefix5_address8": [
        "name_prefix_5",
        "address_prefix_8"
    ],

    "prefix5_firsttoken_address8": [
        "name_prefix_5",
        "name_first_token",
        "address_prefix_8"
    ],

    "prefix3_prefix5_address8": [
        "name_prefix_3",
        "name_prefix_5",
        "address_prefix_8"
    ],

}


# ============================================================
# EVALUATE MULTI-BLOCK STRATEGY
# ============================================================

def evaluate_strategy(
    s1_df,
    indexes,
    source_name,
    blocks
):

    total_true = 0
    recovered = 0

    candidate_counts = []

    for _, row in s1_df.iterrows():

        s1_id = row[
            "entity_id"
        ]

        # -----------------------------------------------
        # True matches
        # -----------------------------------------------

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


        # -----------------------------------------------
        # UNION BLOCKS
        # -----------------------------------------------

        candidates = set()

        for block in blocks:

            block_candidates = (
                get_block_candidates(
                    row,
                    indexes[block],
                    block
                )
            )

            candidates.update(
                block_candidates
            )


        candidate_counts.append(
            len(candidates)
        )


        # -----------------------------------------------
        # TRUE MATCHES RECOVERED
        # -----------------------------------------------

        recovered += len(
            true_ids.intersection(
                candidates
            )
        )


    # ====================================================
    # METRICS
    # ====================================================

    recall = (
        recovered
        / total_true
        * 100
        if total_true > 0
        else 0
    )

    avg_candidates = (
        np.mean(candidate_counts)
        if candidate_counts
        else 0
    )

    median_candidates = (
        np.median(candidate_counts)
        if candidate_counts
        else 0
    )

    max_candidates = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )


    return {

        "source": source_name,

        "strategy": "+".join(
            blocks
        ),

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

        "max_candidates": int(
            max_candidates
        )
    }


# ============================================================
# RUN EVALUATION
# ============================================================

results = []


print("\nEvaluating S2...")

for strategy_name, blocks in strategies.items():

    print(
        f"  Testing {strategy_name}..."
    )

    result = evaluate_strategy(
        s1_eval,
        s2_indexes,
        "S2",
        blocks
    )

    results.append(
        result
    )


print("\nEvaluating S3...")

for strategy_name, blocks in strategies.items():

    print(
        f"  Testing {strategy_name}..."
    )

    result = evaluate_strategy(
        s1_eval,
        s3_indexes,
        "S3",
        blocks
    )

    results.append(
        result
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n===== STEP 19 RESULTS =====\n"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


print(
    f"\nSaved: {OUTPUT_FILE}"
)


print(
    "\n===== STEP 19 COMPLETE =====\n"
)