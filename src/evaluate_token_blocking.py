import pandas as pd
import numpy as np
import re
import unicodedata
from collections import defaultdict, Counter


print("\n===== STEP 20: ADVANCED TOKEN BLOCKING EVALUATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "dataset/train"
OUTPUT_FILE = "output/token_blocking_results.tsv"

GT_SAMPLE_SIZE = 10000
BACKGROUND_SIZE = 500000
RANDOM_STATE = 42

# Tokens appearing this many times or fewer are considered
# candidates for rare-token blocking.
RARE_TOKEN_MAX_DF = 50

# Ignore extremely short tokens.
MIN_TOKEN_LENGTH = 3


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

    tokens = value.split()

    return tokens[0] if tokens else ""


def last_token(value):

    if not value:
        return ""

    tokens = value.split()

    return tokens[-1] if tokens else ""


def prefix(value, n):

    if not value:
        return ""

    return value[:n]


def get_tokens(value):

    if not value:
        return []

    return [
        token
        for token in value.split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


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
# PARSE GROUND TRUTH IDS
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
# EXTRACT TRUE S2/S3 IDS
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
# PREPARE EXTERNAL EVALUATION DATA
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
# NORMALIZATION
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

    df["name_prefix_5"] = (
        df["name_compact"]
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

    df["name_first_token"] = (
        df["name_norm"]
        .map(first_token)
    )

    df["name_last_token"] = (
        df["name_norm"]
        .map(last_token)
    )

    df["name_tokens"] = (
        df["name_norm"]
        .map(get_tokens)
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
# TOKEN DOCUMENT FREQUENCY
# ============================================================

def calculate_token_frequency(df):

    counter = Counter()

    for tokens in df["name_tokens"]:

        # Count each token once per business record.
        unique_tokens = set(tokens)

        for token in unique_tokens:
            counter[token] += 1

    return counter


print("\nCalculating S2 token frequencies...")

s2_token_df = calculate_token_frequency(
    s2_eval
)

print(
    f"Unique S2 tokens: {len(s2_token_df)}"
)


print("Calculating S3 token frequencies...")

s3_token_df = calculate_token_frequency(
    s3_eval
)

print(
    f"Unique S3 tokens: {len(s3_token_df)}"
)


# ============================================================
# BUILD TOKEN INDEX
# ============================================================

def build_token_index(df):

    index = defaultdict(set)

    for country, entity_id, tokens in zip(
        df["country_norm"],
        df["entity_id"],
        df["name_tokens"]
    ):

        if not country:
            continue

        for token in set(tokens):

            if len(token) < MIN_TOKEN_LENGTH:
                continue

            index[
                (
                    country,
                    token
                )
            ].add(entity_id)

    return index


print("\nBuilding S2 token index...")

s2_token_index = build_token_index(
    s2_eval
)


print("Building S3 token index...")

s3_token_index = build_token_index(
    s3_eval
)


# ============================================================
# BUILD OTHER BLOCKING INDEXES
# ============================================================

def build_simple_index(
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


def build_indexes(df):

    print("Building standard indexes...")

    indexes = {}

    indexes["name_prefix_5"] = (
        build_simple_index(
            df,
            "name_prefix_5"
        )
    )

    indexes["address_prefix_8"] = (
        build_simple_index(
            df,
            "address_prefix_8"
        )
    )

    indexes["name_first_token"] = (
        build_simple_index(
            df,
            "name_first_token"
        )
    )

    indexes["name_last_token"] = (
        build_simple_index(
            df,
            "name_last_token"
        )
    )

    return indexes


print("\nBuilding S2 indexes...")

s2_indexes = build_indexes(
    s2_eval
)


print("\nBuilding S3 indexes...")

s3_indexes = build_indexes(
    s3_eval
)


# ============================================================
# RARE TOKEN SELECTION
# ============================================================

def select_rare_tokens(
    tokens,
    token_frequency
):

    candidates = []

    for token in set(tokens):

        if len(token) < MIN_TOKEN_LENGTH:
            continue

        frequency = token_frequency.get(
            token,
            0
        )

        if (
            frequency > 0
            and frequency <= RARE_TOKEN_MAX_DF
        ):

            candidates.append(
                (
                    frequency,
                    token
                )
            )

    # Lowest document frequency first.
    candidates.sort()

    # Use at most two rare tokens per S1.
    return [
        token
        for _, token in candidates[:2]
    ]


# ============================================================
# GET STANDARD CANDIDATES
# ============================================================

def get_standard_candidates(
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
# GET TOKEN CANDIDATES
# ============================================================

def get_token_candidates(
    row,
    token_index
):

    country = row["country_norm"]

    if not country:
        return set()

    candidates = set()

    for token in set(
        row["name_tokens"]
    ):

        if len(token) < MIN_TOKEN_LENGTH:
            continue

        candidates.update(
            token_index.get(
                (
                    country,
                    token
                ),
                set()
            )
        )

    return candidates


# ============================================================
# GET RARE TOKEN CANDIDATES
# ============================================================

def get_rare_token_candidates(
    row,
    token_index,
    token_frequency
):

    country = row["country_norm"]

    if not country:
        return set()

    rare_tokens = select_rare_tokens(
        row["name_tokens"],
        token_frequency
    )

    candidates = set()

    for token in rare_tokens:

        candidates.update(
            token_index.get(
                (
                    country,
                    token
                ),
                set()
            )
        )

    return candidates


# ============================================================
# MULTI-BLOCK STRATEGIES
# ============================================================

strategies = {

    "prefix5_lasttoken": [
        "prefix5",
        "lasttoken"
    ],

    "prefix5_token": [
        "prefix5",
        "token"
    ],

    "prefix5_rare_token": [
        "prefix5",
        "rare_token"
    ],

    "prefix5_address8_rare_token": [
        "prefix5",
        "address8",
        "rare_token"
    ],

    "prefix5_firsttoken_address8_rare_token": [
        "prefix5",
        "firsttoken",
        "address8",
        "rare_token"
    ],

}


# ============================================================
# EVALUATE STRATEGY
# ============================================================

def evaluate_strategy(
    s1_df,
    indexes,
    token_index,
    token_frequency,
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

        # ----------------------------------------------------
        # TRUE MATCHES
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # UNION CANDIDATES
        # ----------------------------------------------------

        candidates = set()


        for block in blocks:

            if block == "prefix5":

                block_candidates = (
                    get_standard_candidates(
                        row,
                        indexes[
                            "name_prefix_5"
                        ],
                        "name_prefix_5"
                    )
                )


            elif block == "address8":

                block_candidates = (
                    get_standard_candidates(
                        row,
                        indexes[
                            "address_prefix_8"
                        ],
                        "address_prefix_8"
                    )
                )


            elif block == "firsttoken":

                block_candidates = (
                    get_standard_candidates(
                        row,
                        indexes[
                            "name_first_token"
                        ],
                        "name_first_token"
                    )
                )


            elif block == "lasttoken":

                block_candidates = (
                    get_standard_candidates(
                        row,
                        indexes[
                            "name_last_token"
                        ],
                        "name_last_token"
                    )
                )


            elif block == "token":

                block_candidates = (
                    get_token_candidates(
                        row,
                        token_index
                    )
                )


            elif block == "rare_token":

                block_candidates = (
                    get_rare_token_candidates(
                        row,
                        token_index,
                        token_frequency
                    )
                )


            else:

                block_candidates = set()


            candidates.update(
                block_candidates
            )


        candidate_counts.append(
            len(candidates)
        )


        # ----------------------------------------------------
        # RECOVER TRUE MATCHES
        # ----------------------------------------------------

        recovered += len(
            true_ids.intersection(
                candidates
            )
        )


    # ========================================================
    # METRICS
    # ========================================================

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
# RUN S2
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
        s2_token_index,
        s2_token_df,
        "S2",
        blocks
    )

    results.append(
        result
    )


# ============================================================
# RUN S3
# ============================================================

print("\nEvaluating S3...")


for strategy_name, blocks in strategies.items():

    print(
        f"  Testing {strategy_name}..."
    )

    result = evaluate_strategy(
        s1_eval,
        s3_indexes,
        s3_token_index,
        s3_token_df,
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
    "\n===== STEP 20 RESULTS =====\n"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
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
    "\n===== STEP 20 COMPLETE =====\n"
)