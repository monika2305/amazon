import pandas as pd
import numpy as np
import re
import unicodedata

from collections import defaultdict, Counter


print("\n===== STEP 21: FINAL BLOCKING VALIDATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "dataset/train"

S1_FILE = f"{TRAIN_DIR}/train_source1.tsv"
S2_FILE = f"{TRAIN_DIR}/train_source2.tsv"
S3_FILE = f"{TRAIN_DIR}/train_source3.tsv"
GT_FILE = f"{TRAIN_DIR}/train_ground_truth.tsv"

OUTPUT_FILE = "output/final_blocking_validation.tsv"

GT_SAMPLE_SIZE = 10000

CHUNK_SIZE = 250000

RANDOM_STATE = 42

RARE_TOKEN_MAX_DF = 50

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

    parts = value.split()

    return parts[0] if parts else ""


def get_tokens(value):

    if not value:
        return []

    return [
        token
        for token in value.split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def prefix(value, n):

    if not value:
        return ""

    return value[:n]


# ============================================================
# LOAD S1 + GROUND TRUTH
# ============================================================

print("Loading S1 and ground truth...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

print(
    f"S1 records: {len(s1)}"
)


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
    f"Ground truth rows sampled: {len(gt_sample)}"
)


# ============================================================
# PARSE GROUND TRUTH
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
# TRUE S2 / S3 IDS
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
# PREPARE S1 EVALUATION DATA
# ============================================================

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


s1_eval["name_prefix_5"] = (
    s1_eval["name_compact"]
    .map(
        lambda x: prefix(x, 5)
    )
)


s1_eval["address_compact"] = (
    s1_eval["address_norm"]
    .map(compact_text)
)


s1_eval["address_prefix_8"] = (
    s1_eval["address_compact"]
    .map(
        lambda x: prefix(x, 8)
    )
)


s1_eval["name_first_token"] = (
    s1_eval["name_norm"]
    .map(first_token)
)


s1_eval["name_tokens"] = (
    s1_eval["name_norm"]
    .map(get_tokens)
)


# ============================================================
# PASS 1:
# FULL DATASET TOKEN FREQUENCY
# ============================================================

def calculate_full_token_frequency(
    filepath,
    source_name
):

    print(
        f"\nCalculating FULL {source_name} token frequencies..."
    )

    token_counter = Counter()

    chunk_number = 0

    total_rows = 0

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        usecols=[
            "business_name"
        ],
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        total_rows += len(chunk)

        names = (
            chunk["business_name"]
            .fillna("")
            .map(normalize_text)
        )

        for name in names:

            tokens = set(
                get_tokens(name)
            )

            for token in tokens:

                token_counter[token] += 1

        if chunk_number % 5 == 0:

            print(
                f"  Processed {total_rows:,} rows..."
            )


    print(
        f"  Finished {total_rows:,} rows."
    )

    print(
        f"  Unique tokens: {len(token_counter):,}"
    )

    return token_counter


s2_token_frequency = (
    calculate_full_token_frequency(
        S2_FILE,
        "S2"
    )
)


s3_token_frequency = (
    calculate_full_token_frequency(
        S3_FILE,
        "S3"
    )
)


# ============================================================
# RARE TOKEN SUMMARY
# ============================================================

def rare_token_summary(
    token_frequency
):

    rare = [

        token

        for token, frequency
        in token_frequency.items()

        if (
            frequency <=
            RARE_TOKEN_MAX_DF
        )
    ]

    return rare


s2_rare_tokens = rare_token_summary(
    s2_token_frequency
)

s3_rare_tokens = rare_token_summary(
    s3_token_frequency
)


print(
    f"\nS2 rare tokens <= {RARE_TOKEN_MAX_DF}: "
    f"{len(s2_rare_tokens):,}"
)

print(
    f"S3 rare tokens <= {RARE_TOKEN_MAX_DF}: "
    f"{len(s3_rare_tokens):,}"
)


# ============================================================
# BUILD FULL-DATASET INDEXES
# ============================================================

def add_to_index(
    index,
    key,
    entity_id
):

    if not key:
        return

    index[key].add(
        entity_id
    )


def process_source_for_indexes(
    filepath,
    source_name,
    token_frequency
):

    print(
        f"\nBuilding FULL {source_name} blocking indexes..."
    )

    prefix5_index = defaultdict(set)

    address8_index = defaultdict(set)

    first_token_index = defaultdict(set)

    rare_token_index = defaultdict(set)

    chunk_number = 0

    total_rows = 0


    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        total_rows += len(chunk)


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        chunk["country_norm"] = (
            chunk["country"]
            .map(normalize_text)
        )

        chunk["name_norm"] = (
            chunk["business_name"]
            .map(normalize_text)
        )

        chunk["address_norm"] = (
            chunk["business_address"]
            .map(normalize_text)
        )


        # ----------------------------------------------------
        # CREATE BLOCKING FEATURES
        # ----------------------------------------------------

        chunk["name_compact"] = (
            chunk["name_norm"]
            .map(compact_text)
        )

        chunk["name_prefix_5"] = (
            chunk["name_compact"]
            .map(
                lambda x: prefix(x, 5)
            )
        )

        chunk["address_compact"] = (
            chunk["address_norm"]
            .map(compact_text)
        )

        chunk["address_prefix_8"] = (
            chunk["address_compact"]
            .map(
                lambda x: prefix(x, 8)
            )
        )

        chunk["name_first_token"] = (
            chunk["name_norm"]
            .map(first_token)
        )


        # ----------------------------------------------------
        # BUILD INDEXES
        # ----------------------------------------------------

        for (
            country,
            entity_id,
            name_prefix,
            address_prefix,
            first_token_value,
            name
        ) in zip(

            chunk["country_norm"],

            chunk["entity_id"],

            chunk["name_prefix_5"],

            chunk["address_prefix_8"],

            chunk["name_first_token"],

            chunk["name_norm"]
        ):

            if not country:
                continue


            # -----------------------------------------------
            # NAME PREFIX 5
            # -----------------------------------------------

            if name_prefix:

                add_to_index(
                    prefix5_index,
                    (
                        country,
                        name_prefix
                    ),
                    entity_id
                )


            # -----------------------------------------------
            # ADDRESS PREFIX 8
            # -----------------------------------------------

            if address_prefix:

                add_to_index(
                    address8_index,
                    (
                        country,
                        address_prefix
                    ),
                    entity_id
                )


            # -----------------------------------------------
            # FIRST TOKEN
            # -----------------------------------------------

            if first_token_value:

                add_to_index(
                    first_token_index,
                    (
                        country,
                        first_token_value
                    ),
                    entity_id
                )


            # -----------------------------------------------
            # RARE TOKENS
            # -----------------------------------------------

            tokens = set(
                get_tokens(name)
            )

            for token in tokens:

                frequency = (
                    token_frequency.get(
                        token,
                        0
                    )
                )

                if (
                    frequency > 0
                    and frequency <=
                    RARE_TOKEN_MAX_DF
                ):

                    add_to_index(
                        rare_token_index,
                        (
                            country,
                            token
                        ),
                        entity_id
                    )


        if chunk_number % 5 == 0:

            print(
                f"  Processed {total_rows:,} rows..."
            )


    print(
        f"  Finished {total_rows:,} rows."
    )

    print(
        f"  Prefix5 index keys: "
        f"{len(prefix5_index):,}"
    )

    print(
        f"  Address8 index keys: "
        f"{len(address8_index):,}"
    )

    print(
        f"  First-token index keys: "
        f"{len(first_token_index):,}"
    )

    print(
        f"  Rare-token index keys: "
        f"{len(rare_token_index):,}"
    )


    return {

        "prefix5": prefix5_index,

        "address8": address8_index,

        "firsttoken": first_token_index,

        "rare_token": rare_token_index
    }


# ============================================================
# BUILD S2 INDEXES
# ============================================================

s2_indexes = process_source_for_indexes(
    S2_FILE,
    "S2",
    s2_token_frequency
)


# ============================================================
# BUILD S3 INDEXES
# ============================================================

s3_indexes = process_source_for_indexes(
    S3_FILE,
    "S3",
    s3_token_frequency
)


# ============================================================
# GET RARE TOKENS FOR S1
# ============================================================

def get_s1_rare_tokens(
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
            and frequency <=
            RARE_TOKEN_MAX_DF
        ):

            candidates.append(
                (
                    frequency,
                    token
                )
            )


    candidates.sort()

    # Maximum two rare tokens per S1.
    return [
        token
        for _, token
        in candidates[:2]
    ]


# ============================================================
# EVALUATE FINAL BLOCKING STRATEGIES
# ============================================================

def evaluate_source(
    source_name,
    indexes,
    token_frequency
):

    print(
        f"\nEvaluating final blocking for {source_name}..."
    )


    strategies = {

        "prefix5": [
            "prefix5"
        ],

        "prefix5_address8": [
            "prefix5",
            "address8"
        ],

        "prefix5_firsttoken_address8": [
            "prefix5",
            "firsttoken",
            "address8"
        ],

        "prefix5_address8_rare": [
            "prefix5",
            "address8",
            "rare_token"
        ],

        "prefix5_firsttoken_address8_rare": [
            "prefix5",
            "firsttoken",
            "address8",
            "rare_token"
        ]
    }


    results = []


    for strategy_name, blocks in strategies.items():

        print(
            f"  Testing {strategy_name}..."
        )


        total_true = 0

        recovered = 0

        candidate_counts = []

        zero_candidate_count = 0


        for _, row in s1_eval.iterrows():

            s1_id = row[
                "entity_id"
            ]


            # ------------------------------------------------
            # TRUE MATCHES
            # ------------------------------------------------

            true_ids = {

                entity_id

                for entity_id
                in true_map.get(
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
            # GENERATE UNION CANDIDATES
            # ------------------------------------------------

            candidates = set()


            # PREFIX 5
            if "prefix5" in blocks:

                key = (
                    row["country_norm"],
                    row["name_prefix_5"]
                )

                candidates.update(
                    indexes[
                        "prefix5"
                    ].get(
                        key,
                        set()
                    )
                )


            # ADDRESS PREFIX 8
            if "address8" in blocks:

                key = (
                    row["country_norm"],
                    row["address_prefix_8"]
                )

                candidates.update(
                    indexes[
                        "address8"
                    ].get(
                        key,
                        set()
                    )
                )


            # FIRST TOKEN
            if "firsttoken" in blocks:

                key = (
                    row["country_norm"],
                    row["name_first_token"]
                )

                candidates.update(
                    indexes[
                        "firsttoken"
                    ].get(
                        key,
                        set()
                    )
                )


            # RARE TOKEN
            if "rare_token" in blocks:

                rare_tokens = (
                    get_s1_rare_tokens(
                        row["name_tokens"],
                        token_frequency
                    )
                )


                for token in rare_tokens:

                    key = (
                        row["country_norm"],
                        token
                    )

                    candidates.update(
                        indexes[
                            "rare_token"
                        ].get(
                            key,
                            set()
                        )
                    )


            # ------------------------------------------------
            # CANDIDATE STATISTICS
            # ------------------------------------------------

            candidate_count = len(
                candidates
            )

            candidate_counts.append(
                candidate_count
            )


            if candidate_count == 0:

                zero_candidate_count += 1


            # ------------------------------------------------
            # RECOVER TRUE MATCHES
            # ------------------------------------------------

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

            np.mean(
                candidate_counts
            )

            if candidate_counts

            else 0
        )


        median_candidates = (

            np.median(
                candidate_counts
            )

            if candidate_counts

            else 0
        )


        max_candidates = (

            max(
                candidate_counts
            )

            if candidate_counts

            else 0
        )


        zero_candidate_pct = (

            zero_candidate_count
            / len(s1_eval)
            * 100

            if len(s1_eval) > 0

            else 0
        )


        results.append({

            "source": source_name,

            "strategy": strategy_name,

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
            ),

            "zero_candidate_s1": (
                zero_candidate_count
            ),

            "zero_candidate_%": round(
                zero_candidate_pct,
                2
            )
        })


    return results


# ============================================================
# RUN S2
# ============================================================

results = []

results.extend(
    evaluate_source(
        "S2",
        s2_indexes,
        s2_token_frequency
    )
)


# ============================================================
# RUN S3
# ============================================================

results.extend(
    evaluate_source(
        "S3",
        s3_indexes,
        s3_token_frequency
    )
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n===== STEP 21 RESULTS =====\n"
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
    "\n===== STEP 21 COMPLETE =====\n"
)