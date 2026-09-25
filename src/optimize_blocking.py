import pandas as pd
import numpy as np
import re
import unicodedata

from collections import defaultdict


print("\n===== STEP 22: CANDIDATE VOLUME OPTIMIZATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "dataset/train"

S1_FILE = f"{TRAIN_DIR}/train_source1.tsv"
S2_FILE = f"{TRAIN_DIR}/train_source2.tsv"
S3_FILE = f"{TRAIN_DIR}/train_source3.tsv"
GT_FILE = f"{TRAIN_DIR}/train_ground_truth.tsv"

OUTPUT_FILE = "output/blocking_optimization.tsv"

GT_SAMPLE_SIZE = 10000

CHUNK_SIZE = 250000

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


def prefix(value, n):

    if not value:
        return ""

    return value[:n]


def first_token(value):

    if not value:
        return ""

    parts = value.split()

    return parts[0] if parts else ""


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
# S1 EVALUATION DATA
# ============================================================

s1_eval = s1[
    s1["entity_id"].isin(
        true_map.keys()
    )
].copy()


def prepare_s1_features(df):

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

    # --------------------------------------------------------
    # NAME PREFIXES
    # --------------------------------------------------------

    for n in [5, 7, 9, 11]:

        df[
            f"name_prefix_{n}"
        ] = (
            df["name_compact"]
            .map(
                lambda x, n=n:
                prefix(x, n)
            )
        )

    # --------------------------------------------------------
    # ADDRESS PREFIXES
    # --------------------------------------------------------

    for n in [8, 10, 12]:

        df[
            f"address_prefix_{n}"
        ] = (
            df["address_compact"]
            .map(
                lambda x, n=n:
                prefix(x, n)
            )
        )

    # --------------------------------------------------------
    # FIRST TOKEN
    # --------------------------------------------------------

    df["name_first_token"] = (
        df["name_norm"]
        .map(first_token)
    )

    return df


print("\nPreparing S1 features...")

s1_eval = prepare_s1_features(
    s1_eval
)


# ============================================================
# INDEX BUILDER
# ============================================================

def build_index_for_column(
    filepath,
    column_source,
    source_name
):

    print(
        f"\nBuilding {source_name} index for {column_source}..."
    )

    index = defaultdict(set)

    total_rows = 0

    chunk_number = 0

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]


    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        usecols=usecols,
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

        chunk["name_compact"] = (
            chunk["name_norm"]
            .map(compact_text)
        )

        chunk["address_compact"] = (
            chunk["address_norm"]
            .map(compact_text)
        )


        # ----------------------------------------------------
        # SELECT KEY
        # ----------------------------------------------------

        if column_source.startswith(
            "name_prefix_"
        ):

            n = int(
                column_source.split("_")[-1]
            )

            chunk["block_key"] = (
                chunk["name_compact"]
                .map(
                    lambda x, n=n:
                    prefix(x, n)
                )
            )


        elif column_source.startswith(
            "address_prefix_"
        ):

            n = int(
                column_source.split("_")[-1]
            )

            chunk["block_key"] = (
                chunk["address_compact"]
                .map(
                    lambda x, n=n:
                    prefix(x, n)
                )
            )


        elif column_source == "name_first_token":

            chunk["block_key"] = (
                chunk["name_norm"]
                .map(first_token)
            )


        else:

            raise ValueError(
                f"Unknown blocking column: "
                f"{column_source}"
            )


        # ----------------------------------------------------
        # BUILD INDEX
        # ----------------------------------------------------

        for country, key, entity_id in zip(

            chunk["country_norm"],

            chunk["block_key"],

            chunk["entity_id"]
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
            ].add(
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
        f"  Index keys: {len(index):,}"
    )

    return index


# ============================================================
# STRATEGIES
# ============================================================

strategies = [

    # -----------------------------------------------
    # Name only
    # -----------------------------------------------

    "name_prefix_7",

    "name_prefix_9",

    "name_prefix_11",

    # -----------------------------------------------
    # Name + address
    # -----------------------------------------------

    "name7_address8",

    "name7_address10",

    "name7_address12",

    "name9_address8",

    "name9_address10",

    "name9_address12",

    "name11_address8",

    "name11_address10",

    "name11_address12",

    # -----------------------------------------------
    # Name + first token + address
    # -----------------------------------------------

    "name7_first_address10",

    "name9_first_address10",

    "name9_first_address12",

    "name11_first_address10",

    "name11_first_address12",
]


# ============================================================
# DETERMINE REQUIRED INDEXES
# ============================================================

required_indexes = set()

for strategy in strategies:

    if strategy.startswith(
        "name_prefix_"
    ):

        required_indexes.add(
            strategy
        )

    elif strategy.startswith(
        "name7_address"
    ):

        required_indexes.add(
            "name_prefix_7"
        )

        required_indexes.add(
            "address_prefix_"
            + strategy.replace(
                "name7_address",
                ""
            )
        )


    elif strategy.startswith(
        "name9_address"
    ):

        required_indexes.add(
            "name_prefix_9"
        )

        required_indexes.add(
            "address_prefix_"
            + strategy.replace(
                "name9_address",
                ""
            )
        )


    elif strategy.startswith(
        "name11_address"
    ):

        required_indexes.add(
            "name_prefix_11"
        )

        required_indexes.add(
            "address_prefix_"
            + strategy.replace(
                "name11_address",
                ""
            )
        )


    elif strategy.startswith(
        "name7_first_address"
    ):

        required_indexes.add(
            "name_prefix_7"
        )

        required_indexes.add(
            "address_prefix_"
            + strategy.replace(
                "name7_first_address",
                ""
            )
        )

        required_indexes.add(
            "name_first_token"
        )


    elif strategy.startswith(
        "name9_first_address"
    ):

        required_indexes.add(
            "name_prefix_9"
        )

        required_indexes.add(
            "address_prefix_"
            + strategy.replace(
                "name9_first_address",
                ""
            )
        )

        required_indexes.add(
            "name_first_token"
        )


    elif strategy.startswith(
        "name11_first_address"
    ):

        required_indexes.add(
            "name_prefix_11"
        )

        required_indexes.add(
            "address_prefix_"
            + strategy.replace(
                "name11_first_address",
                ""
            )
        )

        required_indexes.add(
            "name_first_token"
        )


# ============================================================
# BUILD INDEXES
# ============================================================

def build_all_indexes(
    filepath,
    source_name
):

    indexes = {}

    for column in sorted(
        required_indexes
    ):

        indexes[column] = (
            build_index_for_column(
                filepath,
                column,
                source_name
            )
        )

    return indexes


print("\n===== BUILDING S2 INDEXES =====")

s2_indexes = build_all_indexes(
    S2_FILE,
    "S2"
)


print("\n===== BUILDING S3 INDEXES =====")

s3_indexes = build_all_indexes(
    S3_FILE,
    "S3"
)


# ============================================================
# GET CANDIDATES FOR ONE STRATEGY
# ============================================================

def get_candidates(
    row,
    indexes,
    strategy
):

    country = row["country_norm"]

    if not country:
        return set()


    candidates = set()


    # ========================================================
    # NAME PREFIX ONLY
    # ========================================================

    if strategy in [
        "name_prefix_7",
        "name_prefix_9",
        "name_prefix_11"
    ]:

        n = int(
            strategy.split("_")[-1]
        )

        key = row[
            f"name_prefix_{n}"
        ]

        if key:

            candidates.update(
                indexes[
                    f"name_prefix_{n}"
                ].get(
                    (
                        country,
                        key
                    ),
                    set()
                )
            )

        return candidates


    # ========================================================
    # NAME + ADDRESS
    # ========================================================

    match = re.match(
        r"name(7|9|11)_address(8|10|12)$",
        strategy
    )

    if match:

        name_n = int(
            match.group(1)
        )

        address_n = int(
            match.group(2)
        )

        name_key = row[
            f"name_prefix_{name_n}"
        ]

        address_key = row[
            f"address_prefix_{address_n}"
        ]


        if name_key:

            candidates.update(
                indexes[
                    f"name_prefix_{name_n}"
                ].get(
                    (
                        country,
                        name_key
                    ),
                    set()
                )
            )


        if address_key:

            candidates.update(
                indexes[
                    f"address_prefix_{address_n}"
                ].get(
                    (
                        country,
                        address_key
                    ),
                    set()
                )
            )


        return candidates


    # ========================================================
    # NAME + FIRST TOKEN + ADDRESS
    # ========================================================

    match = re.match(
        r"name(7|9|11)_first_address(10|12)$",
        strategy
    )

    if match:

        name_n = int(
            match.group(1)
        )

        address_n = int(
            match.group(2)
        )


        # Name prefix
        name_key = row[
            f"name_prefix_{name_n}"
        ]

        if name_key:

            candidates.update(
                indexes[
                    f"name_prefix_{name_n}"
                ].get(
                    (
                        country,
                        name_key
                    ),
                    set()
                )
            )


        # First token
        first_key = (
            row["name_first_token"]
        )

        if first_key:

            candidates.update(
                indexes[
                    "name_first_token"
                ].get(
                    (
                        country,
                        first_key
                    ),
                    set()
                )
            )


        # Address
        address_key = row[
            f"address_prefix_{address_n}"
        ]

        if address_key:

            candidates.update(
                indexes[
                    f"address_prefix_{address_n}"
                ].get(
                    (
                        country,
                        address_key
                    ),
                    set()
                )
            )


        return candidates


    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# EVALUATE SOURCE
# ============================================================

def evaluate_source(
    source_name,
    indexes
):

    print(
        f"\n===== EVALUATING {source_name} ====="
    )

    results = []


    for strategy in strategies:

        print(
            f"  Testing {strategy}..."
        )


        total_true = 0

        recovered = 0

        candidate_counts = []

        zero_candidates = 0


        for _, row in s1_eval.iterrows():

            s1_id = row[
                "entity_id"
            ]


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


            candidates = get_candidates(
                row,
                indexes,
                strategy
            )


            candidate_count = len(
                candidates
            )

            candidate_counts.append(
                candidate_count
            )


            if candidate_count == 0:

                zero_candidates += 1


            recovered += len(
                true_ids.intersection(
                    candidates
                )
            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

            zero_candidates
            / len(s1_eval)
            * 100

            if len(s1_eval) > 0

            else 0
        )


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

            "max_candidates": int(
                max_candidates
            ),

            "zero_candidate_s1": (
                zero_candidates
            ),

            "zero_candidate_%": round(
                zero_candidate_pct,
                2
            )
        })


    return results


# ============================================================
# RUN
# ============================================================

results = []


results.extend(
    evaluate_source(
        "S2",
        s2_indexes
    )
)


results.extend(
    evaluate_source(
        "S3",
        s3_indexes
    )
)


# ============================================================
# DISPLAY
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n===== STEP 22 RESULTS =====\n"
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
    "\n===== STEP 22 COMPLETE =====\n"
)