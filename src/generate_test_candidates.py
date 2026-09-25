import pandas as pd
import re
import unicodedata
import os


print("\n===== STEP 23: FAST TEST CANDIDATE GENERATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"

S1_FILE = f"{TEST_DIR}/test_source1.tsv"
S2_FILE = f"{TEST_DIR}/test_source2.tsv"
S3_FILE = f"{TEST_DIR}/test_source3.tsv"

OUTPUT_FILE = "output/candidate_pairs.tsv"

S1_CHUNK_SIZE = 100000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_series(series):

    return (
        series
        .fillna("")
        .astype(str)
        .map(
            lambda x: unicodedata.normalize(
                "NFKC",
                x
            )
        )
        .str.casefold()
        .str.replace(
            r"[^\w\s]",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


def compact_series(series):

    return series.str.replace(
        " ",
        "",
        regex=False
    )


def prepare_source(df):

    df = df[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    df["country_norm"] = normalize_series(
        df["country"]
    )

    df["name_norm"] = normalize_series(
        df["business_name"]
    )

    df["address_norm"] = normalize_series(
        df["business_address"]
    )

    df["name_compact"] = compact_series(
        df["name_norm"]
    )

    df["address_compact"] = compact_series(
        df["address_norm"]
    )

    df["name_prefix_5"] = (
        df["name_compact"]
        .str[:5]
    )

    df["address_prefix_8"] = (
        df["address_compact"]
        .str[:8]
    )

    return df


# ============================================================
# LOAD S2 / S3
# ============================================================

print("Loading Test S2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str
)

print(
    f"S2 records: {len(s2):,}"
)

print("Normalizing S2...")

s2 = prepare_source(s2)


print("\nLoading Test S3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str
)

print(
    f"S3 records: {len(s3):,}"
)

print("Normalizing S3...")

s3 = prepare_source(s3)


# ============================================================
# KEEP ONLY BLOCKING COLUMNS
# ============================================================

s2_blocks = s2[
    [
        "entity_id",
        "country_norm",
        "name_prefix_5",
        "address_prefix_8"
    ]
].copy()

s3_blocks = s3[
    [
        "entity_id",
        "country_norm",
        "name_prefix_5",
        "address_prefix_8"
    ]
].copy()


s2_blocks = s2_blocks.rename(
    columns={
        "entity_id": "candidate_entity_id"
    }
)

s3_blocks = s3_blocks.rename(
    columns={
        "entity_id": "candidate_entity_id"
    }
)


# ============================================================
# REMOVE EMPTY BLOCK KEYS
# ============================================================

s2_name = s2_blocks[
    (
        s2_blocks["country_norm"] != ""
    )
    &
    (
        s2_blocks["name_prefix_5"] != ""
    )
][
    [
        "candidate_entity_id",
        "country_norm",
        "name_prefix_5"
    ]
].drop_duplicates()


s2_address = s2_blocks[
    (
        s2_blocks["country_norm"] != ""
    )
    &
    (
        s2_blocks["address_prefix_8"] != ""
    )
][
    [
        "candidate_entity_id",
        "country_norm",
        "address_prefix_8"
    ]
].drop_duplicates()


s3_name = s3_blocks[
    (
        s3_blocks["country_norm"] != ""
    )
    &
    (
        s3_blocks["name_prefix_5"] != ""
    )
][
    [
        "candidate_entity_id",
        "country_norm",
        "name_prefix_5"
    ]
].drop_duplicates()


s3_address = s3_blocks[
    (
        s3_blocks["country_norm"] != ""
    )
    &
    (
        s3_blocks["address_prefix_8"] != ""
    )
][
    [
        "candidate_entity_id",
        "country_norm",
        "address_prefix_8"
    ]
].drop_duplicates()


print("\nBlocking tables prepared.")


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if os.path.exists(
    OUTPUT_FILE
):

    os.remove(
        OUTPUT_FILE
    )


first_write = True

total_candidates = 0

processed_s1 = 0


# ============================================================
# PROCESS S1 IN CHUNKS
# ============================================================

print(
    "\nGenerating candidate pairs..."
)


for s1_chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=S1_CHUNK_SIZE
):

    processed_s1 += len(
        s1_chunk
    )


    # --------------------------------------------------------
    # Normalize S1
    # --------------------------------------------------------

    s1_chunk = prepare_source(
        s1_chunk
    )


    s1_blocks = s1_chunk[
        [
            "entity_id",
            "country_norm",
            "name_prefix_5",
            "address_prefix_8"
        ]
    ].copy()


    # ========================================================
    # S2 NAME BLOCK
    # ========================================================

    name_pairs_s2 = s1_blocks.merge(
        s2_name,
        on=[
            "country_norm",
            "name_prefix_5"
        ],
        how="inner"
    )[
        [
            "entity_id",
            "candidate_entity_id"
        ]
    ]

    name_pairs_s2[
        "candidate_source"
    ] = "S2"


    # ========================================================
    # S2 ADDRESS BLOCK
    # ========================================================

    address_pairs_s2 = s1_blocks.merge(
        s2_address,
        on=[
            "country_norm",
            "address_prefix_8"
        ],
        how="inner"
    )[
        [
            "entity_id",
            "candidate_entity_id"
        ]
    ]

    address_pairs_s2[
        "candidate_source"
    ] = "S2"


    # ========================================================
    # S3 NAME BLOCK
    # ========================================================

    name_pairs_s3 = s1_blocks.merge(
        s3_name,
        on=[
            "country_norm",
            "name_prefix_5"
        ],
        how="inner"
    )[
        [
            "entity_id",
            "candidate_entity_id"
        ]
    ]

    name_pairs_s3[
        "candidate_source"
    ] = "S3"


    # ========================================================
    # S3 ADDRESS BLOCK
    # ========================================================

    address_pairs_s3 = s1_blocks.merge(
        s3_address,
        on=[
            "country_norm",
            "address_prefix_8"
        ],
        how="inner"
    )[
        [
            "entity_id",
            "candidate_entity_id"
        ]
    ]

    address_pairs_s3[
        "candidate_source"
    ] = "S3"


    # ========================================================
    # COMBINE
    # ========================================================

    chunk_pairs = pd.concat(
        [
            name_pairs_s2,
            address_pairs_s2,
            name_pairs_s3,
            address_pairs_s3
        ],
        ignore_index=True
    )


    # Rename S1 ID
    chunk_pairs = chunk_pairs.rename(
        columns={
            "entity_id":
                "source1_entity_id"
        }
    )


    # --------------------------------------------------------
    # Remove duplicate candidate pairs
    # --------------------------------------------------------

    chunk_pairs = chunk_pairs.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )


    total_candidates += len(
        chunk_pairs
    )


    # ========================================================
    # WRITE
    # ========================================================

    chunk_pairs.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write
    )

    first_write = False


    # ========================================================
    # PROGRESS
    # ========================================================

    print(
        f"Processed S1: "
        f"{processed_s1:,}"
    )

    print(
        f"Candidates generated so far: "
        f"{total_candidates:,}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n===== STEP 23 RESULTS =====\n"
)

print(
    f"Test S1 entities: "
    f"{processed_s1:,}"
)

print(
    f"Candidate pairs generated: "
    f"{total_candidates:,}"
)

print(
    f"Average candidates/S1: "
    f"{total_candidates / processed_s1:.2f}"
)

print(
    f"\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "\n===== STEP 23 COMPLETE =====\n"
)