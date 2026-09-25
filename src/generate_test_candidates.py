import pandas as pd
import re
import unicodedata
import os


print("\n===== STEP 23: MEMORY-SAFE TEST CANDIDATE GENERATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"

S1_FILE = f"{TEST_DIR}/test_source1.tsv"
S2_FILE = f"{TEST_DIR}/test_source2.tsv"
S3_FILE = f"{TEST_DIR}/test_source3.tsv"

OUTPUT_FILE = "output/candidate_pairs.tsv"

# Much smaller than before.
S1_CHUNK_SIZE = 5000


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

    df["name_compact"] = (
        df["name_norm"]
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    df["address_compact"] = (
        df["address_norm"]
        .str.replace(
            " ",
            "",
            regex=False
        )
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
# LOAD TEST S2
# ============================================================

print("Loading Test S2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print(
    f"S2 records: {len(s2):,}"
)

print("Normalizing S2...")

s2 = prepare_source(s2)


# ============================================================
# LOAD TEST S3
# ============================================================

print("\nLoading Test S3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print(
    f"S3 records: {len(s3):,}"
)

print("Normalizing S3...")

s3 = prepare_source(s3)


# ============================================================
# CREATE SMALL BLOCKING TABLES
# ============================================================

print("\nPreparing blocking tables...")


s2_name = s2[
    [
        "entity_id",
        "country_norm",
        "name_prefix_5"
    ]
].rename(
    columns={
        "entity_id": "candidate_entity_id"
    }
)

s2_name = s2_name[
    (s2_name["country_norm"] != "")
    &
    (s2_name["name_prefix_5"] != "")
]


s2_address = s2[
    [
        "entity_id",
        "country_norm",
        "address_prefix_8"
    ]
].rename(
    columns={
        "entity_id": "candidate_entity_id"
    }
)

s2_address = s2_address[
    (s2_address["country_norm"] != "")
    &
    (s2_address["address_prefix_8"] != "")
]


s3_name = s3[
    [
        "entity_id",
        "country_norm",
        "name_prefix_5"
    ]
].rename(
    columns={
        "entity_id": "candidate_entity_id"
    }
)

s3_name = s3_name[
    (s3_name["country_norm"] != "")
    &
    (s3_name["name_prefix_5"] != "")
]


s3_address = s3[
    [
        "entity_id",
        "country_norm",
        "address_prefix_8"
    ]
].rename(
    columns={
        "entity_id": "candidate_entity_id"
    }
)

s3_address = s3_address[
    (s3_address["country_norm"] != "")
    &
    (s3_address["address_prefix_8"] != "")
]


print("Blocking tables ready.")


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

processed_s1 = 0

total_pairs = 0


# ============================================================
# PROCESS S1 IN SMALL CHUNKS
# ============================================================

print(
    "\nGenerating candidates in small chunks..."
)


for s1_chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    chunksize=S1_CHUNK_SIZE
):

    # --------------------------------------------------------
    # Normalize only this S1 chunk
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
    ]


    # ========================================================
    # S2 NAME
    # ========================================================

    pairs = s1_blocks.merge(
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

    if not pairs.empty:

        pairs = pairs.rename(
            columns={
                "entity_id":
                    "source1_entity_id"
            }
        )

        pairs["candidate_source"] = "S2"

        pairs.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        total_pairs += len(pairs)


    del pairs


    # ========================================================
    # S2 ADDRESS
    # ========================================================

    pairs = s1_blocks.merge(
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

    if not pairs.empty:

        pairs = pairs.rename(
            columns={
                "entity_id":
                    "source1_entity_id"
            }
        )

        pairs["candidate_source"] = "S2"

        pairs.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=False
        )

        total_pairs += len(pairs)


    del pairs


    # ========================================================
    # S3 NAME
    # ========================================================

    pairs = s1_blocks.merge(
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

    if not pairs.empty:

        pairs = pairs.rename(
            columns={
                "entity_id":
                    "source1_entity_id"
            }
        )

        pairs["candidate_source"] = "S3"

        pairs.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=False
        )

        total_pairs += len(pairs)


    del pairs


    # ========================================================
    # S3 ADDRESS
    # ========================================================

    pairs = s1_blocks.merge(
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

    if not pairs.empty:

        pairs = pairs.rename(
            columns={
                "entity_id":
                    "source1_entity_id"
            }
        )

        pairs["candidate_source"] = "S3"

        pairs.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=False
        )

        total_pairs += len(pairs)


    del pairs


    # ========================================================
    # PROGRESS
    # ========================================================

    processed_s1 += len(
        s1_chunk
    )

    if (
        processed_s1 % 50000 == 0
        or processed_s1 == len(s1)
    ):

        print(
            f"Processed S1: "
            f"{processed_s1:,} / "
            f"{len(s1):,}"
        )

        print(
            f"Candidate rows written: "
            f"{total_pairs:,}"
        )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n===== STEP 23 RESULTS =====\n"
)

print(
    f"Test S1 entities: "
    f"{processed_s1:,}"
)

print(
    f"Candidate rows generated: "
    f"{total_pairs:,}"
)

print(
    f"\nSaved: {OUTPUT_FILE}"
)

print(
    "\n===== STEP 23 COMPLETE =====\n"
)
