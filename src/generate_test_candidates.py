import pandas as pd
import re
import unicodedata
from collections import defaultdict


print("\n===== STEP 23: TEST CANDIDATE GENERATION =====\n")


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"
OUTPUT_FILE = "output/candidate_pairs.tsv"

S1_FILE = f"{TEST_DIR}/test_source1.tsv"
S2_FILE = f"{TEST_DIR}/test_source2.tsv"
S3_FILE = f"{TEST_DIR}/test_source3.tsv"

CHUNK_SIZE = 250000


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


# ============================================================
# PREPARE S1
# ============================================================

print("Loading Test Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

print(
    f"S1 records: {len(s1):,}"
)


s1["country_norm"] = (
    s1["country"]
    .map(normalize_text)
)

s1["name_norm"] = (
    s1["business_name"]
    .map(normalize_text)
)

s1["address_norm"] = (
    s1["business_address"]
    .map(normalize_text)
)

s1["name_compact"] = (
    s1["name_norm"]
    .map(compact_text)
)

s1["address_compact"] = (
    s1["address_norm"]
    .map(compact_text)
)

s1["name_prefix_5"] = (
    s1["name_compact"]
    .map(
        lambda x: prefix(x, 5)
    )
)

s1["address_prefix_8"] = (
    s1["address_compact"]
    .map(
        lambda x: prefix(x, 8)
    )
)


# ============================================================
# BUILD INDEX FOR ONE SOURCE
# ============================================================

def build_indexes(
    filepath,
    source_name
):

    print(
        f"\nBuilding {source_name} indexes..."
    )

    name_index = defaultdict(set)

    address_index = defaultdict(set)

    total_rows = 0

    chunk_number = 0


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

        chunk["name_compact"] = (
            chunk["name_norm"]
            .map(compact_text)
        )

        chunk["address_compact"] = (
            chunk["address_norm"]
            .map(compact_text)
        )

        chunk["name_prefix_5"] = (
            chunk["name_compact"]
            .map(
                lambda x: prefix(x, 5)
            )
        )

        chunk["address_prefix_8"] = (
            chunk["address_compact"]
            .map(
                lambda x: prefix(x, 8)
            )
        )


        # ----------------------------------------------------
        # INDEX
        # ----------------------------------------------------

        for (
            country,
            entity_id,
            name_key,
            address_key
        ) in zip(

            chunk["country_norm"],

            chunk["entity_id"],

            chunk["name_prefix_5"],

            chunk["address_prefix_8"]
        ):

            if not country:
                continue


            if name_key:

                name_index[
                    (
                        country,
                        name_key
                    )
                ].add(
                    entity_id
                )


            if address_key:

                address_index[
                    (
                        country,
                        address_key
                    )
                ].add(
                    entity_id
                )


        if chunk_number % 5 == 0:

            print(
                f"  Processed "
                f"{total_rows:,} rows..."
            )


    print(
        f"  Finished "
        f"{total_rows:,} rows."
    )

    print(
        f"  Name index keys: "
        f"{len(name_index):,}"
    )

    print(
        f"  Address index keys: "
        f"{len(address_index):,}"
    )


    return (
        name_index,
        address_index
    )


# ============================================================
# BUILD S2 INDEXES
# ============================================================

s2_name_index, s2_address_index = (
    build_indexes(
        S2_FILE,
        "S2"
    )
)


# ============================================================
# BUILD S3 INDEXES
# ============================================================

s3_name_index, s3_address_index = (
    build_indexes(
        S3_FILE,
        "S3"
    )
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    row,
    name_index,
    address_index
):

    country = row["country_norm"]

    if not country:
        return set()


    candidates = set()


    # --------------------------------------------------------
    # NAME BLOCK
    # --------------------------------------------------------

    name_key = row[
        "name_prefix_5"
    ]

    if name_key:

        candidates.update(
            name_index.get(
                (
                    country,
                    name_key
                ),
                set()
            )
        )


    # --------------------------------------------------------
    # ADDRESS BLOCK
    # --------------------------------------------------------

    address_key = row[
        "address_prefix_8"
    ]

    if address_key:

        candidates.update(
            address_index.get(
                (
                    country,
                    address_key
                ),
                set()
            )
        )


    return candidates


# ============================================================
# OUTPUT SETUP
# ============================================================

print(
    "\nGenerating candidate pairs..."
)

output_columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source"
]


# Remove previous output if it exists.
import os

if os.path.exists(
    OUTPUT_FILE
):

    os.remove(
        OUTPUT_FILE
    )


first_write = True

total_s1 = 0

total_candidates = 0

zero_candidate_s1 = 0


# ============================================================
# PROCESS TEST S1
# ============================================================

rows_buffer = []

BUFFER_SIZE = 100000


for _, row in s1.iterrows():

    total_s1 += 1


    # --------------------------------------------------------
    # S2 candidates
    # --------------------------------------------------------

    s2_candidates = generate_candidates(
        row,
        s2_name_index,
        s2_address_index
    )


    # --------------------------------------------------------
    # S3 candidates
    # --------------------------------------------------------

    s3_candidates = generate_candidates(
        row,
        s3_name_index,
        s3_address_index
    )


    total_for_s1 = (
        len(s2_candidates)
        +
        len(s3_candidates)
    )


    if total_for_s1 == 0:

        zero_candidate_s1 += 1


    # --------------------------------------------------------
    # Store S2
    # --------------------------------------------------------

    for candidate_id in s2_candidates:

        rows_buffer.append({

            "source1_entity_id":
                row["entity_id"],

            "candidate_entity_id":
                candidate_id,

            "candidate_source":
                "S2"
        })


    # --------------------------------------------------------
    # Store S3
    # --------------------------------------------------------

    for candidate_id in s3_candidates:

        rows_buffer.append({

            "source1_entity_id":
                row["entity_id"],

            "candidate_entity_id":
                candidate_id,

            "candidate_source":
                "S3"
        })


    total_candidates += (
        total_for_s1
    )


    # --------------------------------------------------------
    # Write buffer
    # --------------------------------------------------------

    if len(rows_buffer) >= BUFFER_SIZE:

        output_df = pd.DataFrame(
            rows_buffer,
            columns=output_columns
        )

        output_df.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        rows_buffer = []


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if total_s1 % 100000 == 0:

        print(
            f"  Processed S1: "
            f"{total_s1:,} / "
            f"{len(s1):,}"
        )

        print(
            f"  Candidates so far: "
            f"{total_candidates:,}"
        )


# ============================================================
# WRITE REMAINING BUFFER
# ============================================================

if rows_buffer:

    output_df = pd.DataFrame(
        rows_buffer,
        columns=output_columns
    )

    output_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

average_candidates = (
    total_candidates / total_s1
    if total_s1 > 0
    else 0
)

zero_candidate_pct = (
    zero_candidate_s1
    / total_s1
    * 100
    if total_s1 > 0
    else 0
)


print(
    "\n===== STEP 23 RESULTS =====\n"
)

print(
    f"Test S1 entities: "
    f"{total_s1:,}"
)

print(
    f"Total candidate pairs: "
    f"{total_candidates:,}"
)

print(
    f"Average candidates/S1: "
    f"{average_candidates:.2f}"
)

print(
    f"S1 with zero candidates: "
    f"{zero_candidate_s1:,}"
)

print(
    f"Zero-candidate percentage: "
    f"{zero_candidate_pct:.2f}%"
)

print(
    f"\nSaved: {OUTPUT_FILE}"
)

print(
    "\n===== STEP 23 COMPLETE =====\n"
)