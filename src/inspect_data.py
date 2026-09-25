import pandas as pd


# ============================================================
# STEP 1-5: LOAD TRAINING DATA
# ============================================================

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


# ============================================================
# SOURCE 1
# ============================================================

print("\n===== SOURCE 1 =====")

print(s1.head())

print("Shape:", s1.shape)

print("Columns:", s1.columns.tolist())


# ============================================================
# SOURCE 2
# ============================================================

print("\n===== SOURCE 2 =====")

print(s2.head())

print("Shape:", s2.shape)

print("Columns:", s2.columns.tolist())


# ============================================================
# SOURCE 3
# ============================================================

print("\n===== SOURCE 3 =====")

print(s3.head())

print("Shape:", s3.shape)

print("Columns:", s3.columns.tolist())


# ============================================================
# GROUND TRUTH
# ============================================================

print("\n===== GROUND TRUTH =====")

print(gt.head())

print("Shape:", gt.shape)

print("Columns:", gt.columns.tolist())


# ============================================================
# STEP 5: MISSING VALUE CHECK
# ============================================================

print("\n===== MISSING VALUES =====")

print("\nSource 1:")
print(s1.isnull().sum())

print("\nSource 2:")
print(s2.isnull().sum())

print("\nSource 3:")
print(s3.isnull().sum())

print("\nGround Truth:")
print(gt.isnull().sum())


# ============================================================
# STEP 6: MATCH STATISTICS
# ============================================================

print("\n===== MATCH STATISTICS =====")

empty_matches = gt["matched_entity_ids"].isna().sum()

print(
    "Total Source 1 entities:",
    len(gt)
)

print(
    "Entities with no matches:",
    empty_matches
)

print(
    "Entities with matches:",
    len(gt) - empty_matches
)


# ============================================================
# STEP 7: MATCH COUNT DISTRIBUTION
# ============================================================

print("\n===== MATCH COUNT DISTRIBUTION =====")

match_counts = (
    gt["matched_entity_ids"]
    .fillna("")
    .apply(
        lambda x: len(
            [
                i
                for i in str(x).split(",")
                if i.strip()
            ]
        )
    )
)

print(
    "Minimum matches:",
    match_counts.min()
)

print(
    "Maximum matches:",
    match_counts.max()
)

print(
    "Average matches:",
    round(
        match_counts.mean(),
        2
    )
)

print(
    "Median matches:",
    match_counts.median()
)

print("\nMatch count frequencies:")

print(
    match_counts
    .value_counts()
    .sort_index()
    .head(20)
)


# ============================================================
# STEP 8: MATCH SOURCE DISTRIBUTION
# ============================================================

print("\n===== MATCH SOURCE DISTRIBUTION =====")


def classify_match_sources(value):
    """
    Classify a Source 1 entity as:
    - Both S2 and S3
    - Only S2
    - Only S3
    - No match
    - Unknown
    """

    if pd.isna(value) or str(value).strip() == "":
        return "No match"

    ids = [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]

    has_s2 = any(
        x.startswith("S2-")
        for x in ids
    )

    has_s3 = any(
        x.startswith("S3-")
        for x in ids
    )

    if has_s2 and has_s3:
        return "Both S2 and S3"

    elif has_s2:
        return "Only S2"

    elif has_s3:
        return "Only S3"

    else:
        return "Unknown"


match_source_type = gt[
    "matched_entity_ids"
].apply(
    classify_match_sources
)

print(
    match_source_type.value_counts()
)

print("\nPercentages:")

print(
    (
        match_source_type
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# STEP 9: TRAINING COUNTRY DISTRIBUTION
# ============================================================

print("\n===== COUNTRY DISTRIBUTION =====")

print("\nSource 1 countries:")
print(
    s1["country"].value_counts()
)

print("\nSource 2 countries:")
print(
    s2["country"].value_counts()
)

print("\nSource 3 countries:")
print(
    s3["country"].value_counts()
)


# ============================================================
# COUNTRY PERCENTAGES
# ============================================================

print("\n===== COUNTRY PERCENTAGES =====")

print("\nSource 1:")

print(
    (
        s1["country"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nSource 2:")

print(
    (
        s2["country"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nSource 3:")

print(
    (
        s3["country"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# STEP 10: LOAD TEST DATA
# ============================================================

test_s1 = pd.read_csv(
    "dataset/test/test_source1.tsv",
    sep="\t"
)

test_s2 = pd.read_csv(
    "dataset/test/test_source2.tsv",
    sep="\t"
)

test_s3 = pd.read_csv(
    "dataset/test/test_source3.tsv",
    sep="\t"
)


# ============================================================
# TEST DATA SHAPES
# ============================================================

print("\n===== TEST DATA SHAPES =====")

print(
    "Test Source 1:",
    test_s1.shape
)

print(
    "Test Source 2:",
    test_s2.shape
)

print(
    "Test Source 3:",
    test_s3.shape
)


# ============================================================
# TEST DATA COLUMNS
# ============================================================

print("\n===== TEST DATA COLUMNS =====")

print(
    "Test Source 1:",
    test_s1.columns.tolist()
)

print(
    "Test Source 2:",
    test_s2.columns.tolist()
)

print(
    "Test Source 3:",
    test_s3.columns.tolist()
)


# ============================================================
# TEST COUNTRY DISTRIBUTION
# ============================================================

print("\n===== TEST COUNTRY DISTRIBUTION =====")

print("\nTest Source 1 countries:")
print(
    test_s1["country"].value_counts()
)

print("\nTest Source 2 countries:")
print(
    test_s2["country"].value_counts()
)

print("\nTest Source 3 countries:")
print(
    test_s3["country"].value_counts()
)


# ============================================================
# TEST MISSING VALUES
# ============================================================

print("\n===== TEST MISSING VALUES =====")

print("\nTest Source 1:")
print(
    test_s1.isnull().sum()
)

print("\nTest Source 2:")
print(
    test_s2.isnull().sum()
)

print("\nTest Source 3:")
print(
    test_s3.isnull().sum()
)


# ============================================================
# STEP 11: ENTITY ID INTEGRITY CHECK
# ============================================================

print("\n===== ENTITY ID INTEGRITY CHECK =====")


# ------------------------------------------------------------
# Duplicate IDs
# ------------------------------------------------------------

print("\nDuplicate IDs:")

print(
    "Source 1:",
    s1["entity_id"].duplicated().sum()
)

print(
    "Source 2:",
    s2["entity_id"].duplicated().sum()
)

print(
    "Source 3:",
    s3["entity_id"].duplicated().sum()
)


# ------------------------------------------------------------
# ID Prefixes
# ------------------------------------------------------------

print("\nID Prefixes:")

print("\nSource 1:")

print(
    s1["entity_id"]
    .str.extract(r"^(S\d-)")[0]
    .value_counts()
)

print("\nSource 2:")

print(
    s2["entity_id"]
    .str.extract(r"^(S\d-)")[0]
    .value_counts()
)

print("\nSource 3:")

print(
    s3["entity_id"]
    .str.extract(r"^(S\d-)")[0]
    .value_counts()
)


# ------------------------------------------------------------
# Ground Truth Source 1 IDs
# ------------------------------------------------------------

print("\nGround Truth Source 1 ID Check:")

s1_id_set = set(
    s1["entity_id"]
)

gt_s1_ids = set(
    gt["source1_entity_id"]
)

missing_gt_s1 = (
    gt_s1_ids - s1_id_set
)

print(
    "Ground truth S1 IDs:",
    len(gt_s1_ids)
)

print(
    "Missing from Source 1:",
    len(missing_gt_s1)
)


# ------------------------------------------------------------
# Ground Truth matched IDs
# ------------------------------------------------------------

print("\nGround Truth Matched ID Check:")

s2_id_set = set(
    s2["entity_id"]
)

s3_id_set = set(
    s3["entity_id"]
)

all_external_ids = (
    s2_id_set | s3_id_set
)

matched_ids = set()

for value in gt["matched_entity_ids"].dropna():

    ids = [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]

    matched_ids.update(ids)


missing_matched_ids = (
    matched_ids - all_external_ids
)

print(
    "Unique matched IDs:",
    len(matched_ids)
)

print(
    "Missing from S2/S3:",
    len(missing_matched_ids)
)


# ------------------------------------------------------------
# Ground Truth matched ID prefixes
# ------------------------------------------------------------

print(
    "\nGround Truth Matched ID Prefixes:"
)

prefix_counts = {}

for entity_id in matched_ids:

    if entity_id.startswith("S2-"):

        prefix = "S2"

    elif entity_id.startswith("S3-"):

        prefix = "S3"

    else:

        prefix = "Unexpected"

    prefix_counts[prefix] = (
        prefix_counts.get(prefix, 0) + 1
    )

print(prefix_counts)


# ============================================================
# STEP 12: BUSINESS NAME QUALITY ANALYSIS
# ============================================================

print(
    "\n===== BUSINESS NAME QUALITY ANALYSIS ====="
)


def name_analysis(df, source_name):
    """
    Analyze business name quality.
    """

    names = df["business_name"]

    print(
        f"\n--- {source_name} ---"
    )

    print(
        "Missing names:",
        names.isna().sum()
    )

    empty_names = (
        names
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "Empty/whitespace names:",
        empty_names
    )

    unique_names = (
        names
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )

    print(
        "Unique names:",
        unique_names
    )

    print(
        "Total records:",
        len(names)
    )

    repeated_name_records = (
        len(names) - unique_names
    )

    print(
        "Repeated name records:",
        repeated_name_records
    )

    name_lengths = (
        names
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
    )

    print(
        "Minimum name length:",
        name_lengths.min()
    )

    print(
        "Maximum name length:",
        name_lengths.max()
    )

    print(
        "Average name length:",
        round(
            name_lengths.mean(),
            2
        )
    )

    print(
        "Median name length:",
        name_lengths.median()
    )

    ascii_names = (
        names
        .fillna("")
        .astype(str)
        .str.contains(
            r"^[\x00-\x7F]*$",
            regex=True
        )
        .sum()
    )

    non_ascii_names = (
        len(names) - ascii_names
    )

    print(
        "ASCII-only names:",
        ascii_names
    )

    print(
        "Names containing non-ASCII characters:",
        non_ascii_names
    )

    print("\nSample names:")

    print(
        names
        .dropna()
        .astype(str)
        .head(10)
        .to_string(index=False)
    )


name_analysis(
    s1,
    "TRAIN SOURCE 1"
)

name_analysis(
    s2,
    "TRAIN SOURCE 2"
)

name_analysis(
    s3,
    "TRAIN SOURCE 3"
)

name_analysis(
    test_s1,
    "TEST SOURCE 1"
)

name_analysis(
    test_s2,
    "TEST SOURCE 2"
)

name_analysis(
    test_s3,
    "TEST SOURCE 3"
)


# ============================================================
# STEP 13: BUSINESS ADDRESS QUALITY ANALYSIS
# ============================================================

print(
    "\n===== BUSINESS ADDRESS QUALITY ANALYSIS ====="
)


def address_analysis(df, source_name):
    """
    Analyze business address quality.
    """

    addresses = df["business_address"]

    print(
        f"\n--- {source_name} ---"
    )

    print(
        "Missing addresses:",
        addresses.isna().sum()
    )

    empty_addresses = (
        addresses
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "Empty/whitespace addresses:",
        empty_addresses
    )

    unique_addresses = (
        addresses
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )

    print(
        "Unique addresses:",
        unique_addresses
    )

    print(
        "Total records:",
        len(addresses)
    )

    repeated_address_records = (
        len(addresses) - unique_addresses
    )

    print(
        "Repeated address records:",
        repeated_address_records
    )

    address_lengths = (
        addresses
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
    )

    print(
        "Minimum address length:",
        address_lengths.min()
    )

    print(
        "Maximum address length:",
        address_lengths.max()
    )

    print(
        "Average address length:",
        round(
            address_lengths.mean(),
            2
        )
    )

    print(
        "Median address length:",
        address_lengths.median()
    )

    ascii_addresses = (
        addresses
        .fillna("")
        .astype(str)
        .str.contains(
            r"^[\x00-\x7F]*$",
            regex=True
        )
        .sum()
    )

    non_ascii_addresses = (
        len(addresses) - ascii_addresses
    )

    print(
        "ASCII-only addresses:",
        ascii_addresses
    )

    print(
        "Addresses containing non-ASCII characters:",
        non_ascii_addresses
    )

    print("\nSample addresses:")

    print(
        addresses
        .dropna()
        .astype(str)
        .head(10)
        .to_string(index=False)
    )


address_analysis(
    s1,
    "TRAIN SOURCE 1"
)

address_analysis(
    s2,
    "TRAIN SOURCE 2"
)

address_analysis(
    s3,
    "TRAIN SOURCE 3"
)

address_analysis(
    test_s1,
    "TEST SOURCE 1"
)

address_analysis(
    test_s2,
    "TEST SOURCE 2"
)

address_analysis(
    test_s3,
    "TEST SOURCE 3"
)


# ============================================================
# STEP 14: TRUE MATCH PATTERN ANALYSIS
# ============================================================

print(
    "\n===== TRUE MATCH PATTERN ANALYSIS ====="
)


# ------------------------------------------------------------
# Helper for safely handling missing values
# ------------------------------------------------------------

def clean_value(value):
    """
    Convert missing values to an empty string.

    This prevents NaN from becoming the literal string "nan".
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


# ------------------------------------------------------------
# Create lookup tables
# ------------------------------------------------------------

s1_lookup = s1.set_index(
    "entity_id"
)

s2_lookup = s2.set_index(
    "entity_id"
)

s3_lookup = s3.set_index(
    "entity_id"
)


# ------------------------------------------------------------
# Sample ground truth
# ------------------------------------------------------------

sample_size = min(
    100000,
    len(gt)
)

gt_sample = gt.sample(
    n=sample_size,
    random_state=42
).copy()

print(
    "Ground truth S1 rows sampled:",
    len(gt_sample)
)


# ------------------------------------------------------------
# Analyze true matches
# ------------------------------------------------------------

results = []

for _, row in gt_sample.iterrows():

    s1_id = row[
        "source1_entity_id"
    ]

    # Skip singleton
    if pd.isna(
        row["matched_entity_ids"]
    ):
        continue

    # Verify S1 ID exists
    if s1_id not in s1_lookup.index:
        continue

    s1_row = s1_lookup.loc[
        s1_id
    ]

    matched_ids = [
        x.strip()
        for x in str(
            row["matched_entity_ids"]
        ).split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        # ----------------------------------------------------
        # Find S2/S3 record
        # ----------------------------------------------------

        if matched_id.startswith("S2-"):

            if matched_id not in s2_lookup.index:
                continue

            external_row = s2_lookup.loc[
                matched_id
            ]

            source = "S2"

        elif matched_id.startswith("S3-"):

            if matched_id not in s3_lookup.index:
                continue

            external_row = s3_lookup.loc[
                matched_id
            ]

            source = "S3"

        else:
            continue

        # ----------------------------------------------------
        # Safely extract fields
        # ----------------------------------------------------

        s1_name = clean_value(
            s1_row["business_name"]
        )

        ext_name = clean_value(
            external_row["business_name"]
        )

        s1_address = clean_value(
            s1_row["business_address"]
        )

        ext_address = clean_value(
            external_row["business_address"]
        )

        s1_country = clean_value(
            s1_row["country"]
        )

        ext_country = clean_value(
            external_row["country"]
        )

        # ----------------------------------------------------
        # Exact name comparison
        # ----------------------------------------------------

        name_exact = (
            s1_name != ""
            and ext_name != ""
            and s1_name.lower()
            == ext_name.lower()
        )

        # ----------------------------------------------------
        # Exact address comparison
        # ----------------------------------------------------

        address_exact = (
            s1_address != ""
            and ext_address != ""
            and s1_address.lower()
            == ext_address.lower()
        )

        # ----------------------------------------------------
        # Exact country comparison
        # ----------------------------------------------------

        country_exact = (
            s1_country != ""
            and ext_country != ""
            and s1_country.lower()
            == ext_country.lower()
        )

        results.append(
            {
                "source": source,

                "name_exact": name_exact,

                "address_exact": address_exact,

                "country_exact": country_exact,

                "both_name_address_exact": (
                    name_exact
                    and address_exact
                ),

                "name_or_address_exact": (
                    name_exact
                    or address_exact
                )
            }
        )


# ------------------------------------------------------------
# Convert results to DataFrame
# ------------------------------------------------------------

match_analysis = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print(
    "Matched record pairs analyzed:",
    len(match_analysis)
)


if len(match_analysis) > 0:

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print(
        "\n===== OVERALL EXACT AGREEMENT ====="
    )

    print(
        "Name exact:",
        round(
            match_analysis[
                "name_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Address exact:",
        round(
            match_analysis[
                "address_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Country exact:",
        round(
            match_analysis[
                "country_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Name AND address exact:",
        round(
            match_analysis[
                "both_name_address_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Name OR address exact:",
        round(
            match_analysis[
                "name_or_address_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    # --------------------------------------------------------
    # By source
    # --------------------------------------------------------

    print(
        "\n===== EXACT AGREEMENT BY SOURCE ====="
    )

    source_summary = (
        match_analysis
        .groupby("source")[
            [
                "name_exact",
                "address_exact",
                "country_exact",
                "both_name_address_exact",
                "name_or_address_exact"
            ]
        ]
        .mean()
        .mul(100)
        .round(2)
    )

    print(source_summary)

else:

    print(
        "\nWARNING: No matched pairs were analyzed."
    )

    print(
        "Check ground-truth IDs and lookup logic."
    )


# ============================================================
# END
# ============================================================

print(
    "\n===== DATA INSPECTION COMPLETE ====="
)