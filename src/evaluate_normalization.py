import pandas as pd

from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

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
# HELPER
# ============================================================

def clean_value(value):
    """
    Convert missing values to empty string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# LOOKUPS
# ============================================================

print("Creating lookup tables...")

s1_lookup = s1.set_index(
    "entity_id"
)

s2_lookup = s2.set_index(
    "entity_id"
)

s3_lookup = s3.set_index(
    "entity_id"
)


# ============================================================
# SAMPLE GROUND TRUTH
# ============================================================

sample_size = min(
    100000,
    len(gt)
)

gt_sample = gt.sample(
    n=sample_size,
    random_state=42
)

print(
    "Ground truth rows sampled:",
    len(gt_sample)
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# ANALYZE TRUE MATCHES
# ============================================================

for _, row in gt_sample.iterrows():

    s1_id = row["source1_entity_id"]

    if pd.isna(
        row["matched_entity_ids"]
    ):
        continue

    if s1_id not in s1_lookup.index:
        continue

    s1_row = s1_lookup.loc[s1_id]

    # --------------------------------------------------------
    # Source 1 values
    # --------------------------------------------------------

    s1_name = clean_value(
        s1_row["business_name"]
    )

    s1_address = clean_value(
        s1_row["business_address"]
    )

    s1_country = clean_value(
        s1_row["country"]
    )

    # --------------------------------------------------------
    # Normalized Source 1
    # --------------------------------------------------------

    s1_name_norm = (
        normalize_business_name(
            s1_name
        )
    )

    s1_address_norm = (
        normalize_business_address(
            s1_address
        )
    )

    s1_country_norm = (
        normalize_country(
            s1_country
        )
    )

    # --------------------------------------------------------
    # Ground truth matches
    # --------------------------------------------------------

    matched_ids = [
        x.strip()
        for x in str(
            row["matched_entity_ids"]
        ).split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        # ----------------------------------------------------
        # Find external record
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
        # External values
        # ----------------------------------------------------

        ext_name = clean_value(
            external_row["business_name"]
        )

        ext_address = clean_value(
            external_row["business_address"]
        )

        ext_country = clean_value(
            external_row["country"]
        )

        # ----------------------------------------------------
        # Normalized external values
        # ----------------------------------------------------

        ext_name_norm = (
            normalize_business_name(
                ext_name
            )
        )

        ext_address_norm = (
            normalize_business_address(
                ext_address
            )
        )

        ext_country_norm = (
            normalize_country(
                ext_country
            )
        )

        # ====================================================
        # RAW EXACT MATCH
        # ====================================================

        raw_name_exact = (
            s1_name != ""
            and ext_name != ""
            and s1_name.casefold()
            == ext_name.casefold()
        )

        raw_address_exact = (
            s1_address != ""
            and ext_address != ""
            and s1_address.casefold()
            == ext_address.casefold()
        )

        # ====================================================
        # NORMALIZED EXACT MATCH
        # ====================================================

        normalized_name_exact = (
            s1_name_norm["name_normalized"] != ""
            and ext_name_norm["name_normalized"] != ""
            and
            s1_name_norm["name_normalized"]
            ==
            ext_name_norm["name_normalized"]
        )

        normalized_name_compact_exact = (
            s1_name_norm["name_compact"] != ""
            and ext_name_norm["name_compact"] != ""
            and
            s1_name_norm["name_compact"]
            ==
            ext_name_norm["name_compact"]
        )

        normalized_address_exact = (
            s1_address_norm["address_normalized"] != ""
            and ext_address_norm["address_normalized"] != ""
            and
            s1_address_norm["address_normalized"]
            ==
            ext_address_norm["address_normalized"]
        )

        normalized_address_compact_exact = (
            s1_address_norm["address_compact"] != ""
            and ext_address_norm["address_compact"] != ""
            and
            s1_address_norm["address_compact"]
            ==
            ext_address_norm["address_compact"]
        )

        country_exact = (
            s1_country_norm != ""
            and ext_country_norm != ""
            and
            s1_country_norm
            ==
            ext_country_norm
        )

        results.append(
            {
                "source": source,

                "raw_name_exact":
                    raw_name_exact,

                "normalized_name_exact":
                    normalized_name_exact,

                "normalized_name_compact_exact":
                    normalized_name_compact_exact,

                "raw_address_exact":
                    raw_address_exact,

                "normalized_address_exact":
                    normalized_address_exact,

                "normalized_address_compact_exact":
                    normalized_address_compact_exact,

                "country_exact":
                    country_exact
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\nMatched pairs analyzed:",
    len(results_df)
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print(
    "\n===== NORMALIZATION IMPACT ====="
)

if len(results_df) > 0:

    print(
        "Raw name exact:",
        round(
            results_df[
                "raw_name_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Normalized name exact:",
        round(
            results_df[
                "normalized_name_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Compact name exact:",
        round(
            results_df[
                "normalized_name_compact_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Raw address exact:",
        round(
            results_df[
                "raw_address_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Normalized address exact:",
        round(
            results_df[
                "normalized_address_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Compact address exact:",
        round(
            results_df[
                "normalized_address_compact_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Country exact:",
        round(
            results_df[
                "country_exact"
            ].mean() * 100,
            2
        ),
        "%"
    )


# ============================================================
# BY SOURCE
# ============================================================

print(
    "\n===== NORMALIZATION IMPACT BY SOURCE ====="
)

if len(results_df) > 0:

    source_summary = (
        results_df
        .groupby("source")[
            [
                "raw_name_exact",
                "normalized_name_exact",
                "normalized_name_compact_exact",
                "raw_address_exact",
                "normalized_address_exact",
                "normalized_address_compact_exact",
                "country_exact"
            ]
        ]
        .mean()
        .mul(100)
        .round(2)
    )

    print(source_summary)


# ============================================================
# IMPROVEMENT
# ============================================================

print(
    "\n===== IMPROVEMENT FROM NORMALIZATION ====="
)

if len(results_df) > 0:

    raw_name = (
        results_df[
            "raw_name_exact"
        ].mean() * 100
    )

    normalized_name = (
        results_df[
            "normalized_name_exact"
        ].mean() * 100
    )

    compact_name = (
        results_df[
            "normalized_name_compact_exact"
        ].mean() * 100
    )

    raw_address = (
        results_df[
            "raw_address_exact"
        ].mean() * 100
    )

    normalized_address = (
        results_df[
            "normalized_address_exact"
        ].mean() * 100
    )

    compact_address = (
        results_df[
            "normalized_address_compact_exact"
        ].mean() * 100
    )

    print(
        "Name improvement:",
        round(
            normalized_name - raw_name,
            2
        ),
        "percentage points"
    )

    print(
        "Compact name improvement:",
        round(
            compact_name - raw_name,
            2
        ),
        "percentage points"
    )

    print(
        "Address improvement:",
        round(
            normalized_address - raw_address,
            2
        ),
        "percentage points"
    )

    print(
        "Compact address improvement:",
        round(
            compact_address - raw_address,
            2
        ),
        "percentage points"
    )


print(
    "\n===== NORMALIZATION EVALUATION COMPLETE ====="
)