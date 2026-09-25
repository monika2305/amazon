import pandas as pd

from normalization import normalize_dataframe


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


# ============================================================
# TEST NORMALIZATION ON SMALL SAMPLES FIRST
# ============================================================

print("\n===== NORMALIZATION SAMPLE TEST =====")


sample_s1 = s1.head(10).copy()
sample_s2 = s2.head(10).copy()
sample_s3 = s3.head(10).copy()


normalized_s1 = normalize_dataframe(
    sample_s1
)

normalized_s2 = normalize_dataframe(
    sample_s2
)

normalized_s3 = normalize_dataframe(
    sample_s3
)


# ============================================================
# DISPLAY SOURCE 1
# ============================================================

print("\n===== SOURCE 1 NORMALIZED =====")

print(
    normalized_s1[
        [
            "entity_id",
            "business_name",
            "name_normalized",
            "name_compact",
            "business_address",
            "address_normalized",
            "address_compact",
            "country",
            "country_normalized"
        ]
    ].to_string(index=False)
)


# ============================================================
# DISPLAY SOURCE 2
# ============================================================

print("\n===== SOURCE 2 NORMALIZED =====")

print(
    normalized_s2[
        [
            "entity_id",
            "business_name",
            "name_normalized",
            "name_compact",
            "business_address",
            "address_normalized",
            "address_compact",
            "country",
            "country_normalized"
        ]
    ].to_string(index=False)
)


# ============================================================
# DISPLAY SOURCE 3
# ============================================================

print("\n===== SOURCE 3 NORMALIZED =====")

print(
    normalized_s3[
        [
            "entity_id",
            "business_name",
            "name_normalized",
            "name_compact",
            "business_address",
            "address_normalized",
            "address_compact",
            "country",
            "country_normalized"
        ]
    ].to_string(index=False)
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n===== NORMALIZATION VALIDATION =====")

for name, df in [
    ("S1", normalized_s1),
    ("S2", normalized_s2),
    ("S3", normalized_s3)
]:

    print(f"\n{name}:")

    print(
        "Rows:",
        len(df)
    )

    print(
        "Missing normalized names:",
        df["name_normalized"].isna().sum()
    )

    print(
        "Missing normalized addresses:",
        df["address_normalized"].isna().sum()
    )

    print(
        "Missing normalized countries:",
        df["country_normalized"].isna().sum()
    )


print(
    "\n===== NORMALIZATION SAMPLE TEST COMPLETE ====="
)