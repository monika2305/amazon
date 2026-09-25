import re
import unicodedata
import pandas as pd


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text):
    """
    Normalize Unicode representation.

    Important:
    We DO NOT remove non-ASCII characters.
    This preserves Indian/French/other scripts.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Canonical Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    return text


# ============================================================
# LOWERCASE NORMALIZATION
# ============================================================

def normalize_case(text):
    """
    Convert text to lowercase while preserving Unicode.
    """

    text = normalize_unicode(text)

    return text.casefold()


# ============================================================
# SPACE NORMALIZATION
# ============================================================

def normalize_spaces(text):
    """
    Collapse multiple whitespace characters into one space.
    """

    text = normalize_case(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PUNCTUATION NORMALIZATION
# ============================================================

def normalize_punctuation(text):
    """
    Replace punctuation/separators with spaces.

    We intentionally preserve letters and numbers.
    """

    text = normalize_spaces(text)

    # Replace punctuation/symbol separators with spaces.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Collapse spaces again
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# COMPACT REPRESENTATION
# ============================================================

def compact_text(text):
    """
    Remove whitespace from the normalized representation.

    Useful for comparing:
        ABC CORP
        ABC-CORP
        ABCCORP
    """

    text = normalize_punctuation(text)

    return re.sub(
        r"\s+",
        "",
        text
    )


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_business_name(name):
    """
    Generate multiple representations of a business name.

    Returns:
        original
        normalized
        compact
    """

    original = (
        ""
        if pd.isna(name)
        else str(name).strip()
    )

    normalized = normalize_punctuation(
        original
    )

    compact = compact_text(
        original
    )

    return {
        "name_original": original,
        "name_normalized": normalized,
        "name_compact": compact
    }


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_business_address(address):
    """
    Generate multiple representations of an address.

    We preserve numbers because house/building/street
    numbers can be highly informative.
    """

    original = (
        ""
        if pd.isna(address)
        else str(address).strip()
    )

    normalized = normalize_punctuation(
        original
    )

    compact = compact_text(
        original
    )

    return {
        "address_original": original,
        "address_normalized": normalized,
        "address_compact": compact
    }


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(country):
    """
    Normalize country as an open-set string.

    No hard-coded country list is used.
    """

    if pd.isna(country):
        return ""

    return normalize_spaces(
        country
    )


# ============================================================
# NORMALIZE ONE RECORD
# ============================================================

def normalize_record(
    business_name,
    business_address,
    country
):
    """
    Normalize all matching fields for one record.
    """

    name_features = normalize_business_name(
        business_name
    )

    address_features = normalize_business_address(
        business_address
    )

    normalized_country = normalize_country(
        country
    )

    return {
        **name_features,
        **address_features,
        "country_normalized": normalized_country
    }


# ============================================================
# NORMALIZE COMPLETE DATAFRAME
# ============================================================

def normalize_dataframe(df):
    """
    Add normalized matching columns to a dataframe.

    Original columns remain unchanged.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    name_features = result[
        "business_name"
    ].apply(
        normalize_business_name
    )

    name_features = pd.DataFrame(
        name_features.tolist(),
        index=result.index
    )

    # --------------------------------------------------------
    # Business address
    # --------------------------------------------------------

    address_features = result[
        "business_address"
    ].apply(
        normalize_business_address
    )

    address_features = pd.DataFrame(
        address_features.tolist(),
        index=result.index
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    result["country_normalized"] = (
        result["country"]
        .apply(normalize_country)
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = pd.concat(
        [
            result,
            name_features,
            address_features
        ],
        axis=1
    )

    return result


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("===== NORMALIZATION TEST =====")

    test_records = [
        {
            "business_name":
                "ABC Corporation Pvt. Ltd.",
            "business_address":
                "1795 Westchester Drive, High Point, NC",
            "country":
                "US"
        },
        {
            "business_name":
                "ABC-Corporation PVT LTD",
            "business_address":
                "1795 Westchester Drive, High Point, NC",
            "country":
                " US "
        },
        {
            "business_name":
                "श्री राम टेक्सटाइल्स",
            "business_address":
                "मुंबई, महाराष्ट्र, भारत",
            "country":
                "India"
        },
        {
            "business_name":
                "Société Générale",
            "business_address":
                "20 Rue de la Paix, Paris",
            "country":
                "France"
        }
    ]

    for record in test_records:

        result = normalize_record(
            record["business_name"],
            record["business_address"],
            record["country"]
        )

        print("\nOriginal:")
        print(record)

        print("\nNormalized:")
        print(result)