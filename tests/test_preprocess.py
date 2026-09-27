"""Unit tests for conservative entity-resolution text preprocessing."""

import math
import unittest

from src.preprocess import (
    normalize_address,
    normalize_business_name,
    normalize_business_name_core,
    normalize_country,
    normalize_record,
)


class PreprocessTests(unittest.TestCase):
    def test_null_and_empty_values_are_safe(self):
        for value in (None, "", "   ", math.nan):
            self.assertEqual(normalize_business_name(value), "")
            self.assertEqual(normalize_business_name_core(value), "")
            self.assertEqual(normalize_address(value), "")
            self.assertEqual(normalize_country(value), "")

    def test_name_normalization_preserves_the_legal_suffix(self):
        self.assertEqual(normalize_business_name("  ACME & Sons, Inc. "), "acme and sons inc")

    def test_core_name_removes_only_terminal_unambiguous_suffixes(self):
        self.assertEqual(normalize_business_name_core("ACME & Sons, Inc."), "acme and sons")
        self.assertEqual(normalize_business_name_core("Example Pvt. Ltd."), "example")
        self.assertEqual(normalize_business_name_core("The Company Store"), "the company store")
        self.assertEqual(normalize_business_name_core("LLC"), "llc")

    def test_address_keeps_numbers_postal_tokens_and_cautious_aliases(self):
        self.assertEqual(
            normalize_address(" 12-B, Main Rd., Apt #5, 560 001 "),
            "12 b main road apartment 5 560 001",
        )
        self.assertEqual(normalize_address("St. John's Dr."), "st john s dr")

    def test_country_is_open_set_not_a_hard_coded_mapping(self):
        self.assertEqual(normalize_country(" République Française "), "république française")
        self.assertEqual(normalize_country("U.S.A."), "u s a")

    def test_record_preprocessing_adds_fields_without_changing_raw_values(self):
        record = {"name": "A&B LLC", "address": "10 Oak Rd", "country": "France"}
        processed = normalize_record(record)
        self.assertEqual(record, {"name": "A&B LLC", "address": "10 Oak Rd", "country": "France"})
        self.assertEqual(processed["name_normalized"], "a and b llc")
        self.assertEqual(processed["name_core_normalized"], "a and b")
        self.assertEqual(processed["address_normalized"], "10 oak road")
        self.assertEqual(processed["country_normalized"], "france")


if __name__ == "__main__":
    unittest.main()
