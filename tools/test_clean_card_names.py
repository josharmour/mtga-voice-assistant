#!/usr/bin/env python3
"""
Test the clean_card_name function to ensure HTML tags are properly removed.
"""

import re


def clean_card_name(name: str) -> str:
    """Remove HTML tags from card names"""
    if not name:
        return name
    return re.sub(r'<[^>]+>', '', name)


def test_clean_card_name():
    """Test cases for clean_card_name function"""
    test_cases = [
        # (input, expected_output)
        ("<nobr>Full-Throttle</nobr> Fanatic", "Full-Throttle Fanatic"),
        ("<nobr>Bane-Marked</nobr> Leonin", "Bane-Marked Leonin"),
        ("<nobr>Half-Elf</nobr> Monk", "Half-Elf Monk"),
        ("<nobr>Yuan-Ti</nobr> <nobr>Fang-Blade</nobr>", "Yuan-Ti Fang-Blade"),
        ("<sprite=\"SpriteSheet_MiscIcons\" name=\"arena_a\">Demilich", "Demilich"),
        ("Elas <i>il</i>-Kor, Sadistic Pilgrim", "Elas il-Kor, Sadistic Pilgrim"),
        ("Normal Card Name", "Normal Card Name"),
        ("", ""),
        (None, None),
    ]

    print("Testing clean_card_name function...")
    print("=" * 80)

    all_passed = True
    for i, (input_name, expected) in enumerate(test_cases, 1):
        result = clean_card_name(input_name)
        passed = result == expected

        if passed:
            print(f"✓ Test {i} PASSED")
        else:
            print(f"✗ Test {i} FAILED")
            print(f"  Input:    {repr(input_name)}")
            print(f"  Expected: {repr(expected)}")
            print(f"  Got:      {repr(result)}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    success = test_clean_card_name()
    exit(0 if success else 1)
