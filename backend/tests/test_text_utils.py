from app import text_utils


def test_strip_accents_removes_diacritics():
    assert text_utils.strip_accents("Hôm nay trời đẹp quá") == "hom nay troi dep qua"


def test_normalize_text_replaces_teencode():
    assert text_utils.normalize_text("hok biet dc dau") == "không biết được dau"
