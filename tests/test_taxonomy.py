from vap.taxonomy import load_taxonomy

def test_taxonomy_load():
    tax = load_taxonomy("configs/taxonomy.yaml")
    assert tax.is_valid("DRIVE")
    assert tax.is_valid("GRAB_SKID")
    assert not tax.is_valid("NON_EXISTENT")
