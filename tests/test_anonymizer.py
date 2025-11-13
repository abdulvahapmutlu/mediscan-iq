from mediscan_iq.preprocess.anonymizer import anonymize

def test_email_phone_masking()
    raw = Patient Mr. John Doe email john.doe@hospital.org, phone +1 415-555-0199. MRN 12345678.
    out, counts = anonymize(raw)
    assert john.doe@hospital.org not in out
    assert +1 415-555-0199 not in out
    assert counts.get(email, 0) = 1
    assert counts.get(phone, 0) = 1
    assert counts.get(mrn, 0) = 1

def test_whitespace_reduction()
    raw = Line 1   n   Line 2
    out, _ = anonymize(raw)
    assert    not in out
    assert out.splitlines() == [Line 1, Line 2]
