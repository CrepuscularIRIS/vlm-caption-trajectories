import caption_ao as ao  # run with PYTHONPATH=code/reference (caption_v51, caption_v3)

def errs(t, crops=0, avail=("original_image",)):
    p = ao.parse_step(t); return p, ao.check_step(p, 1, crops, list(avail))

def test_answer():
    p, e = errs("<answer>C</answer>"); assert e == [] and p["kind"] == "answer" and p["answer"] == "C"

def test_think_then_answer():
    p, e = errs("<think>\nlooks like C\n</think>\n<answer>C</answer>"); assert e == [] and p["answer"] == "C"

def test_grounding():
    p, e = errs('<grounding>{"bbox_2d":[0.1,0.1,0.5,0.5],"source":"original_image"}</grounding>')
    assert e == [] and p["kind"] == "grounding"

def test_bad():
    assert errs("C")[1]
    assert errs("<answer>C</answer><answer>D</answer>")[1]
    assert errs("The answer is <answer>C</answer>")[1]
    assert errs('<grounding>{"bbox_2d":[0.1,0.1,0.5,0.5],"source":"observation_1"}</grounding>')[1]
    assert errs('<grounding>{"bbox_2d":[0.1,0.1,0.5,0.5],"source":"original_image"}</grounding>', crops=8)[1]

def test_receipts_shared():
    import caption_v51 as v
    assert ao.receipt_text is v.receipt_text and ao.first_user_text is v.first_user_text and ao.SYSTEM != v.SYSTEM
