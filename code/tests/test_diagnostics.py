from llm_reward_evolver.diagnostics import (
    analyze_original_reward_anchor,
    detect_forbidden_domain_terms,
    uses_original_reward,
)


def test_anchor_accepts_direct_original_reward():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    shaping = 0.1
    return original_reward + shaping
"""
    report = analyze_original_reward_anchor(code, min_multiplier=0.5)
    assert not report.suspicious


def test_anchor_rejects_tiny_literal_scale():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return original_reward * 0.001
"""
    report = analyze_original_reward_anchor(code, min_multiplier=0.5)
    assert report.suspicious
    assert report.min_multiplier == 0.001


def test_anchor_rejects_tiny_variable_scale():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    anchor = 0.001
    return anchor * original_reward
"""
    report = analyze_original_reward_anchor(code, min_multiplier=0.5)
    assert report.suspicious
    assert report.min_multiplier == 0.001


def test_anchor_rejects_missing_original_reward():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return 0.1
"""
    report = analyze_original_reward_anchor(code, min_multiplier=0.5)
    assert report.suspicious


def test_domain_guard_detects_task_terms():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    left_leg_contact = next_obs[6]
    return original_reward + left_leg_contact
"""
    hits = detect_forbidden_domain_terms(code, "leg,contact,fuel")
    assert hits == ["contact", "leg"]


def test_domain_guard_accepts_generic_names():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    n6 = next_obs[6]
    return original_reward + n6
"""
    assert detect_forbidden_domain_terms(code, "leg,contact,fuel") == []


def test_uses_original_reward_detection():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return original_reward
"""
    assert uses_original_reward(code)
