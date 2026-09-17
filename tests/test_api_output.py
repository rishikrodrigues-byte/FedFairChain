from app.main import health, model_info


def test_health_endpoint_function():
    assert health() == {"status": "healthy", "model_loaded": True}


def test_model_info_endpoint_function():
    info = model_info()
    assert info["project"] == "FedFairChain"
    assert info["chain_order"] == ["DIABETES", "CVD", "HYPERTENSION"]
    assert info["base_feature_count"] == 15
    assert info["fairness_aware"] is True
    assert info["diagnosis_aware"] is True
    assert info["explainability"] == "SHAP"
