import numpy as np
import pytest

from domainbed.fewshot.methods import (CLIPZeroShot, KNN, NCM, Ridge,
                                       SimpleShotSupport, TipAdapter, TipAdapterFull,
                                       VisualCache)


def example():
    x = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.2],
                  [-1.0, 0.0, 0.2], [0.0, -1.0, 0.2]])
    y = np.array([0, 1, 2, 3])
    q = np.array([[0.8, 0.1, 0.2], [-0.7, 0.1, 0.2], [0.1, -0.9, 0.2]])
    return x, y, q


def test_one_shot_cosine_ncm_and_one_nn_same_rankings():
    x, y, q = example()
    config = {"num_classes": 4, "metric": "cosine"}
    ncm = NCM().fit(x, y, config)
    nn = KNN().fit(x, y, {**config, "k": 1})
    np.testing.assert_array_equal(np.argsort(ncm.predict_scores(q), axis=1),
                                  np.argsort(nn.predict_scores(q), axis=1))
    np.testing.assert_array_equal(ncm.predict(q), nn.predict(q))


def test_ridge_satisfies_centered_normal_equation():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(18, 6))
    y = np.arange(18) % 3
    ridge = Ridge().fit(x, y, {"num_classes": 3, "alpha": 0.7})
    z = x - x.mean(axis=0)
    target = np.eye(3)[y] - np.eye(3)[y].mean(axis=0)
    residual = z.T @ (z @ ridge.weight - target) + 0.7 * ridge.weight
    np.testing.assert_allclose(residual, 0, atol=1e-10)


@pytest.mark.parametrize("method,config", [
    (NCM, {"metric": "euclidean"}), (KNN, {"k": 3, "weighted": True}),
    (Ridge, {"alpha": 1}), (SimpleShotSupport, {}), (VisualCache, {"beta": 5.5}),
    (CLIPZeroShot, {"text_features": np.eye(4, 3)}),
    (TipAdapter, {"text_features": np.eye(4, 3), "alpha": 1, "beta": 5.5}),
])
def test_query_independence_and_shape(method, config):
    x, y, q = example()
    classifier = method().fit(x, y, {"num_classes": 4, **config})
    full = classifier.predict_scores(q)
    assert full.shape == (3, 4)
    assert np.isfinite(full).all()
    for index in range(3):
        np.testing.assert_allclose(classifier.predict_scores(q[index:index + 1])[0],
                                   full[index], atol=1e-12)
    np.testing.assert_allclose(classifier.predict_scores(q[::-1])[::-1], full, atol=1e-12)
    np.testing.assert_allclose(classifier.predict_scores(q), full, atol=1e-12)


def test_knn_ties_are_deterministic_and_k_is_capped():
    x = np.array([[1., 0.], [-1., 0.], [0., 1.]])
    y = np.array([0, 1, 2])
    model = KNN().fit(x, y, {"num_classes": 3, "k": 20, "metric": "euclidean"})
    assert model.k_effective == 3
    assert model.predict(np.array([[0., 0.]]))[0] == 0


def test_full_tip_adapter_matches_author_cache_equations():
    import torch

    x, y, q = example()
    views = np.stack([x + 0.01, x - 0.02, x + 0.03], axis=1)
    text = np.eye(4, 3)
    config = {"num_classes": 4, "support_views": views,
              "text_features": text, "alpha": 2.55, "beta": 6.29}
    actual = TipAdapterFull().fit(x, y, config).predict_scores(q)
    keys = torch.tensor(views.mean(axis=1))
    keys = torch.nn.functional.normalize(keys, dim=-1)
    query = torch.nn.functional.normalize(torch.tensor(q), dim=-1)
    weights = torch.nn.functional.normalize(torch.tensor(text), dim=-1)
    affinity = query @ keys.T
    values = torch.nn.functional.one_hot(torch.tensor(y), num_classes=4).double()
    expected = 100 * query @ weights.T + 2.55 * torch.exp(-6.29 * (1 - affinity)) @ values
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-12, atol=1e-12)
