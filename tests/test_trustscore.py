from backend.scorer import compute_trust_score


def test_compute_trust_score_basic():
    sentiment = 0.8
    rating_norm = 0.6
    has_image = True
    like_norm = 0.2
    comment_length_norm = 0.5

    trust = compute_trust_score(sentiment, rating_norm, has_image, like_norm, comment_length_norm)

    # Compute manually to compare
    image_score = 1.0
    uyum = 1 - abs(sentiment - rating_norm)
    expected = (
        0.40 * sentiment +
        0.35 * rating_norm +
        0.05 * image_score +
        0.05 * like_norm +
        0.10 * uyum +
        0.05 * comment_length_norm
    )
    expected = round(float(expected), 4)

    assert trust == expected
