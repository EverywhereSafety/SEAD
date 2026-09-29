"""Class balancing used by SAGE supervised fine-tuning."""

def balanced_class_weights(pass_count: int, block_count: int) -> tuple[float, float]:
    """Return normalized inverse-frequency ``(PASS, BLOCK)`` weights."""

    if pass_count <= 0 or block_count <= 0:
        raise ValueError("both PASS and BLOCK counts must be positive")
    total = pass_count + block_count
    return block_count / total, pass_count / total
