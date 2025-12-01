def get_conditional_len(y_catalog: dict) -> int:
    """
    Get the context length for the flow model given dropped variables
    and the y_catalog.
    """
    total_size = sum(y_catalog["variables"][v]["size"] for v in y_catalog["variables"])
    drop_size = sum(
        y_catalog["variables"][v]["size"] for v in y_catalog["drop_variables"]
    )
    return int(total_size - drop_size)