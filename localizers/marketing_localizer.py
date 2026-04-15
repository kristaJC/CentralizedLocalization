from generic_localizer import GenericLocalizer


class MarketingLocalizer(GenericLocalizer):
    """
    Marketing localization inherits all behavior from GenericLocalizer.

    Inheritance chain: MarketingLocalizer → GenericLocalizer → LocalizationRun

    Required input sheet columns:
        - en_US        : English string to translate
        - char_limit   : (optional) max character count; must be numeric if provided

    Any additional columns (e.g. token, context, type) are treated as context
    and passed to the model to inform tone and word choice.

    Languages are manually selected per request (TargetLanguages field in the
    submission form), unlike InGame which has a fixed language set per game.
    """
    pass
