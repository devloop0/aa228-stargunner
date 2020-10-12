def settings_is_valid(settings, required_settings):
    """
    Simple check to determine whether provided
    settings object contains all the settings
    specified in required_settings.

    Inputs:
    - settings: dict
    - required_settings: list<str>
    """
    settings_set = set(settings.keys())
    required_set = set(required_settings)
    return len(settings_set.intersection(required_set)) == len(required_set)
