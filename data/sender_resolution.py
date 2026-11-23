MESSAGING_PACKAGES = frozenset({
    "com.whatsapp",
    "org.telegram.messenger",
    "com.facebook.orca",
    "com.instagram.android",
    "com.discord",
    "com.google.android.apps.messaging",
    "org.thoughtcrime.securesms",
})

EMAIL_PACKAGES = frozenset({
    "com.google.android.gm",
    "com.microsoft.office.outlook",
})


def resolve_sender_key(package_name: str, title) -> str:
    if (package_name in MESSAGING_PACKAGES or package_name in EMAIL_PACKAGES) and title:
        return f"{package_name}:{title}"
    return package_name
