from src.met_api import fetch_object
from src.schema import ArtworkDossier


class UnusableRecordError(Exception):
    """Raised when a record can't produce a usable dossier — no rights, or no image."""
    pass


def build_dossier(object_id: int) -> ArtworkDossier:
    response = fetch_object(object_id)

    if response["status_code"] != 200 or response["error"]:
        raise UnusableRecordError(f"Object {object_id}: API error — {response['error']}")

    data = response["data"]

    if not data.get("isPublicDomain"):
        raise UnusableRecordError(f"Object {object_id}: not in public domain")

    image_url = data.get("primaryImage") or ""
    if not image_url:
        raise UnusableRecordError(f"Object {object_id}: not a usable image")

    return ArtworkDossier(
        object_id=data["objectID"],
        title=data.get("title") or "Untitled",
        artist=data.get("artistDisplayName") or None,
        date=data.get("objectDate") or None,
        medium=data.get("medium") or None,
        department=data["department"],
        credit_line=data.get("creditLine") or None,
        image_url=image_url,
        is_public_domain=data["isPublicDomain"],
    )