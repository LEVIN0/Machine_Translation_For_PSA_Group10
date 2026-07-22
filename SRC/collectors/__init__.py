"""Site collectors package: generic collector plus per-site configs."""

from .base import DOMAIN_ALIAS, SiteCollector
from .sites import SITES, collect_all, collect_site

__all__ = ["SiteCollector", "DOMAIN_ALIAS", "SITES", "collect_site", "collect_all"]
