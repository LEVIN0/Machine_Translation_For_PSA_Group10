"""Per-site collector configurations (>=12 documented sources) and entry points.

URLs are starting points only; every fetch failure is non-fatal by design
(see SiteCollector and docs/ETHICS.md for the politeness policy).
"""

from .base import DOMAIN_ALIAS, SiteCollector

_DEFAULT_SELECTORS = ["article p", "main p", ".field-item p", "p"]

SITES = [
    # NOTE: Ministry of Health Kenya (health.go.ke) was evaluated and dropped
    # as a scraping source: press statements and disease-alert pages contain
    # almost no HTML text (content is published as PDFs/images). Documented in
    # the Week 1 report challenges. Amref replaces it as a health source.

    {
        "name": "who_covid_qna",
        "domain": "Health",
        "source": "WHO",
        "start_urls": ["https://www.who.int/news-room/questions-and-answers/item/coronavirus-disease-covid-19"],
        "link_patterns": [],  # start_url is the article page itself
        "content_selectors": [".sf-content-block p", "article p", "main p", "p"],
        "max_pages": 15,
    },
    {
        "name": "who_fact_sheets",
        "domain": "Health",
        "source": "WHO",
        "start_urls": ["https://www.who.int/news-room/fact-sheets"],
        "link_patterns": [r"/news-room/fact-sheets/detail/"],
        "content_selectors": ["article p", "main p", "p"],
        "max_pages": 100,
    },
    {
        "name": "unicef_parenting",
        "domain": "Health",
        "source": "UNICEF",
        "start_urls": ["https://www.unicef.org/parenting/health"],
        "link_patterns": [r"/parenting/"],
        "content_selectors": ["article p", "main p", "p"],
        "max_pages": 15,
    },
    {
        "name": "redcross_news",
        "domain": "Health",
        "source": "Kenya Red Cross",
        # /news is 404; real listing pages are the WP category page and /blog.
        "start_urls": [
            "https://www.redcross.or.ke/category/news",
            "https://www.redcross.or.ke/blog",
        ],
        # Articles are root-level slugs; exclude known section pages.
        "link_patterns": [
            r"redcross\.or\.ke/(?!about-us|careers|contact-us|governance|"
            r"impact-stories|category|tag|disclaimer|donate|blog|newsletter|"
            r"get-involved|our-work|comments|feed|wp-)[a-z0-9-]+/?$"
        ],
        "content_selectors": ["article p", "main p", ".entry-content p", "p"],
        "max_pages": 25,
    },
    {
        "name": "amref_kenya",
        "domain": "Health",
        "source": "Amref Health Africa",
        "start_urls": ["https://amref.org/kenya/news/"],
        # Articles live at amref.org/kenya/<slug>/; exclude section pages.
        "link_patterns": [
            r"amref\.org/kenya/(?!board-members|consulting|about|contact|"
            r"careers|donate|news)[a-z0-9-]+/$"
        ],
        "content_selectors": ["article p", "main p", ".entry-content p", "p"],
        "max_pages": 40,
        "pagination": {
            "template": "https://amref.org/kenya/news/page/{n}/",
            "start": 2, "pages": 2,
        },
    },

    
    {
        "name": "ntsa",
        "domain": "Security",  # road safety announcements -> Security per DOMAIN_ALIAS
        "source": "NTSA",
        "start_urls": ["https://www.ntsa.go.ke/news"],
        "link_patterns": [r"/news/"],
        "content_selectors": list(_DEFAULT_SELECTORS),
        "max_pages": 15,
            "verify_ssl": False,  # .go.ke cert chain breaks under antivirus TLS inspection
    },
    {
        "name": "kenya_met",
        "domain": "Security",  # weather/flood warnings
        "source": "Kenya Meteorological Department",
        "start_urls": ["https://meteo.go.ke/news"],
        "link_patterns": [r"/news/", r"/node/"],
        "content_selectors": list(_DEFAULT_SELECTORS),
        "max_pages": 20,
            "verify_ssl": False,  # .go.ke cert chain breaks under antivirus TLS inspection
    },
    {
        "name": "kilimo",
        "domain": "Agriculture",
        "source": "Ministry of Agriculture",
        "start_urls": ["https://kilimo.go.ke/category/news/"],
        "link_patterns": [r"kilimo\.go\.ke/\d{4}/", r"/category/news/"],
        "content_selectors": ["article p", ".entry-content p", "main p", "p"],
        "max_pages": 20,
            "verify_ssl": False,  # .go.ke cert chain breaks under antivirus TLS inspection
    },
    {
        "name": "education_ke",
        "domain": "Education",
        "source": "Ministry of Education",
        "start_urls": ["https://www.education.go.ke/news"],
        "link_patterns": [r"/news/", r"/node/"],
        "content_selectors": list(_DEFAULT_SELECTORS),
        "max_pages": 20,
            "verify_ssl": False,  # .go.ke cert chain breaks under antivirus TLS inspection
    },
    {
        "name": "kra_notices",
        "domain": "Governance",
        "source": "Kenya Revenue Authority",
        "start_urls": ["https://www.kra.go.ke/news-center/public-notices"],
        "link_patterns": [r"/news-center/"],
        "content_selectors": ["article p", "main p", ".field-item p", "p"],
        "max_pages": 40,
        "pagination": {
            "template": "https://www.kra.go.ke/news-center/public-notices?page={n}",
            "start": 1, "pages": 3,
        },
    },
    {
        "name": "ecitizen",
        "domain": "Governance",
        "source": "eCitizen Kenya",
        "start_urls": ["https://www.ecitizen.go.ke"],
        "link_patterns": [],
        "content_selectors": ["main p", "article p", "p"],
        "max_pages": 15,
    },
    {
        "name": "nps_kenya",
        "domain": "Security",
        "source": "National Police Service",
        "start_urls": ["https://www.nationalpolice.go.ke/news"],
        "link_patterns": [r"/news/"],
        "content_selectors": list(_DEFAULT_SELECTORS),
        "max_pages": 15,
            "verify_ssl": False,  # .go.ke cert chain breaks under antivirus TLS inspection
    },
    {
        "name": "infonet_agri",
        "domain": "Agriculture",
        "source": "Infonet-Biovision (Biovision Africa Trust)",
        # Kenyan agri/health info portal; articles have Swahili versions
        # ("Sw" toggle) - note for Week 2 parallel harvesting.
        "start_urls": [
            "https://infonet-biovision.org/crops-fruits-vegetables",
            "https://infonet-biovision.org/animal-health-and-disease",
        ],
        "link_patterns": [
            r"infonet-biovision\.org/(PlantHealth|crops-fruits-vegetables|"
            r"animal-health-and-disease|indigenous-plants|plant_pests|"
            r"agro-ecological-zones)/[a-z0-9-]+"
        ],
        "content_selectors": [
            "article p, article li", "main p, main li",
            ".field--name-body p, .field--name-body li", "p"
        ],
        "max_pages": 60,
        "min_words": 7,        # kill short photo captions
        "max_records": 2500,    # keep dataset domain-balanced (was 22k rows!)
    },
    {
        "name": "nema_insights",
        "domain": "Security",  # disaster preparedness / environmental safety
        "source": "National Environment Management Authority",
        # BEST EFFORT: verify yield after first run, tune pattern if needed.
        "start_urls": ["https://nema.go.ke/category/insights/article/"],
        "link_patterns": [
            r"nema\.go\.ke/(?!board-of-management|contact-us|county-offices|"
            r"knowledge-base|nema-departments|category|tag|wp-|about|licensing)"
            r"[a-z0-9-]+/$"
        ],
        "content_selectors": ["article p", "main p", ".entry-content p", "p"],
        "max_pages": 20,
    },
    {
        "name": "ca_cyber",
        "domain": "Security",  # cybersecurity awareness
        "source": "Communications Authority of Kenya",
        "start_urls": ["https://www.ca.go.ke/cyber-security"],
        "link_patterns": [],  # the page itself is the PSA content
        "content_selectors": [
            "article p, article li", "main p, main li", "p"
        ],
        "max_pages": 5,
    },
    {
        "name": "eacc_news",
        "domain": "Governance",  # anti-corruption
        "source": "Ethics and Anti-Corruption Commission",
        # BEST EFFORT: verify yield after first run.
        "start_urls": [
            "https://eacc.go.ke/default/news/",
            "https://eacc.go.ke/default/category/news/",
        ],
        "link_patterns": [
            r"eacc\.go\.ke/default/(?!category|tag|wp-content|wp-includes|"
            r"contact-us|faqs|by-email|by-phone|Downloads|achievements|news/?$)"
            r"[a-zA-Z0-9-]+/?$"
        ],
        "content_selectors": ["article p", "main p", ".entry-content p", "p"],
        "max_pages": 15,
    },
    {
        "name": "kicd_news",
        "domain": "Education",
        "source": "Kenya Institute of Curriculum Development",
        # BEST EFFORT: news listing treated as content page (teasers).
        "start_urls": ["https://kicd.ac.ke/news"],
        "link_patterns": [],
        "content_selectors": ["article p", "main p", ".item-page p", "p"],
        "max_pages": 5,
    },
]

_SITE_BY_NAME = {s["name"]: s for s in SITES}


def collect_site(name, verbose=True):
    """Collect records from one configured site by name.

    Raises KeyError for unknown site names (programmer error, not network).
    """
    cfg = _SITE_BY_NAME[name]
    return SiteCollector(cfg).collect(verbose=verbose)


def collect_all(names=None, max_pages=None, verbose=True):
    """Collect from all (or selected) sites; per-site failures are non-fatal."""
    names = names or [s["name"] for s in SITES]
    records = []
    for name in names:
        try:
            cfg = dict(_SITE_BY_NAME[name])
            if max_pages is not None:
                cfg["max_pages"] = max_pages
            if verbose:
                print(f"[collect_all] --- {name} ({cfg['source']}) ---")
            records.extend(SiteCollector(cfg).collect(verbose=verbose))
        except Exception as exc:
            print(f"[collect_all] WARNING: site '{name}' failed entirely ({exc}); continuing")
    return records


__all__ = ["SITES", "DOMAIN_ALIAS", "collect_site", "collect_all"]
