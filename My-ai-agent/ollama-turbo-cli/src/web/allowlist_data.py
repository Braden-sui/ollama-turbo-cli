"""
Curated default allowlist and blocklist for trusted sources.
This list biases toward major international, national, and high-signal outlets,
plus government (.gov, .gouv, .go.<cc>), multilateral orgs (.int), and
high-signal science and policy domains. It is merged with env/config at runtime.

Notes
- Matching is suffix-based: a host is considered allowlisted if it equals a
  pattern or ends with .<pattern>.
- Include base TLD buckets (gov, edu, ac.uk, gouv.fr, go.jp) to cover official
  government/academic domains broadly.
- Users can extend or override via WEB_NEWS_SOURCES_ALLOW and WEB_ALLOWLIST_FILE.
"""

# Broad TLD buckets and multilateral orgs
DEFAULT_ALLOWLIST = [
    # Buckets
    "gov", "edu", "ac.uk", "gouv.fr", "go.jp", "go.kr", "go.id", "go.au",
    "gov.uk", "gov.ie", "gov.it", "gov.sg", "gov.hk", "gov.tw", "gov.za", "gov.ng", "gov.br", "gov.mx", "gov.in", "gov.cn",
    "europa.eu", "ec.europa.eu", "eu.int", "int",
    # Multilateral / intergovernmental / standards
    "un.org", "who.int", "oecd.org", "imf.org", "worldbank.org", "wto.org",
    "nato.int", "icao.int", "itu.int", "iso.org", "iec.ch",
    # US federal / agencies
    "whitehouse.gov", "congress.gov", "federalreserve.gov", "treasury.gov", "sec.gov",
    "cftc.gov", "commerce.gov", "ftc.gov", "fcc.gov", "justice.gov", "state.gov",
    "fbi.gov", "cia.gov", "dni.gov", "dod.gov", "army.mil", "navy.mil", "af.mil",
    "hhs.gov", "nih.gov", "cdc.gov", "fda.gov", "cms.gov",
    "noaa.gov", "nasa.gov", "epa.gov", "usda.gov", "energy.gov", "transportation.gov",
    "faa.gov", "nhtsa.gov", "census.gov", "bls.gov", "uspto.gov", "gsa.gov",
    # UK gov
    "gov.uk", "parliament.uk", "ons.gov.uk", "nhs.uk", "bankofengland.co.uk",
    # EU agencies / institutions
    "ema.europa.eu", "eba.europa.eu", "esma.europa.eu", "ecb.europa.eu",
    # Canada gov
    "canada.ca", "gc.ca", "statcan.gc.ca", "bankofcanada.ca",
    # Australia gov
    "gov.au", "abs.gov.au", "rba.gov.au",
    # Germany gov
    "bund.de", "destatis.de", "bundesbank.de",
    # France gov
    "gouv.fr", "insee.fr", "banque-france.fr",
    # Japan gov
    "go.jp", "stat.go.jp", "boj.or.jp",
    # Mexico gov
    "gob.mx",
    # Spain gov
    "lamoncloa.gob.es", "boe.es",
    # Italy gov
    "governo.it", "istat.it", "bancaditalia.it",
    # Netherlands gov
    "rijksoverheid.nl", "cbs.nl", "autoriteitpersoonsgegevens.nl",
    # Singapore gov
    "gov.sg", "mas.gov.sg",
    # Hong Kong gov
    "gov.hk",
    # Taiwan gov
    "gov.tw",
    # International reputable media (English)
    "apnews.com", "reuters.com", "bbc.com", "bbc.co.uk", "theguardian.com",
    "nytimes.com", "wsj.com", "washingtonpost.com", "bloomberg.com",
    "economist.com", "ft.com", "aljazeera.com", "npr.org", "pbs.org",
    "cbsnews.com", "abcnews.go.com", "cnn.com", "time.com", "newsweek.com",
    "axios.com", "politico.com", "politico.eu", "afp.com", "dpa.com",
    "latimes.com", "usatoday.com", "cnbc.com", "marketwatch.com",
    # Canada media
    "cbc.ca", "ctvnews.ca", "globalnews.ca", "theglobeandmail.com", "financialpost.com",
    # Australia media
    "abc.net.au", "theaustralian.com.au", "smh.com.au", "afr.com",
    # UK media
    "thetimes.co.uk", "telegraph.co.uk", "independent.co.uk", "sky.com", "skynews.com",
    # France media
    "lemonde.fr", "lefigaro.fr", "liberation.fr", "france24.com", "rfi.fr",
    # Germany media
    "dw.com", "spiegel.de", "zeit.de", "faz.net", "sueddeutsche.de",
    # Spain media
    "elpais.com", "elmundo.es", "abc.es", "lavanguardia.com", "rtve.es",
    # Italy media
    "ansa.it", "repubblica.it", "corriere.it", "ilsole24ore.com",
    # Ireland media
    "rte.ie", "irishtimes.com",
    # Nordics media
    "aftonbladet.se", "svt.se", "vg.no", "nrk.no", "yle.fi",
    # Asia media
    "straitstimes.com", "japantimes.co.jp", "asahi.com", "mainichi.jp", "yomiuri.co.jp",
    "scmp.com", "koreatimes.co.kr", "koreaherald.com", "thehindu.com", "indianexpress.com",
    # Middle East media
    "haaretz.com", "timesofisrael.com", "al-monitor.com", "arabnews.com",
    # Latin America media
    "folha.uol.com.br", "oglobo.globo.com", "clarin.com", "lanacion.com.ar", "eluniversal.com.mx",
    # Business/Tech/Science
    "forbes.com", "wired.com", "techcrunch.com", "theverge.com", "arstechnica.com",
    "spectrum.ieee.org", "cacm.acm.org", "dl.acm.org",
    "nature.com", "science.org", "sciencemag.org", "jamanetwork.com", "nejm.org", "thelancet.com", "bmj.com", "cell.com", "pnas.org", "plos.org",
    "springer.com", "springernature.com", "sciencedirect.com", "wiley.com", "tandfonline.com", "sagepub.com",
    "arxiv.org", "biorxiv.org", "medrxiv.org",
    # Additional global government and law patterns/seeds (converted to suffix matches)
    "mil", "fed.us",
    "justice.fr", "assemblee-nationale.fr",
    "parl.ca",
    "aph.gov.au", "govt.nz", "parliament.nz",
    "bmi.bund.de", "gesetze-im-internet.de",
    "planalto.gov.br", "ibge.gov.br",
    "meti.go.jp", "mhlw.go.jp",
    "korea.kr", "bok.or.kr", "kostat.go.kr",
    "nic.in", "rbi.org.in", "mospi.gov.in",
    "moh.gov.sg",
    "statssa.gov.za",
    "mfa.gov.cn",
    "gob.es", "ine.es",
    "gov.it", "senato.it", "camera.it",
    "gouv.be", "belgium.be", "statbel.fgov.be",
    "admin.ch", "bundeskanzlei.admin.ch", "bfs.admin.ch",
    "gov.se", "riksdagen.se", "scb.se",
    "gov.no", "regjeringen.no", "ssb.no",
    "gov.dk", "ft.dk", "dst.dk",
    "gov.fi", "eduskunta.fi", "stat.fi",
    "gov.pl", "sejm.gov.pl", "nac.gov.pl", "stat.gov.pl",
    "gov.nl",
    "gov.tr", "tccb.gov.tr", "mevzuat.gov.tr", "tuik.gov.tr",
    "gov.il", "knesset.gov.il", "cbs.gov.il",
    "gov.gr", "hellenicparliament.gr", "statistics.gr",
    "gov.pt", "parlamento.pt", "ine.pt",
    "gov.cz", "psp.cz", "czso.cz",
    "gov.hu", "ksh.hu",
    "gov.ua", "rada.gov.ua", "bank.gov.ua", "ukrstat.gov.ua",
    "gov.ru", "duma.gov.ru", "cbr.ru", "gks.ru",
    # Courts and legal resources
    "supremecourt.gov", "uscourts.gov", "law.cornell.edu", "courtlistener.com",
    "curia.europa.eu", "bailii.org", "austlii.edu.au", "canlii.org",
    # Statistics and multilaterals (additional)
    "ilo.org", "wipo.int", "unesco.org", "undp.org", "unhcr.org", "unicef.org", "eia.gov", "bea.gov",
    # Health/science indexes and registries
    "pubmed.ncbi.nlm.nih.gov", "crossref.org", "openalex.org", "clinicaltrials.gov", "osti.gov",
    # Additional science publishers and societies
    "royalsocietypublishing.org", "iopscience.iop.org", "aps.org", "aip.scitation.org", "acm.org", "ieee.org",
    "oup.com", "cambridge.org", "aaas.org",
    # Standards and specs
    "ietf.org", "w3.org", "unicode.org", "whatwg.org", "nist.gov", "ansi.org", "bsigroup.com", "din.de", "etsi.org", "cencenelec.eu",
    # Geography/weather and geospatial
    "nhc.noaa.gov", "metoffice.gov.uk", "dwd.de", "usgs.gov", "esa.int", "earthdata.nasa.gov", "copernicus.eu",
    # Libraries and identifiers
    "loc.gov", "catalog.loc.gov", "hathitrust.org", "archive.org", "opencitations.net", "orcid.org",
    # Ethics and safety
    "nrc.gov", "icrc.org", "oecd.ai", "ai.gov",
    # Ireland specific further seeds
    "courts.ie", "revenue.ie", "garda.ie",
]

# Blocklist for problematic live feeds/maps that are poor for citation stability
DEFAULT_BLOCKLIST = [
    "liveuamap.com",
]
