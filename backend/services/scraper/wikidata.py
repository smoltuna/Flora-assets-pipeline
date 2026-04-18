"""Wikidata SPARQL client — queries structured botanical triples."""
from dataclasses import dataclass, field

import httpx


@dataclass
class WikidataData:
    latin_name: str
    qid: str | None = None
    native_range: list[str] = field(default_factory=list)
    conservation_status: str | None = None
    family: str | None = None
    common_names: dict[str, str] = field(default_factory=dict)  # lang → name
    native_range_description: str | None = None
    raw_bindings: list[dict] = field(default_factory=list)


_SPARQL = "https://query.wikidata.org/sparql"
_HEADERS = {
    "User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)",
    "Accept": "application/sparql-results+json",
}


async def fetch_wikidata(latin_name: str) -> WikidataData | None:
    # First resolve the Wikidata QID by latin name
    lookup_query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item wdt:P225 "{latin_name}" .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 1
    """
    async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS) as client:
        resp = await client.get(_SPARQL, params={"query": lookup_query, "format": "json"})
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        if not bindings:
            return None

        qid_uri = bindings[0]["item"]["value"]
        qid = qid_uri.split("/")[-1]

        # Now fetch properties for this QID
        props_query = f"""
        SELECT ?prop ?propLabel ?value ?valueLabel ?lang WHERE {{
          BIND(wd:{qid} AS ?entity)
          {{
            ?entity wdt:P171 ?value .  # parent taxon
            BIND("parent_taxon" AS ?prop)
          }} UNION {{
            ?entity wdt:P141 ?value .  # conservation status
            BIND("conservation_status" AS ?prop)
          }} UNION {{
            ?entity wdt:P183 ?value .  # endemic to
            BIND("endemic_to" AS ?prop)
          }} UNION {{
            ?entity wdt:P136 ?value .  # genre/family (sometimes used)
            BIND("genre" AS ?prop)
          }} UNION {{
            ?entity wdt:P105 ?value .  # taxon rank
            BIND("taxon_rank" AS ?prop)
          }} UNION {{
            ?entity wdt:P18 ?value .   # image
            BIND("image" AS ?prop)
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
          BIND("en" AS ?lang)
        }}
        """
        props_resp = await client.get(_SPARQL, params={"query": props_query, "format": "json"})
        props_resp.raise_for_status()
        prop_bindings = props_resp.json().get("results", {}).get("bindings", [])

        # Also fetch common names in target languages
        names_query = f"""
        SELECT ?lang ?name WHERE {{
          BIND(wd:{qid} AS ?entity)
          ?entity wdt:P1843 ?name .
          BIND(LANG(?name) AS ?lang)
          FILTER(?lang IN ("de", "fr", "es", "it", "zh", "ja", "en"))
        }}
        """
        names_resp = await client.get(_SPARQL, params={"query": names_query, "format": "json"})
        names_resp.raise_for_status()
        name_bindings = names_resp.json().get("results", {}).get("bindings", [])

    data = WikidataData(latin_name=latin_name, qid=qid)
    data.raw_bindings = prop_bindings

    for b in prop_bindings:
        prop = b.get("prop", {}).get("value", "")
        val_label = b.get("valueLabel", {}).get("value", "")
        if prop == "parent_taxon" and val_label:
            data.family = val_label
        elif prop == "conservation_status" and val_label:
            data.conservation_status = val_label
        elif prop == "endemic_to" and val_label:
            data.native_range.append(val_label)

    for b in name_bindings:
        lang = b.get("lang", {}).get("value", "")
        name = b.get("name", {}).get("value", "")
        if lang and name:
            data.common_names[lang] = name

    if data.native_range:
        data.native_range_description = ", ".join(data.native_range)

    return data
