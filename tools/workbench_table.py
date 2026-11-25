"""Resolve the exact tab-delimited study table linked from a Workbench study page."""

from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import Request, urlopen


USER_AGENT = "Longitudinal-omics-reproduction/1.0"
TABLE_PAGE = "https://www.metabolomicsworkbench.org/data/data_table_output.php"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def table_url(study, study_type, analysis):
    page = f"{TABLE_PAGE}?STUDY_ID={study}&STUDY_TYPE={study_type}&RESULT_TYPE=1"
    request = Request(page, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=90) as response:
        document = response.read().decode("utf-8", errors="replace")
    parser = Links()
    parser.feed(document)
    name = f"MSdata_{study}_{analysis}.txt"
    candidates = set()
    for href in parser.hrefs:
        url = urljoin(page, href)
        parts = urlsplit(url)
        if (parts.scheme == "https" and parts.hostname == "www.metabolomicsworkbench.org"
                and parts.path == "/data/showfile_t.php" and parse_qs(parts.query).get("DF") == [name]):
            candidates.add(url)
    if len(candidates) != 1:
        raise ValueError(f"Expected one download for {study}/{analysis} at {page}; found {len(candidates)}.")
    return candidates.pop()
