import bs4
import json
import os
import requests
import spacy
from bs4 import BeautifulSoup
from collections import defaultdict
from tqdm.notebook import tqdm

spacy.prefer_gpu()


class NoPageException(Exception):
    pass


class NoDataException(Exception):
    pass


def create_dataset(seed_title: str):
    base_out_path = "datasets"
    os.makedirs(base_out_path, exist_ok=True)

    if os.path.isfile("datasets/wiki/wiki_dataset.json"):
        with open(
            "datasets/wiki/wiki_dataset.json",
            encoding="utf-8"
        ) as file:
            dataset = json.load(file)
    else:
        dataset = {
            "page_ids": [],
            "page_titles": [],
            "token_num": 0,
            "total_pages": 0,
            "total_historical_pages": 0,
            "total_other_pages": 0,
            "paragraphs": []
        }

    titles = get_related_pages(seed_title)

    for title in tqdm(titles):
        if title not in dataset["page_titles"]:
            print(title)
            try:
                page = get_wikipedia_page(title)
            except NoPageException:
                print("Could not find Wikipedia page for ", title)
                continue
            try:
                page_id, page_wikidata_id = get_wikidata_info(title)
            except NoDataException:
                print("Missing data for ", title)
                continue

            # La pagina è storica se presenta almeno una proprieta' legata a 
            # un'entita' storica. Le proprieta' e le entita' storiche sono
            # specificate in apposite liste della funzione is_historical.
            page_historical = is_historical(page_wikidata_id)

            paragraphs, total_tokens = process_page(
                page, title, page_id, 
                page_wikidata_id, page_historical
            )
            
            dataset["page_ids"].append(page["pageid"])
            dataset["page_titles"].append(title)
            dataset["token_num"] += total_tokens
            dataset["total_pages"] += 1
            
            if page_historical:
                dataset["total_historical_pages"] += 1
            else:
                dataset["total_other_pages"] += 1
            
            # Extend aggiunge ogni elemento della lista paragraphs, mentre append
            # aggiungerebbe la lista intera come elemento singolo.
            dataset["paragraphs"].extend(paragraphs)
            
            with open(
                "datasets/wiki/wiki_dataset.json", "w", 
                encoding="utf-8"
            ) as file:
                json.dump(
                    dataset, file, 
                    ensure_ascii=False, 
                    indent=4
                )
                
    print("token_num", dataset["token_num"])
    print("total_pages", dataset["total_pages"])

    return dataset


def get_related_pages(title: str):
    session = requests.Session()

    # L'API di Wikipedia restituisce sempre 20 pagine correlate a un dato titolo.
    # Non è possibile modificare il numero di pagine correlate restituite.
    url = f"https://en.wikipedia.org/api/" \
        + f"rest_v1/page/related/{title}"

    request = session.get(url=url)
    data = request.json()

    related_pages_titles = [title]

    for page in data["pages"]:
        related_pages_titles.append(page["title"])

    return related_pages_titles


def get_wikipedia_page(title: str):
    session = requests.Session()

    url = "https://en.wikipedia.org/w/api.php"

    # L'azione parse analizza il contenuto di una pagina MediaWiki e restituisce
    # un'analisi dei suoi dati, tra cui i titoli delle sezioni e i collegamenti
    # interni ed esterni.
    params = {
        "action": "parse",
        "page": title,
        "format": "json"
    }

    request = session.get(url=url, params=params)
    data = request.json()
    
    if "error" in data:
        raise NoPageException
    
    # Il campo "*" contiene il testo grezzo HTML.
    return {
        "pageid": data["parse"]["pageid"],
        "html": data["parse"]["text"]["*"]
    }


def get_wikidata_info(title: str):
    session = requests.Session()

    url = "https://en.wikipedia.org/w/api.php"

    # L'azione query è l'azione predefinita e viene usata per eseguire query sul database.
    # Il parametro prob specifica le proprieta' delle pagine che si vogliono ottenere.
    # Con ppprop si possono richiedere le informazioni sugli elementi Wikibase relativi
    # a collegamenti interni a Wikipedia.
    # Con redirects:1 si ottengono le informazioni di pagine di destinazione di
    # reindirizzamenti.
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "redirects": 1,
        "format": "json"
    }

    request = session.get(url=url, params=params)
    data = request.json()
    page_ids, wikidata_ids = [], []
    
    if "pages" not in data["query"]:
        raise NoDataException

    for _, v in data["query"]["pages"].items():
        if "missing" in v:
            raise NoDataException
        if "pageprops" not in v:
            raise NoDataException
        page_ids.append(v["pageid"])
        wikidata_ids.append(v["pageprops"]["wikibase_item"])

    # Esempio di return: (51825, 'Q37151')
    return page_ids[0], wikidata_ids[0]


def is_historical(wikidata_id: str):
    """True = 1; False = 0."""
    
    # Mappa delle proprietà.
    p = {
        "conflict": "P607",
        "history of topic": "P2184",
        "instance of": "P31",
        "member of political party": "P102",
        "military branch": "P241",
        "monogram": "P1543",
        "noble title": "P97",
        "occupation": "P106",
        "part of": "P361",
        "position held": "P39",
        "studied by": "P2579",
        "subclass of": "P279",
        "topic's main Wikimedia portal": "P1151",
        "topic's main category": "P910"
    }

    # Mappa delle entità.
    e = {
        "historical period": "Q11514315",
        "Portal:World War II": "Q3247957",
        "Category:World War II": "Q6816704",
        "siege": "Q188055",
        "war": "Q198",
        "Napoleonic Wars": "Q78994",
        "politician": "Q82955",
        "revolution": "Q10931",
        "Portal:French Revolution": "Q3247542",
        "Category:French Revolution": "Q7216178",
        "battle": "Q178561",
        "combat": "Q650711",
        "historical event": "Q13418847",
        "history": "Q309",
        "military operation": "Q645883",
        "study of history": "Q1066186",
        "Category:Historical events": "Q32571532",
        "military officer": "Q189290",
        "riot": "Q124757",
        "protest": "Q273120",
        "political crisis": "Q3002772",
        "diplomat": "Q193391",
        "political party": "Q7278",
        "civil war": "Q8465",
        "ethnic conflict": "Q766875",
        "military unit": "Q176799",
        "revolutionary": "Q3242115",
        "military unit branch class": "Q62933934",
        "political organization": "Q7210356",
        "militant": "Q17010072",
        "demonstration": "Q175331",
        "rebel": "Q1125062",
        "classical antiquity": "Q486761",
        "ancient history": "Q41493",
        "historical country": "Q3024240",
        "world war": "Q103495",
        "Portal:World War I": "Q10651811",
        "Category:World War I": "Q6816935",
        "conflict": "Q180684",
        "military campaign": "Q831663",
        "peace treaty": "Q625298",
        "intergovernmental organization": "Q245065",
        "alliance": "Q878249",
        "treaty": "Q131569",
        "international crisis": "Q5791104",
        "bilateral treaty": "Q9557810",
        "arms race": "Q322348",
        "army": "Q37726",
        "Imperial Conference": "Q2139567",
        "war front": "Q842332",
        "protectorate": "Q164142",
        "intervention": "Q1168287",
        "famine": "Q168247",
        "synod": "Q111161",
        "encyclical": "Q221409",
        "cold war": "Q4176199",
        "perpetual war": "Q1469686",
        "proxy war": "Q864113",
        "covert operation": "Q1546073",
        "aerial bombing": "Q17912683",
        "nuclear tests series": "Q98391050",
        "landing operation": "Q646740",
        "amphibious warfare": "Q348120",
        "incident": "Q18669875",
        "secret treaty": "Q1498487",
        "multilateral treaty": "Q6934728",
        "aviation accident": "Q744913",
        "clandestine operation": "Q3354903",
        "nuclear weapons test": "Q210112",
        "underground nuclear weapons test": "Q3058675",
        "aerial bombing of a city": "Q4688003",
        "bombardment": "Q678146",
        "nuclear explosion": "Q2656967",
        "nuclear disaster": "Q15725976",
        "calendar era": "Q4375074",
        "dynasty": "Q164950",
        "archaeological age": "Q15401699",
        "historical region": "Q1620908",
        "historical ethnic group": "Q4204501",
        "archaeological period": "Q15401633",
        "mass migration": "Q6784066",
        "aspect of history": "Q17524420",
        "era": "Q6428674",
        "periodization": "Q816829",
        "history of a country or state": "Q17544377",
        "interdict": "Q6291244",
        "schism": "Q41521",
        "papal election": "Q29102902",
        "election": "Q40231",
        "government": "Q7188",
        "religious controversy": "Q7311342",
        "reform": "Q900406",
        "religious war": "Q1827102",
        "geological period": "Q392928",
        "peace conference": "Q7157512",
        "international conference": "Q18564543",
        "legislation": "Q49371",
        "genocide denial": "Q339452",
        "government program": "Q22222786",
        "pan-nationalism": "Q1428239",
        "Public Act of Parliament of the United Kingdom": "Q105774620",
        "Act of Congress in the United States": "Q476068",
        "rebellion": "Q124734",
        "Conscription crisis": "Q3002736",
        "resistance movement": "Q138796",
        "provisional government": "Q59281",
        "Soviet Decree": "Q4349624",
        "separate peace": "Q2300991",
        "referendum": "Q43109",
        "labour movement": "Q208701",
        "Australian federal election": "Q22284407",
        "Act of Parliament of the United Kingdom": "Q4677783",
        "intergovernmental organization": "Q245065",
        "political ideology": "Q12909644",
        "cultural movement": "Q2198855",
        "federal government": "Q1006644",
        "military alliance": "Q1127126",
        "armed conflict": "Q350604",
        "political movement": "Q2738074",
        "constitution": "Q7755",
        "coup d'état": "Q45382",
        "counter-revolution": "Q755138",
        "armed organization": "Q17149090",
        "former administrative territorial entity": "Q19953632",
        "colony": "Q133156",
        "military occupation": "Q188686",
        "World War I": "Q361",
        "military intervention": "Q5919191",
        "revolt": "Q6107280",
        "military offensive": "Q2001676",
        "Reichstag election in Germany": "Q1504429",
        "United Kingdom general election": "Q15283424",
        "Turkish general election": "Q22333900",
        "war reparations": "Q194181",
        "disease outbreak": "Q3241045",
        "influenza pandemic": "Q2723958",
        "pandemic": "Q12184",
        "genocide": "Q41397",
        "annexation": "Q194465",
        "historical fact": "Q4204519",
        "dissolution of an administrative territorial entity": "Q18603729",
        "general strike": "Q49775",
        "war of independence": "Q21994376",
        "war of national liberation": "Q1006311",
        "national liberation movement": "Q209225",
        "political police": "Q2101516",
        "crime against humanity": "Q173462",
        "fortified line": "Q2973801",
        "Law of Italy": "Q2135331",
        "civil code": "Q1923776",
        "french judiciary code": "Q50380591",
        "Legislation": "Q820655",
        "ancient city": "Q15661340",
        "naval battle": "Q1261499",
        "political murder": "Q1139665",
        "conspiracy": "Q930164",
        "assassination": "Q3882219",
        "Renaissance": "Q4692",
        "coalition": "Q124964",
        "ancient civilization": "Q28171280",
        "age": "Q17522177",
        "invasion": "Q467011",
        "chronicle": "Q185363",
        "crisis": "Q381072",
        "historical Chinese state": "Q50068795",
        "plague": "Q133780",
        "public health emergency of international concern": "Q17076801"
    }
    
    statements = get_statements(wikidata_id)
 
    if check_prop_entity(p["topic's main category"], e["Category:Historical events"], statements) \
        or check_prop_entity(p["topic's main Wikimedia portal"], e["Portal:French Revolution"], statements) \
        or check_prop_entity(p["topic's main Wikimedia portal"], e["Portal:World War II"], statements) \
        or check_prop_entity(p["topic's main Wikimedia portal"], e["Portal:World War I"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:French Revolution"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:World War II"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:World War I"], statements) \
        or check_prop_entity(p["instance of"], e["historical period"], statements) \
        or check_prop_entity(p["instance of"], e["public health emergency of international concern"], statements) \
        or check_prop_entity(p["instance of"], e["plague"], statements) \
        or check_prop_entity(p["instance of"], e["historical Chinese state"], statements) \
        or check_prop_entity(p["instance of"], e["crisis"], statements) \
        or check_prop_entity(p["instance of"], e["chronicle"], statements) \
        or check_prop_entity(p["instance of"], e["invasion"], statements) \
        or check_prop_entity(p["instance of"], e["age"], statements) \
        or check_prop_entity(p["instance of"], e["ancient civilization"], statements) \
        or check_prop_entity(p["instance of"], e["coalition"], statements) \
        or check_prop_entity(p["instance of"], e["assassination"], statements) \
        or check_prop_entity(p["instance of"], e["conspiracy"], statements) \
        or check_prop_entity(p["instance of"], e["political murder"], statements) \
        or check_prop_entity(p["instance of"], e["naval battle"], statements) \
        or check_prop_entity(p["instance of"], e["ancient city"], statements) \
        or check_prop_entity(p["instance of"], e["Legislation"], statements) \
        or check_prop_entity(p["instance of"], e["french judiciary code"], statements) \
        or check_prop_entity(p["instance of"], e["civil code"], statements) \
        or check_prop_entity(p["instance of"], e["Law of Italy"], statements) \
        or check_prop_entity(p["instance of"], e["fortified line"], statements) \
        or check_prop_entity(p["instance of"], e["crime against humanity"], statements) \
        or check_prop_entity(p["instance of"], e["political police"], statements) \
        or check_prop_entity(p["instance of"], e["national liberation movement"], statements) \
        or check_prop_entity(p["instance of"], e["war of national liberation"], statements) \
        or check_prop_entity(p["instance of"], e["war of independence"], statements) \
        or check_prop_entity(p["instance of"], e["general strike"], statements) \
        or check_prop_entity(p["instance of"], e["dissolution of an administrative territorial entity"], statements) \
        or check_prop_entity(p["instance of"], e["historical fact"], statements) \
        or check_prop_entity(p["instance of"], e["annexation"], statements) \
        or check_prop_entity(p["instance of"], e["genocide"], statements) \
        or check_prop_entity(p["instance of"], e["pandemic"], statements) \
        or check_prop_entity(p["instance of"], e["influenza pandemic"], statements) \
        or check_prop_entity(p["instance of"], e["disease outbreak"], statements) \
        or check_prop_entity(p["instance of"], e["war reparations"], statements) \
        or check_prop_entity(p["instance of"], e["Turkish general election"], statements) \
        or check_prop_entity(p["instance of"], e["United Kingdom general election"], statements) \
        or check_prop_entity(p["instance of"], e["Reichstag election in Germany"], statements) \
        or check_prop_entity(p["instance of"], e["military offensive"], statements) \
        or check_prop_entity(p["instance of"], e["revolt"], statements) \
        or check_prop_entity(p["instance of"], e["military intervention"], statements) \
        or check_prop_entity(p["instance of"], e["military occupation"], statements) \
        or check_prop_entity(p["instance of"], e["colony"], statements) \
        or check_prop_entity(p["instance of"], e["former administrative territorial entity"], statements) \
        or check_prop_entity(p["instance of"], e["armed organization"], statements) \
        or check_prop_entity(p["instance of"], e["counter-revolution"], statements) \
        or check_prop_entity(p["instance of"], e["coup d'état"], statements) \
        or check_prop_entity(p["instance of"], e["constitution"], statements) \
        or check_prop_entity(p["instance of"], e["political movement"], statements) \
        or check_prop_entity(p["instance of"], e["armed conflict"], statements) \
        or check_prop_entity(p["instance of"], e["military alliance"], statements) \
        or check_prop_entity(p["instance of"], e["federal government"], statements) \
        or check_prop_entity(p["instance of"], e["cultural movement"], statements) \
        or check_prop_entity(p["instance of"], e["political ideology"], statements) \
        or check_prop_entity(p["instance of"], e["intergovernmental organization"], statements) \
        or check_prop_entity(p["instance of"], e["Act of Parliament of the United Kingdom"], statements) \
        or check_prop_entity(p["instance of"], e["Australian federal election"], statements) \
        or check_prop_entity(p["instance of"], e["labour movement"], statements) \
        or check_prop_entity(p["instance of"], e["referendum"], statements) \
        or check_prop_entity(p["instance of"], e["separate peace"], statements) \
        or check_prop_entity(p["instance of"], e["Soviet Decree"], statements) \
        or check_prop_entity(p["instance of"], e["provisional government"], statements) \
        or check_prop_entity(p["instance of"], e["resistance movement"], statements) \
        or check_prop_entity(p["instance of"], e["Conscription crisis"], statements) \
        or check_prop_entity(p["instance of"], e["rebellion"], statements) \
        or check_prop_entity(p["instance of"], e["Act of Congress in the United States"], statements) \
        or check_prop_entity(p["instance of"], e["Public Act of Parliament of the United Kingdom"], statements) \
        or check_prop_entity(p["instance of"], e["encyclical"], statements) \
        or check_prop_entity(p["instance of"], e["pan-nationalism"], statements) \
        or check_prop_entity(p["instance of"], e["government program"], statements) \
        or check_prop_entity(p["instance of"], e["genocide denial"], statements) \
        or check_prop_entity(p["instance of"], e["legislation"], statements) \
        or check_prop_entity(p["instance of"], e["international conference"], statements) \
        or check_prop_entity(p["instance of"], e["peace conference"], statements) \
        or check_prop_entity(p["instance of"], e["geological period"], statements) \
        or check_prop_entity(p["instance of"], e["religious war"], statements) \
        or check_prop_entity(p["instance of"], e["reform"], statements) \
        or check_prop_entity(p["instance of"], e["religious controversy"], statements) \
        or check_prop_entity(p["instance of"], e["government"], statements) \
        or check_prop_entity(p["instance of"], e["election"], statements) \
        or check_prop_entity(p["instance of"], e["papal election"], statements) \
        or check_prop_entity(p["instance of"], e["schism"], statements) \
        or check_prop_entity(p["instance of"], e["interdict"], statements) \
        or check_prop_entity(p["instance of"], e["history of a country or state"], statements) \
        or check_prop_entity(p["instance of"], e["periodization"], statements) \
        or check_prop_entity(p["instance of"], e["era"], statements) \
        or check_prop_entity(p["instance of"], e["mass migration"], statements) \
        or check_prop_entity(p["instance of"], e["archaeological period"], statements) \
        or check_prop_entity(p["instance of"], e["historical ethnic group"], statements) \
        or check_prop_entity(p["instance of"], e["historical region"], statements) \
        or check_prop_entity(p["instance of"], e["archaeological age"], statements) \
        or check_prop_entity(p["instance of"], e["dynasty"], statements) \
        or check_prop_entity(p["instance of"], e["calendar era"], statements) \
        or check_prop_entity(p["instance of"], e["nuclear disaster"], statements) \
        or check_prop_entity(p["instance of"], e["nuclear explosion"], statements) \
        or check_prop_entity(p["instance of"], e["bombardment"], statements) \
        or check_prop_entity(p["instance of"], e["aerial bombing of a city"], statements) \
        or check_prop_entity(p["instance of"], e["underground nuclear weapons test"], statements) \
        or check_prop_entity(p["instance of"], e["nuclear weapons test"], statements) \
        or check_prop_entity(p["instance of"], e["clandestine operation"], statements) \
        or check_prop_entity(p["instance of"], e["aviation accident"], statements) \
        or check_prop_entity(p["instance of"], e["multilateral treaty"], statements) \
        or check_prop_entity(p["instance of"], e["secret treaty"], statements) \
        or check_prop_entity(p["instance of"], e["incident"], statements) \
        or check_prop_entity(p["instance of"], e["amphibious warfare"], statements) \
        or check_prop_entity(p["instance of"], e["landing operation"], statements) \
        or check_prop_entity(p["instance of"], e["nuclear tests series"], statements) \
        or check_prop_entity(p["instance of"], e["aerial bombing"], statements) \
        or check_prop_entity(p["instance of"], e["covert operation"], statements) \
        or check_prop_entity(p["instance of"], e["perpetual war"], statements) \
        or check_prop_entity(p["instance of"], e["cold war"], statements) \
        or check_prop_entity(p["instance of"], e["synod"], statements) \
        or check_prop_entity(p["instance of"], e["famine"], statements) \
        or check_prop_entity(p["instance of"], e["intervention"], statements) \
        or check_prop_entity(p["instance of"], e["protectorate"], statements) \
        or check_prop_entity(p["instance of"], e["war front"], statements) \
        or check_prop_entity(p["instance of"], e["Imperial Conference"], statements) \
        or check_prop_entity(p["instance of"], e["army"], statements) \
        or check_prop_entity(p["instance of"], e["international crisis"], statements) \
        or check_prop_entity(p["instance of"], e["bilateral treaty"], statements) \
        or check_prop_entity(p["instance of"], e["arms race"], statements) \
        or check_prop_entity(p["instance of"], e["treaty"], statements) \
        or check_prop_entity(p["instance of"], e["alliance"], statements) \
        or check_prop_entity(p["instance of"], e["intergovernmental organization"], statements) \
        or check_prop_entity(p["instance of"], e["peace treaty"], statements) \
        or check_prop_entity(p["instance of"], e["military campaign"], statements) \
        or check_prop_entity(p["instance of"], e["world war"], statements) \
        or check_prop_entity(p["instance of"], e["historical country"], statements) \
        or check_prop_entity(p["instance of"], e["conflict"], statements) \
        or check_prop_entity(p["instance of"], e["rebel"], statements) \
        or check_prop_entity(p["subclass of"], e["rebel"], statements) \
        or check_prop_entity(p["instance of"], e["demonstration"], statements) \
        or check_prop_entity(p["instance of"], e["political organization"], statements) \
        or check_prop_entity(p["instance of"], e["military unit"], statements) \
        or check_prop_entity(p["instance of"], e["civil war"], statements) \
        or check_prop_entity(p["instance of"], e["ethnic conflict"], statements) \
        or check_prop_entity(p["instance of"], e["military unit branch class"], statements) \
        or check_prop_entity(p["instance of"], e["siege"], statements) \
        or check_prop_entity(p["instance of"], e["war"], statements) \
        or check_prop_entity(p["instance of"], e["revolution"], statements) \
        or check_prop_entity(p["instance of"], e["political party"], statements) \
        or check_prop_entity(p["instance of"], e["battle"], statements) \
        or check_prop_entity(p["instance of"], e["combat"], statements) \
        or check_prop_entity(p["instance of"], e["historical event"], statements) \
        or check_prop_entity(p["instance of"], e["military operation"], statements) \
        or check_prop_entity(p["instance of"], e["riot"], statements) \
        or check_prop_entity(p["instance of"], e["protest"], statements) \
        or check_prop_entity(p["instance of"], e["political crisis"], statements) \
        or check_prop_entity(p["part of"], e["Renaissance"], statements) \
        or check_prop_entity(p["part of"], e["classical antiquity"], statements) \
        or check_prop_entity(p["part of"], e["siege"], statements) \
        or check_prop_entity(p["part of"], e["World War I"], statements) \
        or check_prop_entity(p["part of"], e["war"], statements) \
        or check_prop_entity(p["part of"], e["ancient history"], statements) \
        or check_prop_entity(p["part of"], e["Napoleonic Wars"], statements) \
        or check_prop_entity(p["part of"], e["battle"], statements) \
        or check_prop_entity(p["part of"], e["combat"], statements) \
        or check_prop_entity(p["subclass of"], e["siege"], statements) \
        or check_prop_entity(p["subclass of"], e["war"], statements) \
        or check_prop_entity(p["subclass of"], e["battle"], statements) \
        or check_prop_entity(p["subclass of"], e["combat"], statements) \
        or check_prop_entity(p["subclass of"], e["historical event"], statements) \
        or check_prop_entity(p["subclass of"], e["military operation"], statements) \
        or check_prop_entity(p["occupation"], e["politician"], statements) \
        or check_prop_entity(p["occupation"], e["diplomat"], statements) \
        or check_prop_entity(p["occupation"], e["militant"], statements) \
        or check_prop_entity(p["occupation"], e["revolutionary"], statements) \
        or check_prop_entity(p["occupation"], e["military officer"], statements) \
        or check_prop_entity(p["studied by"], e["study of history"], statements) \
        or p["noble title"] in statements \
        or p["monogram"] in statements \
        or p["position held"] in statements \
        or p["member of political party"] in statements \
        or p["conflict"] in statements \
        or p["military branch"] in statements:
        return 1

    return 0


def get_statements(wikidata_id: str):
    session = requests.Session()

    url = "https://www.wikidata.org/w/api.php"

    # L'azione wbgetclaims viene usata per ottenere informazioni sulle entita'
    # (elementi di Wikidata) tramite i loro identificatori.
    params = {
        "action": "wbgetclaims",
        "entity": wikidata_id,
        "format": "json"
    }

    request = session.get(url=url, params=params)
    data = request.json()

    statements = defaultdict(lambda: [])
    
    for prop in data["claims"]:
        for el in data["claims"][prop]:
            if el["type"] == "statement":
                mainsnak = el["mainsnak"]
                if (mainsnak["datatype"] == "wikibase-item") \
                    and (mainsnak["snaktype"] == "value"):
                    entity = mainsnak["datavalue"]["value"]["id"]
                    statements[prop].append(entity)
    
    # Per ogni entità vengono restituite le sue proprietà e le connessioni
    # (tramite le proprietà) ad altre entità. 
    # Esempio per il dizionario: 'P27': ['Q837855', 'Q912068'].
    return dict(statements)


def check_prop_entity(
    prop_id,
    entity_id,
    statements
):
    if prop_id not in statements:
        return False
    else:
        return entity_id in statements[prop_id]
    

def process_page(
    page: dict,
    title: str,
    page_id: str,
    page_wikidata_id: str,
    page_historical: int 
):
    nlp = spacy.load("en_core_web_sm")
    
    results = []
    total_tokens = 0

    html = page["html"]
    # Il contenuto della pagina di Wikipedia è in formato HTML.
    # Quindi occorre effettuare un parsing per estrarre i paragrafi.
    soup = BeautifulSoup(html, "html.parser")

    # Qui vengono estratti i paragrafi. Ma i paragrafi contengono ancora
    # diversi tag HTML (span e riferimenti href ecc.).
    paragraphs = soup.find_all("p")
    
    for i, paragraph in tqdm(
        enumerate(paragraphs)
    ):
        # Se il paragrafo contiene la classe mw-empty-elt, allora viene ignorato
        # perché è vuoto.
        if (paragraph.has_attr("class")) \
            and ("mw-empty-elt" in paragraph.get("class")):
            continue
        paragraph = preprocess_paragraph(paragraph)
        result, token_idx = process_paragraph(
            paragraph, title, page_id, 
            page_wikidata_id, page_historical, nlp, i
        )
        results.append(result)
        total_tokens += token_idx

    return results, total_tokens


def preprocess_paragraph(paragraph: bs4.element.Tag):
    
    # Gli elementi (el) del paragrafo sono gli elementi racchiusi da tag.
    # Quindi per esempio <a href="/wiki/Historicity" title="Historicity">historicity</a>
    # e <a class="mw-redirect" href="/wiki/King_Hel%C3%BC_of_Wu" title="King Helü of Wu">King Helü of Wu</a>.
    for el in paragraph.find_all():
        # Vengono rimossi i superscripts (riferimenti alla bibliografia).
        if el.name == "sup":
            el.decompose()
            continue
        
        # Si trasforma in testo semplice ogni elemento HTML che non è un
        # anchor ("a") o che è un anchor ma non ha un titolo, che significa che
        # non è un riferimento a un'altra entità di Wikipedia. 
        if (el.name != "a") or (not el.has_attr("title")):
            if el.name == None:
                continue
            el.unwrap()
            
    return paragraph


def process_paragraph(
    paragraph: bs4.element.Tag,
    title: str,
    page_id: str,
    page_wikidata_id: str,
    page_historical: int,
    nlp: spacy.language.Language,
    par_num: int
):
    result = {
        "title": title,
        "historical": page_historical,
        "page_id": page_id,
        "wikidata_id": page_wikidata_id,
        "par_num": par_num,
        "clean_content": "",
        "entities": []
    }

    token_idx = 0

    res_tokenized = []

    for content in paragraph.contents:
        # Se c'è ancora un anchor ("a"), allora è un'entità referenziata che
        # ha anche un titolo, perché quelle senza titolo sono già state trasformate
        # in testo semplice.
        if content.name == "a":
            entity = dict()
            raw_text = content.get_text()
            # Con strip vengono rimossi gli spazi bianchi a capo e in coda.
            text = content.get_text().strip()
            title = content.get("title")
            tokens_length = len(nlp(text))

            # Controlliamo che il testo non sia vuoto per evitare di aggiungere
            # testo vuoto. Altrimenti quando si congiunge il testo tokenizzato alla
            # fine del ciclo si rischia di introdurre spazi non voluti.
            if len(text) > 0:
                res_tokenized.append(text)

            if "(page does not exist)" in title:
                token_idx += tokens_length
                continue
            
            try:
                entity_page_id, entity_wikidata_id = get_wikidata_info(title)
            except NoDataException:
                token_idx += tokens_length
                continue

            # Questa è la surface form dell'entità. Per esempio: Western world.
            entity["surface_form"] = text
            entity["page_title"] = title
            entity["page_id"] = entity_page_id
            entity["wikidata_id"] = entity_wikidata_id
            # Questo è l'indice della prima parola (token) del "content" paragrafo.
            entity["token_offset"] = token_idx
            # Questo è il numero di parole (token) del "content" del paragrafo.
            entity["tokens_length"] = tokens_length
            entity["historical"] = is_historical(entity_wikidata_id)
            token_idx += tokens_length
            result["entities"].append(entity)
        else:
            raw_text = str(content)
            text = raw_text.strip()

            # Rimuoviamo il CSS che appare nel testo e che viene usato per
            # esempio per la formattazione di coordinate geografiche e frazioni. 
            if ".mw-parser-output" in text:
                text = ""

            tokens_length = len(nlp(text))

            if len(text) > 0:
                res_tokenized.append(text)

            token_idx += tokens_length

    result["clean_content"] = " ".join(res_tokenized)

    return result, token_idx


