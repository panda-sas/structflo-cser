"""Text-based distractor elements: prose, captions, footnotes, arrows, tables, etc."""

import random
import textwrap
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

from structflo.cser._geometry import boxes_intersect, try_place_box
from structflo.cser.config import PageConfig
from structflo.cser.rendering.text import load_font

# ---------------------------------------------------------------------------
# Text corpora
# ---------------------------------------------------------------------------

PROSE_WORDS = [
    # Articles / connectives
    "the", "a", "an", "and", "with", "for", "was", "were", "has", "have",
    "been", "from", "that", "this", "these", "those", "such", "which", "when",
    "where", "while", "although", "however", "therefore", "furthermore",
    "subsequently", "additionally", "notably", "respectively", "previously",
    # Compound / structure
    "compound", "compounds", "molecule", "molecules", "structure", "structures",
    "analog", "analogs", "analogue", "analogues", "derivative", "derivatives",
    "scaffold", "scaffolds", "core", "moiety", "fragment", "fragments",
    "substituent", "substituents", "modification", "modifications",
    "series", "congener", "congeners", "isomer", "isomers", "enantiomer",
    "diastereomer", "racemate", "prodrug", "salt", "cocrystal", "polymorph",
    # Activity / potency
    "activity", "activities", "potency", "potent", "efficacy", "efficacious",
    "inhibition", "inhibitor", "inhibitors", "inhibitory", "selectivity",
    "selective", "affinity", "binding", "active", "inactive", "antagonist",
    "agonist", "partial", "allosteric", "competitive", "reversible",
    "covalent", "irreversible", "nanomolar", "micromolar", "picomolar",
    "submicromolar", "dose", "response", "concentration", "potent",
    # Biology / target
    "target", "targets", "receptor", "receptors", "enzyme", "enzymes",
    "kinase", "protease", "phosphatase", "deubiquitinase", "GPCR",
    "protein", "domain", "binding", "site", "pocket", "interface",
    "cellular", "cell", "cells", "assay", "assays", "vitro", "vivo",
    "model", "pathway", "cascade", "signal", "transduction", "apoptosis",
    "proliferation", "migration", "invasion",
    # Chemistry / synthesis
    "synthesis", "synthetic", "reaction", "reactions", "yield", "yields",
    "purity", "purified", "isolated", "crude", "chromatography", "column",
    "method", "procedure", "conditions", "reagent", "reagents", "catalyst",
    "solvent", "temperature", "stirred", "reflux", "cooled", "quenched",
    "workup", "extraction", "filtration", "recrystallization", "evaporated",
    "concentrated", "dissolved", "added", "obtained", "afforded", "prepared",
    # ADME / PK
    "solubility", "permeability", "bioavailability", "oral", "plasma",
    "clearance", "half-life", "metabolite", "metabolites", "metabolic",
    "stability", "stable", "distribution", "absorption", "elimination",
    "excretion", "hepatic", "renal", "microsomal", "fraction", "unbound",
    "protein", "efflux", "transporter", "CYP", "glucuronide",
    # SAR / physical chem
    "substitution", "electron", "donating", "withdrawing", "lipophilic",
    "hydrophilic", "polar", "nonpolar", "aromatic", "aliphatic", "cyclic",
    "heterocyclic", "fluorine", "nitrogen", "oxygen", "hydrogen", "carbon",
    "methyl", "ethyl", "propyl", "butyl", "phenyl", "benzyl", "chloro",
    "fluoro", "bromo", "hydroxyl", "amino", "cyano", "nitro", "sulfonyl",
    # Results / analysis language
    "results", "indicate", "show", "demonstrate", "suggest", "confirm",
    "observed", "found", "reported", "measured", "determined", "calculated",
    "analysis", "data", "values", "range", "average", "mean", "standard",
    "deviation", "significant", "increase", "decrease", "fold", "improvement",
    "reduction", "enhancement", "loss", "gain", "shift", "maintained",
    "retained", "improved", "diminished", "abolished", "enhanced",
    # Table / figure language
    "table", "figure", "scheme", "panel", "chart", "graph", "plot",
    "supplementary", "shown", "listed", "summarized", "depicted",
    # Drug discovery process
    "screening", "library", "virtual", "docking", "pharmacophore",
    "FBDD", "HTS", "primary", "secondary", "counter", "profiling",
    "optimization", "candidate", "clinical", "preclinical", "development",
    "campaign", "progression", "advancement", "nomination",
    # Physical chemistry
    "logP", "logD", "pKa", "molecular", "weight", "formula", "melting",
    "crystal", "amorphous", "hygroscopic", "photostable", "thermally",
]

CAPTION_TEMPLATES = [
    # Synthesis / scheme
    "Synthesis of target compounds.",
    "Synthesis of key intermediates.",
    "Synthetic route to the lead scaffold.",
    "Retrosynthetic analysis of target compounds.",
    "Overview of the convergent synthesis.",
    "Multi-step synthesis of analogs.",
    "General synthetic approach for ring-substituted derivatives.",
    "Optimized synthesis of the lead series.",
    "Modular synthesis of compound library.",
    "One-pot synthesis of bicyclic intermediates.",
    # Structures
    "Representative structures from screening.",
    "Selected compounds and their structures.",
    "Core scaffold and key analogs.",
    "Structures of potent inhibitors identified.",
    "Chemical structures and physicochemical properties.",
    "Structural diversity in the compound set.",
    "Key compounds in the optimization campaign.",
    "Structures of matched molecular pairs.",
    "Scaffold variation at position C-4.",
    "Bioisosteric replacements explored.",
    "Overview of reaction scheme.",
    "Chemical structures and labels.",
    # SAR
    "SAR summary for the lead series.",
    "SAR analysis at position R1.",
    "Structure-activity relationships at C-4.",
    "Matched molecular pair analysis.",
    "Effect of substituents on potency.",
    "Key SAR findings highlighted.",
    "Impact of fluorination on activity.",
    "SAR at the amide nitrogen.",
    "Effect of ring size on selectivity.",
    "Halogen scan results at position 4.",
    # Biology / assay
    "Dose-response curves for selected compounds.",
    "Concentration-response relationships.",
    "Selectivity panel data for top compounds.",
    "Cellular activity of lead compounds.",
    "In vitro ADME profile of selected analogs.",
    "Metabolic stability comparison across species.",
    "Permeability data (PAMPA assay).",
    "CYP inhibition panel results.",
    "hERG inhibition data for the series.",
    "Plasma protein binding results.",
    # Protein / structural biology
    "X-ray crystal structure of the protein-ligand complex.",
    "Co-crystal structure with bound inhibitor.",
    "Key binding interactions with the target protein.",
    "Docking pose of compound in the active site.",
    "Overlay of co-crystal structures.",
    "Surface representation of the binding pocket.",
    "Hydrogen bond network in the active site.",
    # Mechanism / PK
    "Proposed mechanism of action.",
    "Mechanism-based inactivation study.",
    "Mouse PK profile of selected compounds.",
    "Rat oral pharmacokinetic parameters.",
    "Plasma concentration vs. time curves.",
    "Blood-brain barrier penetration data.",
    "Metabolic stability across species.",
    # MMP / comparison
    "Matched molecular pair transformations and delta-activity.",
    "Comparison of ring systems on potency.",
    "Scaffold hopping results.",
    "Bioisostere screening results.",
    # Misc
    "Key physicochemical parameters of synthesized compounds.",
    "Lead compound series ranked by potency.",
    "Overview of the discovery campaign milestones.",
    "Crystallographic data collection and refinement statistics.",
    "Compounds selected for in vivo profiling.",
    "Summary of optimization across three subseries.",
    "Heatmap of selectivity across the kinome.",
    "Free energy perturbation results for key analogs.",
]

STRAY_FRAGMENTS = [
    # Journal / DOI refs
    "J Med Chem", "J Med Chem 2024, 67, 1234–1251",
    "DOI:10.1000/xyz", "DOI:10.1021/acs.jmedchem.4c00123",
    "DOI:10.1016/j.ejmech.2024.116412", "DOI:10.1039/d4md00321g",
    "Bioorg Med Chem Lett", "Eur J Med Chem 2023, 258, 115584",
    "ACS Med Chem Lett", "Drug Discov Today 2024, 29, 103891",
    "ChemMedChem 2023, 18, e202300201", "Nat Chem Biol 2024",
    "JACS 2024, 146, 8912", "Org Lett 2024, 26, 3412",
    # NMR data
    "1H NMR (400 MHz, CDCl3)", "1H NMR (500 MHz, DMSO-d6)",
    "13C NMR (101 MHz, CDCl3)", "13C NMR (125 MHz, DMSO-d6)",
    "19F NMR (376 MHz, CDCl3)", "31P NMR (162 MHz, D2O)",
    "HRMS (ESI+): calcd for C18H20N3O2", "HRMS (ESI-) m/z calcd",
    "MS (ESI+) m/z: 412.3 [M+H]+", "MS (APCI) m/z: 356.2",
    "1H NMR (400 MHz)", "13C NMR (101 MHz)",
    # Physical data
    "mp 142-145 °C", "mp 198-201 °C (dec.)", "mp >250 °C",
    "mp 88-90 °C", "mp 231-234 °C", "bp 78 °C (0.5 mmHg)",
    "[α]D = -23.5", "[α]D20 = +14.2 (c 1.0, MeOH)",
    "[α]D25 = -8.7 (c 0.5, CHCl3)",
    "Rf = 0.45 (EtOAc/hexane 3:7)", "Rf = 0.32 (DCM/MeOH 9:1)",
    # Compound metadata
    "HPLC purity >95%", "HPLC purity >98% (UV 254 nm)",
    "purity: 97.3% (ELSD)", ">99% ee (chiral HPLC)",
    "ee = 96%", "dr = 9:1", "purity >99% (HPLC)",
    "HPLC purity: 95.2% (220 nm)",
    # Physical chemistry
    "pKa = 7.4", "cLogP 3.2", "LogD7.4 = 1.8",
    "MW = 412.5", "tPSA = 78 Å²", "HBD = 2, HBA = 5",
    "sol. = 124 µg/mL (PBS pH 7.4)", "kin. sol. = 48 µM",
    "cLogP = 2.9, tPSA = 67 Å²", "MW = 389.4 Da",
    "logD = 2.1", "pKa (amine) = 8.3",
    # Dates / publication status
    "Received: Jan 2025", "Accepted: Mar 2025", "Published online",
    "Received: 14 October 2024", "Accepted: 9 December 2024",
    "Revised: 22 November 2024", "First published: 3 Feb 2025",
    "Epub ahead of print", "In press", "Corrected proof",
    # Supplementary / admin
    "Supporting Information", "Author Manuscript", "CONFIDENTIAL",
    "Supplementary Table S1", "Table S3", "Figure S7", "Scheme S2",
    "SI page 23", "See SI for details", "Full data in Supplementary",
    "Supplementary", "Table S1", "Scheme 3", "Rev. 2021",
    # Patent / IP
    "Patent WO2025/123456", "WO 2024/098765 A1",
    "US 11,234,567 B2", "EP 3,456,789 A1",
    "© 2025 ACS", "© 2024 Elsevier Ltd.", "All rights reserved.",
    "Open Access CC-BY 4.0", "CC BY-NC 4.0",
    # Hard negatives: ID-like strings NOT being annotated
    "See CHEMBL4051 for details", "ZINC00123456 (inactive)",
    "Ref: PUBCHEM2341157", "cf. MCULE-1234567",
    "CPD-00441 showed no activity", "data from MOL-98231",
    "activity vs. ENAMINE-T001", "HIT-00923 excluded",
    "IC50 values: STD-00001–STD-00010",
    "Selected from ZINC library", "PUBCHEM CID: 44259",
    "CHEMBL dataset ver. 33", "cross-ref: ZINC15",
    "ChEMBL ID: 2109628", "see MCULE catalog entry",
    "CPD-03341 not tested", "inactive: MOL-00023",
    "from ENAMINE stock", "see also HIT-04410",
    # Analytical conditions
    "Method A: 5–95% MeCN over 2 min", "Gradient: H2O/MeCN 10–90%",
    "Column: C18, 2.1 × 50 mm, 1.7 µm", "Flow rate: 1.5 mL/min",
    "UV detection at 214 nm", "ELSD detection", "PDA detector",
    "Column: BEH C18, 50 × 2.1 mm", "run time: 3.5 min",
    # Internal document markers
    "Draft v1.3 — Do Not Distribute", "INTERNAL USE ONLY",
    "Compound Registration #CR-2024-0891", "CRO: Evotec",
    "Project: ONC-2024", "Batch: 2024-BCH-0042",
    "Notebook: ELN-44321, page 87", "Analyst: J. Smith",
    "Reviewed by: QC Dept.", "Approved: 2025-01-15",
    "Study ID: PK-2024-0087", "Protocol: TOX-2024-113",
    # Other stray text
    "n = 3 independent experiments", "Mean ± SD (n=3)",
    "All values are mean ± SEM.", "Determined by ITC.",
    "Version 3.1", "Last modified: Dec 2024",
    "Prepared by: Discovery Chemistry", "CAS No. 50-78-2",
    "InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
]

HEADER_TEMPLATES = [
    "Abstract", "Introduction", "Methods", "Results", "Discussion",
    "Experimental Section", "Conclusions", "References", "Supplementary Information",
    "Materials and Methods", "Acknowledgments", "Author Contributions",
]

FOOTNOTE_TEMPLATES = [
    # Corresponding author / affiliation
    "* Corresponding author. Email: author@university.edu",
    "* To whom correspondence should be addressed.",
    "* E-mail: jsmith@pharma.com. Tel: +1-617-555-0199.",
    "* Fax: +44-20-7679-1234.",
    "* Present address: Novartis Institutes for BioMedical Research.",
    "* Contact: lead.author@cam.ac.uk",
    # Author contribution symbols
    "† These authors contributed equally.",
    "† Equal contribution.",
    "† These authors contributed equally to this work.",
    "‡ Current address: Department of Chemistry, MIT.",
    "‡ Deceased.",
    "‡ Current address: Pfizer Global R&D, Groton, CT.",
    "§ Electronic supplementary information (ESI) available.",
    "§ On leave from the University of Tokyo.",
    "¶ Current address: AstraZeneca, Gothenburg, Sweden.",
    "|| These authors are co-senior authors.",
    # Reagents / conditions footnotes
    "a) Reagents and conditions: see text.",
    "a) Reagents and conditions: (i) NaH, DMF, 0 °C; (ii) RBr, rt, 12 h.",
    "a) All reactions performed under nitrogen atmosphere.",
    "b) Isolated yield after column chromatography.",
    "b) Yield over two steps.",
    "b) Crude yield; not further purified.",
    "c) Determined by chiral HPLC.",
    "c) Conversion determined by 1H NMR.",
    "d) Performed at 10 µM compound concentration.",
    "d) Average of two independent runs.",
    "e) Values are mean ± SD of three experiments.",
    "e) Single determination.",
    "f) Assayed at Eurofins Cerep.",
    "f) Not tested due to insufficient material.",
    # Abbreviations / notes
    "Abbreviations: SAR, structure-activity relationship; PK, pharmacokinetics.",
    "N.D. = not determined; N.T. = not tested.",
    "All IC50 values represent geometric means of ≥2 experiments.",
    "Selectivity index = IC50(off-target) / IC50(primary target).",
    "Compounds were tested as racemates unless otherwise noted.",
    "Purity of all final compounds was ≥95% by HPLC.",
]

EQUATION_FRAGMENTS = [
    # IC50 / Ki / EC50
    "IC50 = 3.2 ± 0.5 nM",
    "IC50 = 0.45 µM",
    "IC50 = 14 ± 2 nM",
    "IC50 = 230 nM (enzymatic)",
    "IC50 = 1.8 µM (cellular)",
    "IC50 > 10 µM",
    "IC50 < 1 nM",
    "IC50 = 56 ± 8 nM (n=4)",
    "Ki = 12.4 ± 1.1 µM",
    "Ki = 0.9 nM",
    "Ki = 340 nM (competitive)",
    "Ki(app) = 23 nM",
    "EC50 = 0.8 µM (n=3)",
    "EC50 = 120 nM",
    "EC50 = 3.4 ± 0.6 µM",
    # Kd / binding
    "Kd = 45 nM",
    "Kd = 2.1 ± 0.3 µM (ITC)",
    "Kd = 0.18 nM (SPR)",
    "KD = 380 nM (BLI)",
    "kon = 1.2 × 10⁶ M⁻¹s⁻¹",
    "koff = 3.4 × 10⁻³ s⁻¹",
    "kinact/KI = 4500 M⁻¹s⁻¹",
    # Thermodynamics
    "ΔG = -8.3 kcal/mol",
    "ΔH = -12.1 kcal/mol",
    "−TΔS = 3.8 kcal/mol",
    "ΔΔG = -1.4 kcal/mol",
    "ΔGbind = -9.7 kcal/mol",
    # Physical chemistry
    "logP = 2.14",
    "logD7.4 = 1.53",
    "cLogP = 3.8",
    "pKa = 6.2",
    "pKa(NH) = 9.1",
    "MW = 423.5 g/mol",
    "MW = 387.4 Da",
    "tPSA = 84 Å²",
    "HBD = 2, HBA = 6",
    "Fsp3 = 0.42",
    "LE = 0.38 kcal/mol/heavy atom",
    "LLE = 5.2",
    "LLEAT = 0.45",
    # PK / ADME parameters
    "t1/2 = 4.2 h",
    "t1/2 = 11.3 h (human)",
    "t1/2 = 0.8 h (mouse)",
    "AUC = 1240 ng·h/mL",
    "AUC0-inf = 3400 ng·h/mL",
    "Cmax = 890 ng/mL",
    "Tmax = 1.5 h",
    "CL = 12 mL/min/kg",
    "CLint = 45 µL/min/mg",
    "CLhep = 18.2 mL/min/kg",
    "Vdss = 2.3 L/kg",
    "F = 34%",
    "F(oral) = 62%",
    "%PPB = 99.2%",
    "fu = 0.008",
    "Papp = 18.4 × 10⁻⁶ cm/s",
    "ER = 1.2 (MDCK-MDR1)",
    # Inhibition / activity
    "%inh = 87 ± 3%",
    "%inh @ 1 µM = 94%",
    "%inh @ 10 µM = 42%",
    "%ctrl = 8 ± 2%",
    "Emax = 95%",
    "Hill slope = 1.1",
    "nH = 0.9",
    # Selectivity
    "SI = 120 (CYP3A4/2D6)",
    "selectivity > 1000-fold",
    "IC50 ratio = 450×",
    # Solubility / stability
    "sol. = 85 µg/mL (pH 7.4)",
    "kin. sol. = 210 µM",
    "t1/2(mic, human) = 45 min",
    "t1/2(mic, rat) = 12 min",
    "t1/2(plasma, mouse) = >120 min",
]

JOURNAL_HEADERS = [
    # ACS journals
    "Journal of Medicinal Chemistry",
    "Journal of the American Chemical Society",
    "ACS Chemical Biology",
    "ACS Med. Chem. Lett.",
    "ACS Medicinal Chemistry Letters",
    "ACS Pharmacol. Transl. Sci.",
    "Journal of Chemical Information and Modeling",
    "Organic Letters",
    "ACS Omega",
    "ACS Catalysis",
    "ACS Chemical Neuroscience",
    "ACS Infectious Diseases",
    "Molecular Pharmaceutics",
    # Elsevier journals
    "Bioorganic & Medicinal Chemistry Letters",
    "Bioorganic & Medicinal Chemistry",
    "European Journal of Medicinal Chemistry",
    "Bioorganic Chemistry",
    "Drug Discovery Today",
    "European Journal of Pharmaceutical Sciences",
    "Tetrahedron Letters",
    "Tetrahedron",
    # Wiley journals
    "Angewandte Chemie Int. Ed.",
    "ChemMedChem",
    "Chemistry – A European Journal",
    "Archiv der Pharmazie",
    "Molecular Informatics",
    "ChemBioChem",
    # RSC journals
    "Chemical Communications",
    "Chemical Science",
    "MedChemComm",
    "RSC Medicinal Chemistry",
    "Organic & Biomolecular Chemistry",
    "RSC Advances",
    # Nature / Springer
    "Nature Chemical Biology",
    "Nature Reviews Drug Discovery",
    "Nature Chemistry",
    "Scientific Reports",
    "Journal of Natural Products",
    # Other publishers / journals
    "Proceedings of the National Academy of Sciences",
    "Science",
    "Cell Chemical Biology",
    "PLOS ONE",
    "Pharmaceuticals",
    "Molecules",
    "International Journal of Molecular Sciences",
    "Frontiers in Chemistry",
    "Medicinal Research Reviews",
]

# ---------------------------------------------------------------------------
# Distractor drawing functions
# ---------------------------------------------------------------------------


def add_prose_block(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.prose_block_prob:
        return

    w, h = page.size
    font_size = random.randint(14, 18)
    font = load_font(font_paths, font_size)
    line_h = font_size + 4
    n_lines = random.randint(8, 22)
    block_w = random.randint(w // 3, w * 3 // 4)
    block_h = n_lines * line_h

    box = try_place_box(w, h, block_w, block_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    total_words = n_lines * random.randint(6, 10)
    words = [random.choice(PROSE_WORDS) for _ in range(total_words)]
    paragraph = " ".join(words).capitalize() + "."
    avg_char_w = font_size * 0.55
    chars_per_line = max(20, int(block_w / avg_char_w))
    lines = textwrap.wrap(paragraph, width=chars_per_line)[:n_lines]
    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * line_h), line, font=font, fill=(0, 0, 0))


def add_caption(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.caption_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(16, 20))
    draw = ImageDraw.Draw(page)
    cap = f"Figure {random.randint(1, 12)}. {random.choice(CAPTION_TEMPLATES)}"

    text_bbox = draw.textbbox((0, 0), cap, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1] + 4

    box = try_place_box(w, h, text_w, text_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), cap, font=font, fill=(0, 0, 0))


def add_arrow(
    page: Image.Image,
    cfg: PageConfig,
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.arrow_prob:
        return

    w, h = page.size
    length = random.randint(60, 140)
    arrow_h = 16

    box = try_place_box(w, h, length, arrow_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    mid_y = y0 + arrow_h // 2
    x1 = x0 + length
    draw.line((x0, mid_y, x1, mid_y), fill=(0, 0, 0), width=2)
    draw.line((x1, mid_y, x1 - 10, mid_y - 6), fill=(0, 0, 0), width=2)
    draw.line((x1, mid_y, x1 - 10, mid_y + 6), fill=(0, 0, 0), width=2)


def add_panel_border(
    page: Image.Image,
    cfg: PageConfig,
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.panel_border_prob:
        return

    draw = ImageDraw.Draw(page)
    w, h = page.size
    x0 = random.randint(cfg.margin, w // 2)
    y0 = random.randint(cfg.margin, h // 2)
    x1 = random.randint(x0 + 200, w - cfg.margin)
    y1 = random.randint(y0 + 200, h - cfg.margin)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)


def add_page_number(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.page_number_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(16, 20))
    draw = ImageDraw.Draw(page)
    number = str(random.randint(1, 250))
    px = w - cfg.margin - 40
    py = h - cfg.margin + 10
    draw.text((px, py), number, font=font, fill=(0, 0, 0))


def add_rgroup_table(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.rgroup_table_prob:
        return

    w, h = page.size
    cols = random.randint(2, 4)
    rows = random.randint(3, 6)
    cell_w = random.randint(60, 100)
    cell_h = random.randint(30, 50)
    table_w = cols * cell_w
    table_h = rows * cell_h

    box = try_place_box(w, h, table_w, table_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    for i in range(cols + 1):
        draw.line((x0 + i * cell_w, y0, x0 + i * cell_w, y0 + rows * cell_h),
                  fill=(0, 0, 0), width=1)
    for j in range(rows + 1):
        draw.line((x0, y0 + j * cell_h, x0 + cols * cell_w, y0 + j * cell_h),
                  fill=(0, 0, 0), width=1)

    font = load_font(font_paths, 14)
    cell_choices = [
        "R1", "R2", "R3", "H", "Me", "Cl", "Br", "F", "OH", "OMe", "NH2", "CF3",
        "Et", "iPr", "nBu", "Ph", "Bn", "CN", "NO2", "SO2Me",
        "n/a", ">100", "<0.1", "3.2", "14.5",
    ]
    id_choices = [
        "CHEMBL4051", "ZINC00123", "CPD-00441", "MOL-98231",
        "HIT-00923", "STD-00001", "MCULE-001", "PUBCHEM44",
    ]
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or c == 0) and random.random() < 0.25:
                txt = random.choice(id_choices)
            else:
                txt = random.choice(cell_choices)
            draw.text((x0 + c * cell_w + 6, y0 + r * cell_h + 6), txt,
                      font=font, fill=(0, 0, 0))


def add_stray_text(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.stray_text_prob:
        return

    w, h = page.size
    font_size = random.randint(12, 16)
    font = load_font(font_paths, font_size)
    text = random.choice(STRAY_FRAGMENTS)

    draw = ImageDraw.Draw(page)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0] + 4
    text_h = text_bbox[3] - text_bbox[1] + 4

    box = try_place_box(w, h, text_w, text_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


def add_section_header(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a bold section header like 'Abstract', 'Methods', etc."""
    w, h = page.size
    text = random.choice(HEADER_TEMPLATES)
    font_size = random.randint(20, 30)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    box = try_place_box(w, h, tw, th, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


def add_footnote(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a footnote-style text near the bottom of the page."""
    w, h = page.size
    text = random.choice(FOOTNOTE_TEMPLATES)
    font_size = random.randint(10, 14)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    for _ in range(30):
        x0 = random.randint(cfg.margin, max(cfg.margin, w - cfg.margin - tw))
        y0 = random.randint(max(cfg.margin, h * 3 // 4), max(cfg.margin, h - cfg.margin - th))
        candidate = (x0, y0, x0 + tw, y0 + th)
        padded = (x0 - 8, y0 - 8, x0 + tw + 8, y0 + th + 8)
        if not any(boxes_intersect(padded, b) for b in existing):
            existing.append(candidate)
            draw.text((x0, y0), text, font=font, fill=(80, 80, 80))
            return


def add_equation_fragment(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a small scientific equation / measurement fragment."""
    w, h = page.size
    text = random.choice(EQUATION_FRAGMENTS)
    font_size = random.randint(14, 20)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    box = try_place_box(w, h, tw, th, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


def add_journal_header(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a journal-name style header near the top of the page."""
    w, h = page.size
    text = random.choice(JOURNAL_HEADERS)
    font_size = random.randint(14, 20)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    for _ in range(30):
        x0 = random.randint(cfg.margin, max(cfg.margin, w - cfg.margin - tw))
        y0 = random.randint(10, min(cfg.margin + 60, h - th))
        candidate = (x0, y0, x0 + tw, y0 + th)
        padded = (x0 - 8, y0 - 8, x0 + tw + 8, y0 + th + 8)
        if not any(boxes_intersect(padded, b) for b in existing):
            existing.append(candidate)
            draw.text((x0, y0), text, font=font, fill=(60, 60, 60))
            return


def add_multiline_text_block(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a multi-line block of random scientific-ish text."""
    w, h = page.size
    font_size = random.randint(12, 16)
    font = load_font(font_paths, font_size)
    line_h = font_size + 3
    n_lines = random.randint(3, 8)
    block_w = random.randint(w // 3, w * 2 // 3)
    block_h = n_lines * line_h

    box = try_place_box(w, h, block_w, block_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    extended = PROSE_WORDS + [
        "inhibitor", "selectivity", "potent", "analog", "derivative",
        "molecular", "weight", "peak", "area", "concentration",
        "dose", "response", "curve", "receptor", "antagonist",
        "pharmacokinetic", "bioavailability", "clearance", "metabolite",
    ]
    total_words = n_lines * random.randint(4, 8)
    words = [random.choice(extended) for _ in range(total_words)]
    paragraph = " ".join(words).capitalize() + "."
    avg_char_w = font_size * 0.55
    chars_per_line = max(20, int(block_w / avg_char_w))
    lines = textwrap.wrap(paragraph, width=chars_per_line)[:n_lines]
    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * line_h), line, font=font, fill=(0, 0, 0))
