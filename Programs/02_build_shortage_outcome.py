"""
02_build_shortage_outcome.py - Construct shortage targets per NDC-month.

Uses ASHP/UUDIS drug shortage data (Erin Fox dataset) as the primary source.
Matches ASHP drug names to the panel skeleton's NONPROPRIETARYNAME and
DOSAGEFORMNAME fields using multi-pass name matching with dosage form hints.

IMPORTANT - Target interpretation:
  `shortage_start` marks the month of ASHP's `date_notified`, which is when
  UUDIS pharmacists were notified and verified a national-level shortage.
  This typically precedes FDA's initial_posting_date by days to weeks.
  Downstream models predict ASHP-verified shortage onset.

Ongoing ASHP shortages (status = 'Active') are treated as right-censored:
- they remain active through `STUDY_END`
- they do not emit a synthetic `shortage_end = 1`
- `months_remaining` is left missing because the true resolution date is unknown

Output: Data/intermediate/shortage_outcome.parquet
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("utilities", Path(__file__).parent / "00_utilities.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})


# ---------------------------------------------------------------------------
# ASHP name parsing and matching
# ---------------------------------------------------------------------------

# Dosage form keywords in ASHP names → panel DOSAGEFORMNAME patterns
FORM_KEYWORDS = {
    'injection':    ['INJECTION'],
    'injectable':   ['INJECTION'],
    'inj':          ['INJECTION'],
    ' iv':          ['INJECTION'],
    'intravenous':  ['INJECTION'],
    'infusion':     ['INJECTION'],
    'tablet':       ['TABLET'],
    'tabs':         ['TABLET'],
    'capsule':      ['CAPSULE'],
    'cream':        ['CREAM'],
    'ointment':     ['OINTMENT'],
    'gel':          ['GEL'],
    'lotion':       ['LOTION'],
    'solution':     ['SOLUTION'],
    'oral':         ['TABLET', 'CAPSULE', 'SOLUTION', 'SUSPENSION', 'SYRUP'],
    'suspension':   ['SUSPENSION'],
    'syrup':        ['SYRUP'],
    'elixir':       ['ELIXIR'],
    'suppository':  ['SUPPOSITORY'],
    'suppositories': ['SUPPOSITORY'],
    'patch':        ['PATCH'],
    'transdermal':  ['PATCH'],
    'inhalation':   ['INHALANT', 'AEROSOL', 'POWDER'],
    'inhaler':      ['INHALANT', 'AEROSOL'],
    'nasal':        ['SPRAY', 'SOLUTION'],
    'ophthalmic':   ['SOLUTION', 'SUSPENSION'],
    'otic':         ['SOLUTION', 'SUSPENSION'],
    'topical':      ['CREAM', 'OINTMENT', 'GEL', 'LOTION', 'SOLUTION'],
    'spray':        ['SPRAY'],
    'enema':        ['ENEMA'],
    'implant':      ['IMPLANT'],
    'powder':       ['POWDER'],
    'drops':        ['SOLUTION'],
    'irrigation':   ['IRRIGANT', 'SOLUTION', 'INJECTION'],
    'syringe':      ['INJECTION'],
    'syringes':     ['INJECTION'],
    'vial':         ['INJECTION'],
    'vials':        ['INJECTION'],
    'bag':          ['INJECTION'],
    'bags':         ['INJECTION'],
    'ampule':       ['INJECTION'],
    'pen':          ['INJECTION'],
    'prefilled':    ['INJECTION'],
    'metered dose': ['AEROSOL'],
    'metered-dose': ['AEROSOL'],
    'chewable':     ['TABLET'],
    'odt':          ['TABLET'],
    'sublingual':   ['TABLET'],
    'lozenge':      ['LOZENGE'],
    'pastille':     ['LOZENGE'],
    'film':         ['FILM'],
    'jelly':        ['GEL'],
    'paste':        ['PASTE'],
    'concentrate':  ['SOLUTION'],
    'liquid':       ['SOLUTION', 'SUSPENSION', 'SYRUP', 'ELIXIR'],
    'vaginal':      ['INSERT', 'CREAM', 'TABLET', 'SUPPOSITORY'],
    'rectal':       ['SUPPOSITORY', 'ENEMA'],
    'kit':          ['KIT'],
    'aerosol':      ['AEROSOL'],
    'foam':         ['AEROSOL'],
    'tincture':     ['TINCTURE', 'SOLUTION'],
    'emulsion':     ['EMULSION', 'INJECTION'],
    'pellet':       ['PELLET'],
    'granule':      ['GRANULE'],
    'capsule, extended': ['CAPSULE, EXTENDED RELEASE'],
    'extended-release capsule': ['CAPSULE, EXTENDED RELEASE'],
    'extended release capsule': ['CAPSULE, EXTENDED RELEASE'],
    'extended-release tablet': ['TABLET, EXTENDED RELEASE'],
    'extended release tablet': ['TABLET, EXTENDED RELEASE'],
    'delayed release': ['CAPSULE, DELAYED RELEASE', 'TABLET, DELAYED RELEASE'],
}

# Words to strip when extracting the base drug name
STRIP_WORDS = {
    'injection', 'injectable', 'inj', 'tablet', 'tablets', 'tabs', 'capsule',
    'capsules', 'oral', 'topical', 'cream', 'ointment', 'solution', 'suspension',
    'syrup', 'patch', 'patches', 'suppository', 'suppositories', 'inhalation',
    'inhaler', 'inhalers', 'nasal', 'ophthalmic', 'otic', 'rectal', 'vaginal',
    'transdermal', 'sublingual', 'powder', 'drops', 'gel', 'lotion', 'spray',
    'enema', 'implant', 'insert', 'ring', 'film', 'liquid', 'concentrate',
    'elixir', 'iv', 'im', 'sq', 'vials', 'vial', 'syringe', 'syringes',
    'bags', 'bag', 'infusion', 'prefilled', 'pen', 'pens', 'kit', 'kits',
    'ampul', 'ampule', 'large', 'volume', 'small', 'flush', 'chewable',
    'extended-release', 'extended', 'release', 'delayed-release', 'delayed',
    'immediate-release', 'immediate', 'sustained-release', 'sustained',
    'odt', 'er', 'sr', 'cr', 'xl', 'xr', 'la', 'dr',
    'lozenge', 'lozenges', 'pastilles', 'jelly', 'paste', 'premixed', 'premixes',
    'frozen', 'soln', 'carpujects', 'spansules',
    'metered', 'dose', 'metered-dose', 'intravenous', 'intramuscular',
    'subcutaneous', 'pediatric', 'adult', 'unit', 'preservative', 'free',
    'presentations', 'formulations', 'products', 'generic',
}


def extract_form_hints(name_lower):
    """Extract dosage form hints from an ASHP drug name.

    Uses a two-tier approach: specific form words (tablet, capsule,
    suspension, etc.) take priority over broad category words (oral,
    topical, etc.). This prevents "oral suspension" from matching tablets.
    """
    # Tier 1: specific form words
    SPECIFIC_KEYWORDS = {
        'injection', 'injectable', 'inj', ' iv', 'intravenous', 'infusion',
        'tablet', 'tabs', 'capsule', 'cream', 'ointment', 'gel', 'lotion',
        'solution', 'suspension', 'syrup', 'elixir', 'suppository',
        'suppositories', 'patch', 'spray', 'enema', 'implant', 'powder',
        'drops', 'syringe', 'syringes', 'vial', 'vials', 'bag', 'bags',
        'ampule', 'pen', 'prefilled', 'lozenge', 'pastille', 'film',
        'jelly', 'paste', 'concentrate', 'kit',
        'aerosol', 'foam', 'tincture', 'emulsion', 'pellet', 'granule',
        'liquid', 'inhalation', 'inhaler',
    }
    # Tier 2: broad category words (only used if no specific match)
    BROAD_KEYWORDS = {
        'oral', 'topical', 'nasal', 'ophthalmic', 'otic', 'vaginal',
        'rectal', 'transdermal', 'sublingual', 'chewable', 'odt',
        'metered dose', 'metered-dose', 'irrigation',
        'extended-release capsule', 'extended release capsule',
        'capsule, extended', 'extended-release tablet',
        'extended release tablet', 'delayed release',
    }

    specific_forms = set()
    broad_forms = set()
    for keyword, forms in FORM_KEYWORDS.items():
        if len(keyword) <= 3:
            if not re.search(r'\b' + re.escape(keyword) + r'\b', name_lower):
                continue
        else:
            if keyword not in name_lower:
                continue
        if keyword in SPECIFIC_KEYWORDS:
            specific_forms.update(forms)
        elif keyword in BROAD_KEYWORDS:
            broad_forms.update(forms)

    return specific_forms if specific_forms else broad_forms


def extract_base_name(name):
    """Extract the base drug name from an ASHP drug name.

    Removes brand names in parentheses, strengths, dosage form words,
    and container descriptions. Preserves leading concentration prefixes
    (e.g. "0.9%" in "0.9% Sodium chloride") since these are part of
    the drug name in the NDC directory.
    """
    if pd.isna(name):
        return ''
    n = str(name).strip().lower()
    # Remove parenthetical brand names like (Campral), (EpiEZPen)
    n = re.sub(r'\([^)]*\)', '', n).strip()
    # Preserve leading concentration (e.g. "0.9%", "5%", "23.4%")
    leading_pct = ''
    pct_match = re.match(r'^(\d+\.?\d*%)\s+', n)
    if pct_match:
        leading_pct = pct_match.group(1) + ' '
        n = n[pct_match.end():]
    # Remove mid/trailing strength patterns: "5 mg", "250 mL", "100 units"
    n = re.sub(r'\d+\.?\d*\s*(?:mg|mcg|g|ml|meq|units?|iu|gram|grams|mcg/ml|mg/ml|mg/g|mcg/hr|mg/hr)\b', '', n, flags=re.IGNORECASE)
    # Remove mid/trailing percentage strengths (not leading)
    n = re.sub(r'\d+\.?\d*\s*%', '', n)
    # Remove size descriptions: "15gram Tube", "5 oz bottles", "1 ml"
    n = re.sub(r'\d+\.?\d*\s*(?:oz|ml|cc|l|gram|grams|tube|bottles?|ampules?|pk)\b', '', n, flags=re.IGNORECASE)
    # Remove schedule info
    n = re.sub(r'\bc-(?:ii|iii|iv|v)\b', '', n, flags=re.IGNORECASE)
    # Remove remaining numeric-only tokens (leftover strengths like "325", "0.5")
    n = re.sub(r'\b\d+\.?\d*\b', '', n)
    # Tokenize and remove form/container words
    tokens = n.split()
    base_tokens = [t for t in tokens if t.strip(',-/;') not in STRIP_WORDS and len(t.strip(',-/;')) > 0]
    # Remove orphan connectors left behind after number/form stripping
    while base_tokens and base_tokens[-1].strip(',-') in ('and', 'or', 'with', 'for', 'in', '-'):
        base_tokens.pop()
    result = leading_pct + ' '.join(base_tokens).strip(' ,;-/')
    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result if result else str(name).strip().lower()


def normalize_reason(reason):
    """Normalize ASHP shortage reason strings into canonical categories."""
    if pd.isna(reason):
        return 'Unknown'
    r = str(reason).strip().lower()
    if not r or r == 'nan':
        return 'Unknown'
    if r in ('unknown', 'unknown reason', 'not known', 'not available'):
        return 'Unknown'
    if 'discontinu' in r or 'withdrawn' in r or 'market withdrawal' in r:
        return 'Discontinuation'
    if 'demand' in r or 'supply/demand' in r or 'increased demand' in r:
        return 'Demand increase'
    if 'manufactur' in r:
        return 'Manufacturing'
    if 'raw material' in r or 'active ingredient' in r or 'api' in r or 'active pharmaceutical' in r:
        return 'Raw material/API'
    if 'regulatory' in r or 'gmp' in r or 'compliance' in r or 'fda' in r:
        return 'Regulatory/GMP'
    if 'shipping' in r or 'distribution' in r or 'logistic' in r:
        return 'Shipping/Distribution'
    if 'hurricane' in r or 'disaster' in r or 'natural' in r or 'earthquake' in r or 'storm' in r:
        return 'Natural disaster'
    if 'recall' in r or 'quality' in r or 'contamina' in r or 'sterility' in r or 'particulate' in r:
        return 'Quality/Recall'
    if 'business' in r:
        return 'Business decision'
    if 'capacity' in r:
        return 'Capacity'
    return 'Other'


def normalize_ashp_name(name):
    """Clean whitespace, normalize slashes, fix common typos in ASHP names."""
    if pd.isna(name):
        return name
    n = str(name).strip()
    # Collapse multiple spaces
    n = re.sub(r'\s+', ' ', n)
    return n


# Manual alias table: ASHP name (lowered, after normalize) -> panel NONPROPRIETARYNAME (lowered)
# Only for high-value drugs that cannot be matched algorithmically.
ASHP_ALIASES = {
    'albumin injction': 'albumin human',
    'albumin injection': 'albumin human',
    'albumin (human) injection': 'albumin human',
    'albumin': 'albumin human',
    'alchohol, dehydrated': 'dehydrated alcohol',
    'artenusate injection': 'artesunate',
    'betamamethasone injection': 'betamethasone',
    'hydroxocobolamin injection': 'hydroxocobalamin',
    'pencillin v potassium oral presentations': 'penicillin v potassium',
    'ampicillin sulbactam': 'ampicillin and sulbactam',
    'avibactam/ceftazidime': 'ceftazidime, avibactam',
    'buprenorphine/naloxone tablets': 'buprenorphine and naloxone',
    'amitriptyline/chlordiazepoxide tablets': 'chlordiazepoxide and amitriptyline hydrochloride',
    'amitriptyline/perphenazine tablets': 'perphenazine and amitriptyline hydrochloride',
    'carbidopa/levodopa odt': 'carbidopa and levodopa',
    'carbidopa/levodopa/entacapone': 'carbidopa, levodopa and entacapone',
    'losartan / hydrochlorothiazide tablets': 'losartan potassium and hydrochlorothiazide',
    'valsartan/hydrochlorothiazide tablets': 'valsartan and hydrochlorothiazide',
    'sulfamethoxazole/trimethoprim oral suspension': 'sulfamethoxazole and trimethoprim',
    'dorzolamide and timolol ophthalmic solution': 'dorzolamide hydrochloride and timolol maleate',
    'amphetamine mixed salt extended release capsules': 'dextroamphetamine saccharate, amphetamine aspartate monohydrate, dextroamphetamine sulfate, and amphetamine sulfate',
    'amphetamine mixed salt immediate release tablets': 'dextroamphetamine saccharate, amphetamine aspartate monohydrate, dextroamphetamine sulfate and amphetamine sulfate',
    'amphetamines extended-release capsules': 'dextroamphetamine saccharate, amphetamine aspartate monohydrate, dextroamphetamine sulfate, and amphetamine sulfate',
    'lactated ringer\'s': 'lactated ringer\'s',
    'lactated ringer\'s injection': 'lactated ringer\'s',
    'lactated ringer\'s irrigation': 'lactated ringer\'s',
    'sterile water for injection - large volume': 'water',
    'sterile water for injection, large volume bags': 'water',
    'bacteriostatic water for injection': 'bacteriostatic water',
    'ceftolozane/tazobactam vials': 'ceftolozane and tazobactam',
    'dalfopristin/quinupristin injection': 'quinupristin and dalfopristin',
    'oseltamivir capsules and suspension': 'oseltamivir phosphate',
    'insulin, regular': 'insulin human',
    'epinephrine injection': 'epinephrine',
    'epinephrine auto-injectors': 'epinephrine',
    'propofol injection': 'propofol',
    'fentanyl injection': 'fentanyl citrate',
    'fentanyl transdermal patch': 'fentanyl',
    'midazolam injection': 'midazolam',
    'norepinephrine injection': 'norepinephrine bitartrate',
    'potassium bicarbonate': 'potassium bicarbonate',
    'sodium phosphate injection': 'sodium phosphates',
    'sodium hypochlorite': 'sodium hypochlorite',
    'sodium hypochlorite solution': 'sodium hypochlorite',
    'fat emulsion': 'fat emulsion',
    'thyroid, desiccated': 'thyroid',
    'etomidate': 'etomidate',
    'loxapine': 'loxapine',
    'epoprostenol': 'epoprostenol sodium',
    'ceftazidime': 'ceftazidime',
    'conivaptan': 'conivaptan hydrochloride',
    'moxifloxacin': 'moxifloxacin hydrochloride',
    'lanthanum carbonate': 'lanthanum carbonate',
    'isosorbide mononitrate': 'isosorbide mononitrate',
    'sodium ferrous gluconate': 'sodium ferric gluconate complex',
    'tacrolimus injection': 'tacrolimus',
    'rifabutin 150 mg capsules': 'rifabutin',
    'calcium gluconate injection': 'calcium gluconate',
    'doxorubicin liposomal injection': 'doxorubicin hydrochloride',
    'neomycin sulfate tablets': 'neomycin sulfate',
    'levothyroxine tablets': 'levothyroxine sodium',
    'methylprednisolone injection': 'methylprednisolone sodium succinate',
    'nystatin suspension': 'nystatin',
    'magnesium citrate oral solution': 'magnesium citrate',
    'pimecrolimus 1% cream': 'pimecrolimus',
    'sennosides 8.6 mg tablets': 'sennosides',
    'leuprolide depot': 'leuprolide acetate',
    # Injectables with trailing space or variant naming
    'aminophylline injection': 'aminophylline',
    'amphotericin b injection': 'amphotericin b',
    'amphotericin b lipid complex': 'amphotericin b',
    'ampicillin injection': 'ampicillin sodium',
    'azathioprine injection': 'azathioprine sodium',
    'azithromycin injection': 'azithromycin',
    'benztropine injection': 'benztropine mesylate',
    'betamethasone injection': 'betamethasone acetate and betamethasone sodium phosphate',
    'bezlotoxumab injection': 'bezlotoxumab',
    'bivalirudin injection': 'bivalirudin',
    'buprenorphine injection': 'buprenorphine',
    'butorphanol injection': 'butorphanol tartrate',
    'cefotaxime injection': 'cefotaxime sodium',
    'cefuroxime injection': 'cefuroxime',
    'chlorothiazide injection': 'chlorothiazide sodium',
    'clonidine injection': 'clonidine hydrochloride',
    'cytarabine injection': 'cytarabine',
    'diazepam injection': 'diazepam',
    'dimercaprol injection': 'dimercaprol',
    'doxycycline injection': 'doxycycline hyclate',
    'erythromycin injection': 'erythromycin lactobionate',
    'esmolol injection': 'esmolol hydrochloride',
    'estradiol cypionate injection': 'estradiol cypionate',
    'famotidine injection': 'famotidine',
    'fluconazole injection': 'fluconazole',
    'flumazenil injection': 'flumazenil',
    'fluorescein sodium injection': 'fluorescein sodium',
    'fluorouracil injection': 'fluorouracil',
    'floxuridine injection': 'floxuridine',
    'furosemide injection': 'furosemide',
    'gentamicin injection': 'gentamicin sulfate',
    'granisetron injection': 'granisetron hydrochloride',
    'hydralazine injection': 'hydralazine hydrochloride',
    'ibutilide injection': 'ibutilide fumarate',
    'iron dextran injection': 'iron dextran',
    'isoniazid injection': 'isoniazid',
    'isoproterenol injection': 'isoproterenol hydrochloride',
    'ketorolac ophthalmic solution': 'ketorolac tromethamine',
    'levetiracetam injection': 'levetiracetam',
    'linezolid 2 mg/ml premixed bags': 'linezolid',
    'meperidine injection': 'meperidine hydrochloride',
    'mepivacaine injection': 'mepivacaine hydrochloride',
    'mesna injection': 'mesna',
    'methocarbamol injection': 'methocarbamol',
    'metoclopramide injection': 'metoclopramide',
    'metoprolol injection': 'metoprolol tartrate',
    'metronidazole injection': 'metronidazole',
    'mitomycin injection': 'mitomycin',
    'nafcillin sodium injection': 'nafcillin sodium',
    'nicardipine injection': 'nicardipine hydrochloride',
    'norepinephrine injection': 'norepinephrine bitartrate',
    'orphenadrine injection': 'orphenadrine citrate',
    'oxaliplatin injection': 'oxaliplatin',
    'penicillin g benzathine injection': 'penicillin g benzathine',
    'physostigmine injection': 'physostigmine salicylate',
    'phytonadione injection': 'phytonadione',
    'propofol injection': 'propofol',
    'protamine injection': 'protamine sulfate',
    'pyridoxine injection': 'pyridoxine hydrochloride',
    'rifampin injection': 'rifampin',
    'ropivacaine injection': 'ropivacaine hydrochloride',
    'somatropin injection': 'somatropin',
    'tacrolimus injection': 'tacrolimus',
    'terbutaline injection': 'terbutaline sulfate',
    'thiotepa injection': 'thiotepa',
    'topotecan injection': 'topotecan',
    'tranexamic acid injection': 'tranexamic acid',
    'trimethobenzamide injection': 'trimethobenzamide hydrochloride',
    'valproate sodium injection': 'valproate sodium',
    'valproic acid injection': 'valproic acid',
    'vinblastine injection': 'vinblastine sulfate',
    'vincristine injection': 'vincristine sulfate',
    'testosterone cypionate im injection': 'testosterone cypionate',
    # Oral formulations
    'buspirone tablets': 'buspirone hydrochloride',
    'clorazepate tablets': 'clorazepate dipotassium',
    'atenolol tablets': 'atenolol',
    'methyldopa tablets': 'methyldopa',
    'doxycycline hyclate tablets and capsules': 'doxycycline hyclate',
    'erythromycin ethylsuccinate suspension': 'erythromycin ethylsuccinate',
    'erythromycin ethylsuccinate tablets': 'erythromycin ethylsuccinate',
    'clonazepam odt tablets': 'clonazepam',
    'phenytoin extended-release capsules': 'phenytoin sodium',
    'phenytoin oral suspension': 'phenytoin',
    'theophylline er 12-hour tablets': 'theophylline',
    'theophylline extended release 24 hour capsules': 'theophylline',
    'levetiracetam extended-release tablets': 'levetiracetam',
    'prednisolone oral disintegrating tablets': 'prednisolone',
    'mycophenolate capsules and tablets': 'mycophenolate mofetil',
    'megestrol acetate tablets': 'megestrol acetate',
    'tacrolimus extended-release capsules and tablets': 'tacrolimus',
    'rifepentine tablets': 'rifapentine',
    'tedizolid tablets': 'tedizolid phosphate',
    'calcium acetate capsules and tablets': 'calcium acetate',
    'docusate sodium 250 mg capsules': 'docusate sodium',
    'cyclosporine, modified 50 mg capsules': 'cyclosporine',
    'midostaurin capsules': 'midostaurin',
    'sulfadiazine tablets': 'sulfadiazine',
    'levonorgestrel 1.5 mg oral tablet': 'levonorgestrel',
    'aspirin 325 enteric coated tablet': 'aspirin',
    'ranitidine oral capsules and tablets': 'ranitidine',
    # Solutions/suspensions
    'albuterol inhalation solution 0.083%': 'albuterol sulfate',
    'albuterol metered-dose inhalers': 'albuterol sulfate',
    'cromolyn oral solution': 'cromolyn sodium',
    'dexamethasone oral solution/elixir': 'dexamethasone',
    'diazoxide oral suspension': 'diazoxide',
    'diphenoxylate and atropine oral solution': 'diphenoxylate hydrochloride and atropine sulfate',
    'hydroxocobalamin intramuscular injection': 'hydroxocobalamin',
    'hydroxocobalamin solution for intramuscular injection': 'hydroxocobalamin',
    'promethazine oral liquid': 'promethazine hydrochloride',
    'propranolol oral solution': 'propranolol hydrochloride',
    'propranolol oral solution 500 mL bottles': 'propranolol hydrochloride',
    'risperidone oral solution': 'risperidone',
    'ritonavir oral solution': 'ritonavir',
    'sertraline hydrochloride oral solution': 'sertraline hydrochloride',
    'simethicone oral liquid': 'simethicone',
    'nystatin suspension': 'nystatin',
    'docusate sodium oral liquid bulk bottles': 'docusate sodium',
    # Ophthalmic/topical
    'atropine ophthalmic ointment': 'atropine sulfate',
    'atropine ophthalmic solution': 'atropine sulfate',
    'ciprofloxacin 0.3% ophthalmic': 'ciprofloxacin hydrochloride',
    'gentamicin ophthalmic solution': 'gentamicin sulfate',
    'bacitracin zinc ointment': 'bacitracin zinc',
    'pimecrolimus 1% cream': 'pimecrolimus',
    # IV fluids
    '10% dextrose injection': '5% dextrose',
    '70% dextrose injection': '50% dextrose',
    '10% dextran (dextran 40) injection': 'dextran 40',
    'dextran 10% (dextran 40)': 'dextran 40',
    'theophylline in dextrose injection': 'theophylline',
    # Special
    'acetylcholine intraocular solution': 'acetylcholine chloride',
    'acetic acid 0.25% irrigation solution': 'acetic acid',
    'methylphenidate immediate-release': 'methylphenidate hydrochloride',
    'methylphenidate transdermal patch': 'methylphenidate',
    'heparin vials and syringes': 'heparin sodium',
    'heparin premixes': 'heparin sodium',
    'heparin lock flush': 'heparin sodium',
    'heparin lock flush 100 unit/ml 5 ml syringes': 'heparin sodium',
    'bupivacaine plain': 'bupivacaine hydrochloride',
    'bupivacaine with epinephrine': 'bupivacaine hydrochloride and epinephrine',
    'lidocaine and epinephrine': 'lidocaine hydrochloride and epinephrine',
    'lidocaine injection plain (formerly listed as just 2%) - includes mix in dextrose': 'lidocaine hydrochloride',
    'morphine pca vials': 'morphine sulfate',
    'morphine immediate-release tablets': 'morphine sulfate',
    'oxycodone 5 mg/5 ml oral solution unit-dose cups': 'oxycodone hydrochloride',
    'fluphenazine hcl im injection': 'fluphenazine hydrochloride',
    'dexmedetomidine vials': 'dexmedetomidine hydrochloride',
    'dexmedetomidine premixes': 'dexmedetomidine hydrochloride',
    'dexmedetomidine 4 mcg/ml premixes': 'dexmedetomidine hydrochloride',
    'olanzapine extended release suspension for injection': 'olanzapine pamoate',
    'hyoscyamine injection': 'hyoscyamine sulfate',
    'hyoscyamine sulfate injection': 'hyoscyamine sulfate',
    'cephalexin 500 mg capsules and oral suspension': 'cephalexin',
    'cefdinir suspension and capsules': 'cefdinir',
    # Remaining matchable
    'adenosine 2 ml and 4 ml injection': 'adenosine',
    'alprostadil injection and cartridges': 'alprostadil',
    'alprostadil urethral suppositories': 'alprostadil',
    'amobarbital sodium injection': 'amobarbital sodium',
    'carbachol intraocular solution': 'carbachol',
    'colestipol granules for oral suspension': 'colestipol hydrochloride',
    'insulin aspart protamine / insulin aspart mix 70/30': 'insulin aspart protamine and insulin aspart',
    'insulin isophane and regular': 'insulin human',
    'multiple electrolyte additive': 'potassium chloride',
    'peritoneal dialysis solution': 'dextrose monohydrate',
    'psyllium powder unit dose packets': 'psyllium',
    'silver nitrate solution': 'silver nitrate',
    'silver nitrate sticks': 'silver nitrate',
    'sodium tetradecyl sulfate injection': 'sodium tetradecyl sulfate',
    'tuberculin ppd': 'tuberculin purified protein derivative',
    'trypan blue intraocular solution': 'trypan blue',
    'clotrimazole lozenges': 'clotrimazole',
    'cromolyn oral solution': 'cromolyn sodium',
    'neomycin/polymyxin b gu irrigant': 'neomycin sulfate and polymyxin b sulfate, bacitracin zinc and hydrocortisone',
    'neomycin/polymyxin b sulfate gu irrigant': 'neomycin sulfate and polymyxin b sulfate, bacitracin zinc and hydrocortisone',
    'neomycin/polymyxin b/dexamethasone ophthalmic': 'neomycin and polymyxin b sulfates and dexamethasone',
    'sulfacetamide sodium/prednisolone acetate ophthalmic ointment': 'sulfacetamide sodium and prednisolone acetate',
    'fluorescein / benoxinate ophthalmic solution 0.3%/0.4%': 'fluorescein sodium and benoxinate hydrochloride',
    'fluorescein/benoxinate ophthalmic solution': 'fluorescein sodium and benoxinate hydrochloride',
    'fluorescein/proparacaine opthalmic solution': 'fluorescein sodium',
    'pramoxine/hydrocortisone foam': 'pramoxine hydrochloride and hydrocortisone acetate',
    'betamethasone 0.05% augmented ointment': 'betamethasone dipropionate',
    'isosorbide dinitrate sr capsules': 'isosorbide dinitrate',
    'disopyramide controlled-release capsules': 'disopyramide phosphate',
    'divalproex er and dr': 'divalproex sodium',
    'naloxone nasal spray 4 mg/0.1 ml - rx only': 'naloxone hydrochloride',
    'diltiazem cd (ab3 rated) 360 mg capsules': 'diltiazem hydrochloride',
    'aminolevulinic acid powder for oral solution': 'aminolevulinic acid hydrochloride',
    'ribavirin inhalation': 'ribavirin',
    'lanthanum carbonate': 'lanthanum carbonate',
    'tretinoin sublngual capsules': 'tretinoin',
    'zolmitriptan oral and nasal presentations': 'zolmitriptan',
    'timolol gel forming ophthalmic solution': 'timolol maleate',
    'lidocaine 2% jelly tubes and syringes': 'lidocaine hydrochloride',
    'lidocaine 4% topical solution': 'lidocaine hydrochloride',
    'benzocaine 20% spray and ointment products': 'benzocaine',
    'chlorhexidine 0.12% ud': 'chlorhexidine gluconate',
    'budesonide powder for inhalation': 'budesonide',
    'fluocinolone acetonide intravitreal insert': 'fluocinolone acetonide',
    'triamcinolone acetonide intravitreal injection': 'triamcinolone acetonide',
    'triamcinolone acetonide intravitreal': 'triamcinolone acetonide',
    'octreotide intragluteal injection': 'octreotide acetate',
    'immune globulin (hizentra)': 'immune globulin',
    'nirsevimab-alip': 'nirsevimab',
    'zolpidem 6.25 mg extended-release tablets': 'zolpidem tartrate',
    # Round 2 fixes: corrected targets and new matches
    'ciprofloxacin premixed bags': 'ciprofloxacin',
    'progesterone in oil': 'progesterone',
    'corticotropin': 'repository corticotropin',
    'corticotropin injection': 'repository corticotropin',
    'difluprednate ophthalmic emulsion': 'difluprednate',
    'indigo carmine': 'indigotindisulfonate sodium',
    'indigo carmine injection': 'indigotindisulfonate sodium',
    'penicillin g potassium frozen bags': 'penicillin g potassium',
    'penicillin-iv': 'penicillin g potassium',
    'prochlorperazine spansules': 'prochlorperazine',
    'secretin': 'human secretin',
    'cocaine topical soln': 'cocaine hydrochloride',
    'stromectol': 'ivermectin',
    'polyethylene glycol 3350 with electrolytes': 'polyethylene glycol 3350',
    'testosterone in oil': 'testosterone',
    'testosterone subcutaneous implantable pellets': 'testosterone',
    'nystatin 100,000 units/gram powder': 'nystatin',
    'terbutalineinjection': 'terbutaline sulfate',
    'desmopressin rhinal tubes': 'desmopressin acetate',
    'aprepitant intravenous emulsion': 'fosaprepitant dimeglumine',
    'mometasone furoate inhalation aerosol': 'mometasone furoate',
    'nitrogen mustard': 'mechlorethamine hydrochloride',
    'thrombin, bovine': 'thrombin topical recombinant',
    'thrombin, recombinant': 'thrombin topical recombinant',
    'penicillin g procaine/penicillin g benzathine': 'penicillin g benzathine and penicillin g procaine',
    'fluorescein ophthalmic strips': 'fluorescein sodium',
    'fluorescein strips': 'fluorescein sodium',
    '5% dextrose dehp-free': '5% dextrose',
    '5% dextrose pvc/dehp-free bags': '5% dextrose',
    '0.45% sodium chloride large volume injection': 'sodium chloride',
    '0.9% sodium chloride injection 250 ml and larger bags': 'sodium chloride',
    '14.6% sodium chloride injection': 'sodium chloride',
    '23.4% sodium chloride injection': 'sodium chloride',
    'morphine pca syringes': 'morphine sulfate',
    'morphine oral liquid ud cups': 'morphine sulfate',
    'morphine sulfate carpujects': 'morphine sulfate',
    'morphine carpujects': 'morphine sulfate',
    'racepinephrine inhalation solution': 'epinephrine',
    'fluocinolone 0.01% shampoo': 'fluocinolone acetonide',
    'heparin 20,000 unit/ml': 'heparin sodium',
    'heparin 5000 u/ml': 'heparin sodium',
    'multiple electrolyte, large volume': 'sodium chloride',
    'epoprostenol diluent': 'sodium chloride',
    'alprostadil 0.5 mg/ml pediatric 1 ml ampules': 'alprostadil',
    # Round 3 fixes
    'tranexamic acid in sodium chloride injection': 'tranexamic acid',
    'bazedoxifene/conjugated estrogens': 'conjugated estrogens/bazedoxifene',
    'formoterol fumarate/mometasone furoate': 'mometasone furoate and formoterol fumarate dihydrate',
    'cyclopentolate/phenylephrine': 'cyclopentolate hydrochloride and phenylephrine hydrochloride',
    'cyclopentolate/phenylephrine ophthalmic solution': 'cyclopentolate hydrochloride and phenylephrine hydrochloride',
    'cyclopentolate / phenylephrine 0.2%/1% ophthalmic solution': 'cyclopentolate hydrochloride and phenylephrine hydrochloride',
    'lutathera lu 177 dotatate injection': 'lutetium lu 177 dotatate',
    'bacitracin zinc ointment': 'bacitracin',
    'teprotumumab-trbw': 'teprotumumab',
    'betamamethasone injection': 'betamethasone acetate and betamethasone sodium phosphate',
    'insulin aspart protamine / insulin aspart mix 70/30': 'insulin aspart',
    'immune globulin, intravenous/sc combos': 'immune globulin',
    'immune globulin sq and with hyaluronidase, recombinant': 'immune globulin',
}


def match_ashp_to_panel(ashp_records, panel_groups):
    """Match ASHP drug names to panel skeleton groups.

    Uses a multi-pass approach:
    0. Manual alias table for known difficult matches
    1. Lightly cleaned name match (only strip brand parentheticals)
    2. Exact match of full ASHP name to NONPROPRIETARYNAME
    3. Base name match with dosage form filtering
    4. Token-overlap match with form filtering

    Args:
        ashp_records: DataFrame with 'drug_name' column
        panel_groups: DataFrame with 'NONPROPRIETARYNAME', 'DOSAGEFORMNAME',
                      and list of ndc_11 per group

    Returns:
        Dict mapping ASHP drug_name -> list of (NONPROPRIETARYNAME, DOSAGEFORMNAME) tuples
    """
    # Build lookup structures
    panel_name_lower = {}  # lower(NONPROPRIETARYNAME) -> list of (name, form) tuples
    for _, row in panel_groups.iterrows():
        key = row['NONPROPRIETARYNAME'].strip().lower()
        panel_name_lower.setdefault(key, []).append(
            (row['NONPROPRIETARYNAME'], row['DOSAGEFORMNAME'])
        )

    # Build token index for fuzzy matching
    token_index = {}  # token -> set of lower(NONPROPRIETARYNAME)
    for name_lower in panel_name_lower:
        for tok in name_lower.split():
            tok_clean = tok.strip(',()')
            if len(tok_clean) > 2:
                token_index.setdefault(tok_clean, set()).add(name_lower)

    ashp_names = ashp_records['drug_name'].dropna().unique()
    matches = {}
    unmatched = []

    for ashp_name_raw in ashp_names:
        ashp_name = normalize_ashp_name(ashp_name_raw)
        name_lower = ashp_name.strip().lower()
        form_hints = extract_form_hints(name_lower)
        base = extract_base_name(ashp_name)

        # Also try slash-to-and for combo names: "A/B tablets" -> "a and b"
        slash_base = None
        if '/' in base:
            slash_base = re.sub(r'\s*/\s*', ' and ', base)

        def filter_by_form(candidates, hints):
            """Filter (name, form) pairs by dosage form hints."""
            if not hints:
                return candidates
            filtered = []
            for name, form in candidates:
                form_upper = form.upper()
                if any(h in form_upper for h in hints):
                    filtered.append((name, form))
            return filtered if filtered else candidates

        # Pass -1: manual alias table
        alias_key = name_lower
        # Also try without parenthetical brand names and trailing strength info
        alias_key_no_brand = re.sub(r'\([^)]*\)', '', alias_key).strip()
        alias_key_no_brand = re.sub(r'\s+', ' ', alias_key_no_brand).strip()
        alias_key_stripped = re.sub(r'\s+\d+\s*(mg|mcg|ml|%)\b.*$', '', alias_key).strip()
        alias_key_stripped_nb = re.sub(r'\s+\d+\s*(mg|mcg|ml|%)\b.*$', '', alias_key_no_brand).strip()
        alias_target = (ASHP_ALIASES.get(alias_key) or ASHP_ALIASES.get(alias_key_no_brand)
                        or ASHP_ALIASES.get(alias_key_stripped) or ASHP_ALIASES.get(alias_key_stripped_nb))
        if alias_target and alias_target in panel_name_lower:
            candidates = panel_name_lower[alias_target]
            filtered = filter_by_form(candidates, form_hints)
            matches[ashp_name_raw] = filtered
            continue

        # Pass 0: lightly cleaned name (only strip parenthetical brand names)
        light_clean = re.sub(r'\([^)]*\)', '', name_lower).strip()
        light_clean = re.sub(r'\s+', ' ', light_clean).strip()
        if light_clean in panel_name_lower:
            candidates = panel_name_lower[light_clean]
            filtered = filter_by_form(candidates, form_hints)
            matches[ashp_name_raw] = filtered
            continue

        # Pass 1: exact full-name match
        if name_lower in panel_name_lower:
            candidates = panel_name_lower[name_lower]
            filtered = filter_by_form(candidates, form_hints)
            matches[ashp_name_raw] = filtered
            continue

        # Pass 2: base name exact match
        if base and base in panel_name_lower:
            candidates = panel_name_lower[base]
            filtered = filter_by_form(candidates, form_hints)
            matches[ashp_name_raw] = filtered
            continue

        # Pass 2-slash: try slash-to-and converted base name
        if slash_base and slash_base in panel_name_lower:
            candidates = panel_name_lower[slash_base]
            filtered = filter_by_form(candidates, form_hints)
            matches[ashp_name_raw] = filtered
            continue

        # Pass 2a: substring containment - ASHP base name is a prefix of a
        # panel name. Handles cases like "albumin" matching "albumin human"
        # or "tramadol" matching "tramadol hydrochloride".
        # Exclude combination products (those containing " and ") unless the
        # ASHP name itself contains " and ".
        if base and len(base) >= 4:
            ashp_is_combo = ' and ' in base
            substr_candidates = []
            for pname, entries in panel_name_lower.items():
                if pname.startswith(base + ' ') or pname == base:
                    if not ashp_is_combo and ' and ' in pname:
                        continue
                    substr_candidates.extend(entries)
            if substr_candidates:
                filtered = filter_by_form(substr_candidates, form_hints)
                matches[ashp_name_raw] = filtered
                continue

        # Pass 2b: try base name without trailing salt forms
        base_no_salt = re.sub(
            r'\s+(hydrochloride|hcl|hci|sodium|sulfate|potassium|acetate|'
            r'phosphate|mesylate|maleate|tartrate|fumarate|besylate|'
            r'succinate|citrate|bromide|chloride|lactate|calcium|'
            r'dihydrate|monohydrate)\s*$',
            '', base, flags=re.IGNORECASE
        ).strip()
        if base_no_salt and base_no_salt != base and base_no_salt in panel_name_lower:
            candidates = panel_name_lower[base_no_salt]
            filtered = filter_by_form(candidates, form_hints)
            matches[ashp_name_raw] = filtered
            continue

        # Pass 2c: try adding common salt forms
        for salt in ['hydrochloride', 'sodium', 'sulfate', 'hcl', 'calcium', 'potassium']:
            with_salt = f"{base} {salt}"
            if with_salt in panel_name_lower:
                candidates = panel_name_lower[with_salt]
                filtered = filter_by_form(candidates, form_hints)
                matches[ashp_name_raw] = filtered
                break
        if ashp_name_raw in matches:
            continue

        # Pass 3: token overlap scoring
        # Exclude very common chemical words that cause false matches
        COMMON_CHEM_WORDS = {
            'sodium', 'chloride', 'acid', 'hydrochloride', 'sulfate',
            'potassium', 'calcium', 'phosphate', 'acetate', 'citrate',
            'oxide', 'carbonate', 'hydroxide', 'bromide', 'nitrate',
            'and', 'for', 'with', 'the', 'monohydrate', 'dihydrate',
        }
        base_tokens = set(base.split()) if base else set(name_lower.split())
        base_tokens = {t.strip(',-/()') for t in base_tokens if len(t.strip(',-/()')) > 2}
        if not base_tokens:
            unmatched.append(ashp_name_raw)
            continue

        # For fuzzy matching, use distinctive tokens (not common chem words)
        distinctive_tokens = base_tokens - COMMON_CHEM_WORDS
        # If all tokens are common, fall back to requiring full set match
        search_tokens = distinctive_tokens if distinctive_tokens else base_tokens
        min_threshold = 0.67 if len(base_tokens) >= 2 else 1.0

        candidate_scores = {}
        for tok in search_tokens:
            for panel_name in token_index.get(tok, []):
                panel_tokens = {t.strip(',()')
                                for t in panel_name.split() if len(t.strip(',()')) > 2}
                overlap = len(base_tokens & panel_tokens)
                denom = max(len(base_tokens), len(panel_tokens))
                score = overlap / denom if denom > 0 else 0
                if panel_name not in candidate_scores or score > candidate_scores[panel_name]:
                    candidate_scores[panel_name] = score

        if candidate_scores:
            best_score = max(candidate_scores.values())
            if best_score >= min_threshold:
                best_names = [n for n, s in candidate_scores.items() if s == best_score]
                all_candidates = []
                for n in best_names:
                    all_candidates.extend(panel_name_lower[n])
                filtered = filter_by_form(all_candidates, form_hints)
                matches[ashp_name_raw] = filtered
                continue

        unmatched.append(ashp_name_raw)

    return matches, unmatched


def main():
    print("=" * 70)
    print("02_build_shortage_outcome.py - Building shortage outcome (ASHP source)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and clean ASHP shortage data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading ASHP/UUDIS shortage data...")
    ashp = pd.read_excel(
        RAW_DATA / "ASHP" / "efox shortages small file through 2025 final.xlsx",
        header=None, skiprows=2
    )
    ashp.columns = [
        'drug_name', 'status', 'ahfs', 'reason', 'yr',
        'date_notified', 'date_resolved', 'sole_source',
        'parenteral', 'controlled_schedule'
    ]
    print(f"  Raw rows: {len(ashp):,}")

    # Clean year
    ashp['yr'] = pd.to_numeric(ashp['yr'], errors='coerce')

    # Parse dates
    ashp['date_notified'] = pd.to_datetime(ashp['date_notified'], errors='coerce')
    ashp['date_resolved'] = pd.to_datetime(ashp['date_resolved'], errors='coerce')

    # Fix bad dates: pre-1990 timestamps are Excel serial number artifacts.
    # Recover from the yr column (set to Jan 1 of that year).
    bad_dt = ashp['date_notified'] < pd.Timestamp('1990-01-01')
    n_bad = bad_dt.sum()
    if n_bad > 0:
        ashp.loc[bad_dt, 'date_notified'] = pd.to_datetime(
            ashp.loc[bad_dt, 'yr'].astype(int).astype(str) + '-01-01', errors='coerce'
        )
        print(f"  Fixed {n_bad} bad date_notified values from yr column")

    # Same for date_resolved
    bad_end = ashp['date_resolved'].notna() & (ashp['date_resolved'] < pd.Timestamp('1990-01-01'))
    n_bad_end = bad_end.sum()
    if n_bad_end > 0:
        ashp.loc[bad_end, 'date_resolved'] = pd.NaT
        print(f"  Dropped {n_bad_end} bad date_resolved values")

    # Clean status
    ashp['status_clean'] = ashp['status'].str.strip().str.lower()
    print(f"  Status distribution:")
    print(f"    {ashp['status_clean'].value_counts().to_dict()}")

    # Clean sole_source: only keep YES/NO, rest is garbled
    ashp['sole_source_clean'] = ashp['sole_source'].str.strip().str.upper()
    ashp.loc[~ashp['sole_source_clean'].isin(['YES', 'NO']), 'sole_source_clean'] = pd.NA

    # Clean parenteral
    ashp['parenteral_clean'] = ashp['parenteral'].str.strip().str.lower()
    ashp.loc[~ashp['parenteral_clean'].isin(['y', 'n']), 'parenteral_clean'] = pd.NA

    # Normalize reasons
    ashp['reason_clean'] = ashp['reason'].apply(normalize_reason)
    print(f"  Reason distribution:")
    print(f"    {ashp['reason_clean'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 2. Determine shortage windows
    # ------------------------------------------------------------------
    print("\n[2/7] Determining shortage windows...")

    ashp['shortage_start_dt'] = ashp['date_notified']

    study_end_ts = pd.Timestamp(f"{STUDY_END}-01") + pd.offsets.MonthEnd(0)

    # Shortage end: active -> right-censored. Resolved rows with no usable
    # resolution date have an unknown duration, but they must not be treated as
    # active through STUDY_END; that creates years of false active-shortage
    # history. Keep the onset and collapse those episodes to the start month.
    ashp['shortage_end_dt'] = ashp['date_resolved'].copy()
    ashp.loc[ashp['status_clean'] == 'active', 'shortage_end_dt'] = pd.NaT
    ashp['shortage_end_imputed'] = 0
    resolved_missing_end = (
        (ashp['status_clean'] == 'resolved')
        & ashp['shortage_end_dt'].isna()
        & ashp['shortage_start_dt'].notna()
    )
    if resolved_missing_end.any():
        print(
            f"  WARNING: {int(resolved_missing_end.sum())} resolved ASHP rows "
            "have no parsed date_resolved; setting end = start month"
        )
        ashp.loc[resolved_missing_end, 'shortage_end_dt'] = ashp.loc[
            resolved_missing_end, 'shortage_start_dt'
        ]
        ashp.loc[resolved_missing_end, 'shortage_end_imputed'] = 1
    ashp['episode_censored'] = (
        ((ashp['status_clean'] == 'active') & ashp['shortage_end_dt'].isna())
        | (ashp['shortage_end_imputed'] == 1)
    ).astype(int)
    ashp['effective_end_dt'] = ashp['shortage_end_dt'].fillna(study_end_ts)

    # Drop rows with no start date
    ashp = ashp.dropna(subset=['shortage_start_dt'])
    print(f"  Rows with valid start date: {len(ashp):,}")

    # Fix end < start
    bad_end = ashp['effective_end_dt'] < ashp['shortage_start_dt']
    if bad_end.any():
        print(f"  WARNING: {bad_end.sum()} rows with end < start, setting end = start")
        ashp.loc[bad_end, 'effective_end_dt'] = ashp.loc[bad_end, 'shortage_start_dt']
        ashp.loc[bad_end & ashp['shortage_end_dt'].notna(), 'shortage_end_dt'] = (
            ashp.loc[bad_end & ashp['shortage_end_dt'].notna(), 'shortage_start_dt']
        )

    # Filter to records that overlap our study period
    study_start_ts = pd.Timestamp(f"{STUDY_START}-01")
    overlaps = (ashp['shortage_start_dt'] <= study_end_ts) & (ashp['effective_end_dt'] >= study_start_ts)
    ashp = ashp[overlaps].copy()
    print(f"  Records overlapping study period ({STUDY_START} to {STUDY_END}): {len(ashp):,}")
    print(f"  Unique drug names: {ashp['drug_name'].nunique():,}")

    # ------------------------------------------------------------------
    # 3. Load panel skeleton and build group lookup
    # ------------------------------------------------------------------
    print("\n[3/7] Loading panel skeleton for name matching...")
    skel = pd.read_parquet(
        INTERMEDIATE / "panel_skeleton.parquet",
        columns=['ndc_11', 'NONPROPRIETARYNAME', 'DOSAGEFORMNAME']
    )
    # Get unique groups
    skel_groups = skel[['NONPROPRIETARYNAME', 'DOSAGEFORMNAME']].drop_duplicates()
    skel_groups = skel_groups.dropna(subset=['NONPROPRIETARYNAME', 'DOSAGEFORMNAME'])
    print(f"  Panel groups: {len(skel_groups):,}")
    print(f"  Panel NDCs: {skel['ndc_11'].nunique():,}")

    # Build NDC lookup: (NONPROPRIETARYNAME, DOSAGEFORMNAME) -> list of ndc_11
    ndc_lookup = skel.groupby(['NONPROPRIETARYNAME', 'DOSAGEFORMNAME'])['ndc_11'].apply(
        lambda x: list(x.unique())
    ).to_dict()

    # ------------------------------------------------------------------
    # 4. Match ASHP names to panel groups
    # ------------------------------------------------------------------
    print("\n[4/7] Matching ASHP drug names to panel groups...")
    match_map, unmatched = match_ashp_to_panel(ashp, skel_groups)

    n_matched = len(match_map)
    n_unmatched = len(unmatched)
    total_ashp_names = ashp['drug_name'].nunique()
    print(f"  Matched: {n_matched} / {total_ashp_names} unique drug names "
          f"({100*n_matched/total_ashp_names:.1f}%)")
    print(f"  Unmatched: {n_unmatched}")
    if unmatched:
        print(f"  Sample unmatched names:")
        for name in sorted(unmatched)[:20]:
            print(f"    {name}")

    # Filter to only matched records
    ashp = ashp[ashp['drug_name'].isin(match_map)].copy()
    print(f"  ASHP records after filtering to matched: {len(ashp):,}")

    # ------------------------------------------------------------------
    # 5. Expand shortage windows to NDC x month
    # ------------------------------------------------------------------
    print("\n[5/7] Expanding shortage windows to NDC x month...")
    year_months = generate_year_months(STUDY_START, STUDY_END)
    ym_starts = pd.to_datetime([f"{ym}-01" for ym in year_months])
    ym_ends = ym_starts + pd.offsets.MonthEnd(0)

    shortage_records = []
    for _, row in ashp.iterrows():
        drug_name = row['drug_name']
        s = row['shortage_start_dt']
        e = row['effective_end_dt']
        reason = row['reason_clean']
        ahfs = str(row['ahfs']).strip() if pd.notna(row['ahfs']) else ''
        sole_source = row.get('sole_source_clean', pd.NA)
        parenteral = row.get('parenteral_clean', pd.NA)

        # Get all matched (name, form) groups
        matched_groups = match_map.get(drug_name, [])
        if not matched_groups:
            continue

        # Get all NDCs across matched groups
        matched_ndcs = set()
        for name, form in matched_groups:
            for ndc in ndc_lookup.get((name, form), []):
                matched_ndcs.add(ndc)

        # Find overlapping months
        for i, ym in enumerate(year_months):
            month_start = ym_starts[i]
            month_end = ym_ends[i]
            if s <= month_end and e >= month_start:
                for ndc in matched_ndcs:
                    shortage_records.append({
                        'ndc_11': ndc,
                        'year_month': ym,
                        'shortage': 1,
                        'reason_for_shortage': reason,
                        'therapeutic_category': ahfs,
                        'shortage_generic_name': drug_name.strip(),
                        'shortage_company': '',
                        'shortage_start_dt': s,
                    })

    shortage_panel = pd.DataFrame(shortage_records)
    print(f"  Expanded shortage NDC-months: {len(shortage_panel):,}")

    if len(shortage_panel) == 0:
        print("  WARNING: No shortage records generated!")
        shortage_panel = pd.DataFrame(columns=[
            'ndc_11', 'year_month', 'shortage', 'shortage_start',
            'reason_for_shortage'
        ])
        shortage_panel.to_parquet(INTERMEDIATE / "shortage_outcome.parquet", index=False)
        return shortage_panel

    # Deduplicate: if same NDC-month has multiple shortage records, keep one
    # (prefer the earliest posting)
    shortage_panel = shortage_panel.sort_values('shortage_start_dt')
    shortage_panel = shortage_panel.drop_duplicates(subset=['ndc_11', 'year_month'], keep='first')
    print(f"  After dedup: {len(shortage_panel):,}")

    # ------------------------------------------------------------------
    # 6. Create shortage_start, shortage_end, duration columns
    # ------------------------------------------------------------------
    print("\n[6/7] Creating shortage onset, end, and duration columns...")

    # shortage_start = 1 for the first month of each distinct shortage episode
    shortage_panel['start_ym'] = shortage_panel['shortage_start_dt'].dt.to_period('M').astype(str)
    shortage_panel['shortage_start'] = (
        shortage_panel['year_month'] == shortage_panel['start_ym']
    ).astype(int)

    # Build episode lookup from ASHP records
    episode_lookup = ashp[
        ['drug_name', 'shortage_start_dt', 'shortage_end_dt', 'effective_end_dt',
         'episode_censored', 'shortage_end_imputed']
    ].drop_duplicates(subset=['drug_name', 'shortage_start_dt']).copy()
    episode_lookup['start_ym'] = episode_lookup['shortage_start_dt'].dt.to_period('M').astype(str)
    episode_lookup['effective_end_ym'] = episode_lookup['effective_end_dt'].dt.to_period('M').astype(str)
    observed_end_mask = episode_lookup['shortage_end_dt'].notna()
    episode_lookup.loc[observed_end_mask, 'end_ym'] = (
        episode_lookup.loc[observed_end_mask, 'shortage_end_dt'].dt.to_period('M').astype(str)
    )
    episode_lookup.loc[~observed_end_mask, 'end_ym'] = pd.NA

    episode_lookup['episode_duration'] = (
        (episode_lookup['effective_end_dt'].dt.to_period('M') -
         episode_lookup['shortage_start_dt'].dt.to_period('M'))
        .apply(lambda x: x.n if hasattr(x, 'n') else 0) + 1
    )

    # Map shortage_generic_name + start_ym to episode metadata
    # Each NDC inherits the episode info from its matched ASHP record
    shortage_panel = shortage_panel.merge(
        episode_lookup[['drug_name', 'start_ym', 'end_ym', 'episode_duration',
                        'episode_censored', 'shortage_end_imputed']],
        left_on=['shortage_generic_name', 'start_ym'],
        right_on=['drug_name', 'start_ym'],
        how='left',
    ).drop(columns=['drug_name'], errors='ignore')

    # shortage_end = 1 only for observed resolution months
    shortage_panel['shortage_end'] = (
        (shortage_panel['year_month'] == shortage_panel['end_ym']) &
        (shortage_panel['episode_censored'].fillna(0) == 0)
    ).astype(int)

    # months_remaining
    shortage_panel['_current_ym_period'] = pd.to_datetime(
        shortage_panel['year_month'] + '-01'
    ).dt.to_period('M')
    shortage_panel['_start_period'] = pd.to_datetime(
        shortage_panel['start_ym'] + '-01'
    ).dt.to_period('M')
    shortage_panel['_months_elapsed'] = (
        shortage_panel['_current_ym_period'] - shortage_panel['_start_period']
    ).apply(lambda x: x.n if hasattr(x, 'n') else 0)
    shortage_panel['months_remaining'] = (
        shortage_panel['episode_duration'] - shortage_panel['_months_elapsed'] - 1
    ).clip(lower=0)
    shortage_panel.loc[
        shortage_panel['episode_censored'].fillna(0) == 1, 'months_remaining'
    ] = np.nan

    # Clean up temp columns
    shortage_panel.drop(
        columns=['_current_ym_period', '_start_period', '_months_elapsed',
                 'end_ym', 'start_ym'],
        inplace=True
    )

    print(f"  Shortage ends: {shortage_panel['shortage_end'].sum():,}")
    print(f"  Imputed-end shortage NDC-months: {int(shortage_panel['shortage_end_imputed'].fillna(0).sum()):,}")
    onset_rows = shortage_panel['shortage_start'] == 1
    print(f"  Mean episode duration: "
          f"{shortage_panel.loc[onset_rows, 'episode_duration'].mean():.1f} months")
    print(f"  Median episode duration: "
          f"{shortage_panel.loc[onset_rows, 'episode_duration'].median():.0f} months")

    # Final columns (same schema as before for downstream compatibility)
    output_cols = [
        'ndc_11', 'year_month', 'shortage', 'shortage_start',
        'shortage_end', 'months_remaining', 'episode_duration',
        'episode_censored', 'shortage_end_imputed',
        'reason_for_shortage', 'therapeutic_category',
        'shortage_generic_name', 'shortage_company'
    ]
    shortage_panel = shortage_panel[output_cols]

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    output_path = INTERMEDIATE / "shortage_outcome.parquet"
    shortage_panel.to_parquet(output_path, index=False)
    print(f"\n[7/7] Saved shortage outcome to {output_path}")
    print(f"  Shape: {shortage_panel.shape}")
    print(f"  Unique NDCs with shortage: {shortage_panel['ndc_11'].nunique():,}")
    print(f"  Shortage NDC-months: {shortage_panel['shortage'].sum():,}")
    print(f"  Shortage onsets: {shortage_panel['shortage_start'].sum():,}")
    print(f"  Shortage ends: {shortage_panel['shortage_end'].sum():,}")

    print("\n  Shortage by year:")
    shortage_panel['year'] = shortage_panel['year_month'].str[:4]
    print(shortage_panel.groupby('year')['shortage'].sum().to_string())

    print("\n  Episode duration distribution:")
    onset_durations = shortage_panel.loc[onset_rows, 'episode_duration']
    print(f"    Mean: {onset_durations.mean():.1f} months")
    print(f"    Median: {onset_durations.median():.0f} months")
    print(f"    Max: {onset_durations.max():.0f} months")
    for pct in [25, 50, 75, 90]:
        print(f"    {pct}th percentile: {onset_durations.quantile(pct/100):.0f} months")

    print("\n  Top reasons for shortage:")
    print(shortage_panel['reason_for_shortage'].value_counts().head(10).to_string())

    print("\n  Matching summary:")
    print(f"    ASHP drug names matched: {n_matched}")
    print(f"    ASHP drug names unmatched: {n_unmatched}")

    print("\nDone!")
    return shortage_panel


if __name__ == "__main__":
    main()
