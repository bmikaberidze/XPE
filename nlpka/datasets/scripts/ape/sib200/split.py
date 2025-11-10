'''
Usage:
    python -m nlpka.datasets.scripts.ape.sib200.split
'''

from hypothesis import target
import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

# ace_Arab, ace_Latn, acm_Arab, acq_Arab, aeb_Arab, afr_Latn, ajp_Arab, aka_Latn, als_Latn, amh_Ethi, apc_Arab, arb_Arab, arb_Latn, ars_Arab, ary_Arab, arz_Arab, asm_Beng, ast_Latn, awa_Deva, ayr_Latn, azb_Arab, azj_Latn, bak_Cyrl, bam_Latn, ban_Latn, bel_Cyrl, bem_Latn, ben_Beng, bho_Deva, bjn_Arab, bjn_Latn, bod_Tibt, bos_Latn, bug_Latn, bul_Cyrl, cat_Latn, ceb_Latn, ces_Latn, cjk_Latn, ckb_Arab, crh_Latn, cym_Latn, dan_Latn, deu_Latn, dik_Latn, dyu_Latn, dzo_Tibt, ell_Grek, eng_Latn, epo_Latn, est_Latn, eus_Latn, ewe_Latn, fao_Latn, fij_Latn, fin_Latn, fon_Latn, fra_Latn, fur_Latn, fuv_Latn, gaz_Latn, gla_Latn, gle_Latn, glg_Latn, grn_Latn, guj_Gujr, hat_Latn, hau_Latn, heb_Hebr, hin_Deva, hne_Deva, hrv_Latn, hun_Latn, hye_Armn, ibo_Latn, ilo_Latn, ind_Latn, isl_Latn, ita_Latn, jav_Latn, jpn_Jpan, kab_Latn, kac_Latn, kam_Latn, kan_Knda, kas_Arab, kas_Deva, kat_Geor, kaz_Cyrl, kbp_Latn, kea_Latn, khk_Cyrl, khm_Khmr, kik_Latn, kin_Latn, kir_Cyrl, kmb_Latn, kmr_Latn, knc_Arab, knc_Latn, kon_Latn, kor_Hang, lao_Laoo, lij_Latn, lim_Latn, lin_Latn, lit_Latn, lmo_Latn, ltg_Latn, ltz_Latn, lua_Latn, lug_Latn, luo_Latn, lus_Latn, lvs_Latn, mag_Deva, mai_Deva, mal_Mlym, mar_Deva, min_Arab, min_Latn, mkd_Cyrl, mlt_Latn, mni_Beng, mos_Latn, mri_Latn, mya_Mymr, nld_Latn, nno_Latn, nob_Latn, npi_Deva, nqo_Nkoo, nso_Latn, nus_Latn, nya_Latn, oci_Latn, ory_Orya, pag_Latn, pan_Guru, pap_Latn, pbt_Arab, pes_Arab, plt_Latn, pol_Latn, por_Latn, prs_Arab, quy_Latn, ron_Latn, run_Latn, rus_Cyrl, sag_Latn, san_Deva, sat_Olck, scn_Latn, shn_Mymr, sin_Sinh, slk_Latn, slv_Latn, smo_Latn, sna_Latn, snd_Arab, som_Latn, sot_Latn, spa_Latn, srd_Latn, srp_Cyrl, ssw_Latn, sun_Latn, swe_Latn, swh_Latn, szl_Latn, tam_Taml, taq_Latn, taq_Tfng, tat_Cyrl, tel_Telu, tgk_Cyrl, tgl_Latn, tha_Thai, tir_Ethi, tpi_Latn, tsn_Latn, tso_Latn, tuk_Latn, tum_Latn, tur_Latn, twi_Latn, tzm_Tfng, uig_Arab, ukr_Cyrl, umb_Latn, urd_Arab, uzn_Latn, vec_Latn, vie_Latn, war_Latn, wol_Latn, xho_Latn, ydd_Hebr, yor_Latn, yue_Hant, zho_Hans, zho_Hant, zsm_Latn, zul_Latn
sib200_langs_1 = [ 'ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'als_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'arb_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gaz_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khk_Cyrl', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kmr_Latn', 'knc_Arab', 'knc_Latn', 'kon_Latn', 'kor_Hang', 'lao_Laoo', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'lvs_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'mlt_Latn', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nqo_Nkoo', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pbt_Arab', 'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Olck', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'taq_Latn', 'taq_Tfng', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zsm_Latn', 'zul_Latn' ]
sib200_langs = [
    'zho_Hant', 'prs_Arab', 'nld_Latn', 'glg_Latn', 'mal_Mlym', 'bos_Latn', 'cat_Latn', 'lit_Latn', 'srp_Cyrl', 'hrv_Latn', 
    'hun_Latn', 'pol_Latn', 'ukr_Cyrl', 'deu_Latn', 'spa_Latn', 'hin_Deva', 'sin_Sinh', 'epo_Latn', 'nno_Latn', 'tgl_Latn', 
    'eng_Latn', 'tur_Latn', 'guj_Gujr', 'rus_Cyrl', 'arb_Arab', 'azj_Latn', 'bul_Cyrl', 'als_Latn', 'fra_Latn', 'isl_Latn', 
    'slk_Latn', 'kat_Geor', 'kor_Hang', 'ron_Latn', 'bel_Cyrl', 'jpn_Jpan',

    'dan_Latn', 'slv_Latn', 'zho_Hans', 'afr_Latn', 'ars_Arab', 'khk_Cyrl', 'ast_Latn', 'ban_Latn', 'khm_Khmr', 'ind_Latn',
    'nob_Latn', 'est_Latn', 'kaz_Cyrl', 'ory_Orya', 'pap_Latn', 'tam_Taml', 'tha_Thai', 'amh_Ethi', 'kan_Knda', 'uzn_Latn',
    'ben_Beng', 'mar_Deva', 'pag_Latn', 'acq_Arab', 'mkd_Cyrl', 'ajp_Arab', 'ell_Grek', 'mai_Deva', 'kmr_Latn', 'oci_Latn',
    'ces_Latn', 'szl_Latn', 'vie_Latn', 'apc_Arab', 'jav_Latn', 'mya_Mymr', 'ary_Arab', 'ltg_Latn', 'bho_Deva', 'ceb_Latn', 
    'knc_Arab', 'pes_Arab', 'snd_Arab', 'swe_Latn',

    'cym_Latn', 'lij_Latn', 'uig_Arab', 'urd_Arab', 'bjn_Arab', 'crh_Latn', 'ilo_Latn', 'min_Latn', 'pan_Guru', 'san_Deva', 'war_Latn','aeb_Arab',
    'fao_Latn', 'fon_Latn', 'lvs_Latn', 'mos_Latn', 'sag_Latn', 'scn_Latn', 'sot_Latn', 'asm_Beng', 'awa_Deva', 'bak_Cyrl', 'eus_Latn', 'gaz_Latn',
    'gle_Latn', 'kas_Deva', 'kir_Cyrl', 'kmb_Latn', 'lim_Latn', 'lin_Latn', 'plt_Latn', 'som_Latn', 'swh_Latn', 'tat_Cyrl', 'twi_Latn', 'yue_Hant',
    'arb_Latn', 'azb_Arab', 'bam_Latn', 'bod_Tibt', 'ewe_Latn', 'hau_Latn', 'heb_Hebr', 'hye_Armn', 'kab_Latn', 'kac_Latn', 'kas_Arab', 'kea_Latn',
    'lmo_Latn', 'ltz_Latn', 'run_Latn', 'acm_Arab', 'aka_Latn', 'ayr_Latn', 'bjn_Latn', 'cjk_Latn', 'ckb_Arab', 'dzo_Tibt', 'fin_Latn', 'fuv_Latn',
    'gla_Latn', 'grn_Latn', 'hne_Deva', 'ibo_Latn', 'ita_Latn', 'kam_Latn', 'kbp_Latn', 'kik_Latn', 'knc_Latn', 'kon_Latn', 'lao_Laoo', 'luo_Latn',
    'lus_Latn', 'mag_Deva', 'min_Arab', 'mni_Beng', 'mri_Latn', 'npi_Deva', 'nya_Latn', 'pbt_Arab', 'por_Latn', 'sat_Olck', 'shn_Mymr', 'smo_Latn',
    'sna_Latn', 'ssw_Latn', 'sun_Latn', 'tel_Telu', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'umb_Latn', 'vec_Latn', 'wol_Latn',
    'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'zsm_Latn', 'zul_Latn', 'ace_Latn', 'arz_Arab', 'bem_Latn', 'bug_Latn', 'dik_Latn', 'dyu_Latn', 'hat_Latn',
    'lua_Latn', 'lug_Latn', 'nqo_Nkoo', 'nus_Latn', 'srd_Latn', 'taq_Latn', 'taq_Tfng', 'tir_Ethi', 'ace_Arab', 'fij_Latn', 'fur_Latn', 'kin_Latn',
    'nso_Latn', 'tgk_Cyrl', 'tzm_Tfng', 'mlt_Latn', 'quy_Latn'
]
if len(sib200_langs) != len(sib200_langs_1):
    raise ValueError("sib200_langs and sib200_langs_1 must have the same length")
if set(sib200_langs) != set(sib200_langs_1):
    raise ValueError("sib200_langs and sib200_langs_1 must contain the same elements, regardless of order")


# # Random split
# import random

# # Randomly select 64 languages from the sib200_langs list
# random.shuffle(sib200_langs)

# # Print the selected 64 languages
# print("Selected 64 Source Languages:")
# print("', '".join(sib200_langs[:64]))

# # Print the remaining languages
# print("\nRemaining 141 Target Languages:")
# print("', '".join(sib200_langs[64:]))

# # Resulting split
# source_langs = [
#     "tam_Taml", "san_Deva", "nno_Latn", "asm_Beng", "arz_Arab", "glg_Latn", "ilo_Latn", "bel_Cyrl", "lmo_Latn", "shn_Mymr", 
#     "bho_Deva", "lim_Latn", "bjn_Latn", "lin_Latn", "afr_Latn", "deu_Latn", "kac_Latn", "nso_Latn", "nqo_Nkoo", "luo_Latn", 
#     "ita_Latn", "sna_Latn", "vec_Latn", "kmr_Latn", "tat_Cyrl", "zho_Hant", "khk_Cyrl", "ceb_Latn", "war_Latn", "ckb_Arab", 
#     "kin_Latn", "fuv_Latn", "scn_Latn", "ayr_Latn", "slk_Latn", "kam_Latn", "mya_Mymr", "acm_Arab", "bul_Cyrl", "quy_Latn", 
#     "bem_Latn", "ron_Latn", "tso_Latn", "fin_Latn", "mal_Mlym", "tpi_Latn", "cjk_Latn", "nus_Latn", "awa_Deva", "srd_Latn", 
#     "ary_Arab", "crh_Latn", "heb_Hebr", "knc_Latn", "som_Latn", "cat_Latn", "mni_Beng", "ewe_Latn", "bam_Latn", "twi_Latn", 
#     "ind_Latn", "zul_Latn", "azb_Arab", "gle_Latn"
# ]
# target_langs = [
#     "fra_Latn", "als_Latn", "uig_Arab", "nya_Latn", "cym_Latn", "lij_Latn", "ars_Arab", "vie_Latn", "grn_Latn", "spa_Latn", 
#     "hun_Latn", "dyu_Latn", "ltz_Latn", "zsm_Latn", "sat_Olck", "smo_Latn", "yue_Hant", "pan_Guru", "jav_Latn", "ces_Latn", 
#     "lus_Latn", "tir_Ethi", "taq_Tfng", "lua_Latn", "tzm_Tfng", "snd_Arab", "ajp_Arab", "bod_Tibt", "khm_Khmr", "arb_Latn", 
#     "azj_Latn", "amh_Ethi", "pbt_Arab", "mri_Latn", "mlt_Latn", "sin_Sinh", "bjn_Arab", "knc_Arab", "hye_Armn", "hrv_Latn", 
#     "sun_Latn", "hin_Deva", "ssw_Latn", "epo_Latn", "kan_Knda", "wol_Latn", "lao_Laoo", "por_Latn", "zho_Hans", "urd_Arab", 
#     "gaz_Latn", "mai_Deva", "pag_Latn", "umb_Latn", "tuk_Latn", "dik_Latn", "bos_Latn", "mar_Deva", "run_Latn", "szl_Latn", 
#     "ast_Latn", "ben_Beng", "bug_Latn", "kab_Latn", "apc_Arab", "kaz_Cyrl", "est_Latn", "hne_Deva", "fon_Latn", "kbp_Latn", 
#     "ace_Latn", "acq_Arab", "ban_Latn", "oci_Latn", "tgk_Cyrl", "mkd_Cyrl", "jpn_Jpan", "tel_Telu", "aeb_Arab", "ydd_Hebr", 
#     "ell_Grek", "hau_Latn", "pap_Latn", "taq_Latn", "eng_Latn", "dzo_Tibt", "plt_Latn", "tum_Latn", "hat_Latn", "uzn_Latn", 
#     "yor_Latn", "ukr_Cyrl", "gla_Latn", "srp_Cyrl", "sag_Latn", "tur_Latn", "npi_Deva", "dan_Latn", "tsn_Latn", "tgl_Latn", 
#     "kon_Latn", "lvs_Latn", "lit_Latn", "pes_Arab", "fao_Latn", "ibo_Latn", "rus_Cyrl", "arb_Arab", "guj_Gujr", "ltg_Latn", 
#     "isl_Latn", "nob_Latn", "kor_Hang", "eus_Latn", "kas_Deva", "kir_Cyrl", "prs_Arab", "pol_Latn", "mag_Deva", "min_Arab", 
#     "ory_Orya", "lug_Latn", "kea_Latn", "kat_Geor", "fij_Latn", "tha_Thai", "nld_Latn", "kmb_Latn", "kas_Arab", "bak_Cyrl", 
#     "fur_Latn", "xho_Latn", "kik_Latn", "slv_Latn", "swe_Latn", "ace_Arab", "swh_Latn", "sot_Latn", "mos_Latn", "min_Latn", 
#     "aka_Latn"
# ]


# SIB200 languages that are in both AYA101 and XLM-R
source_langs = [
    "afr_Latn",
    "amh_Ethi",
    "arb_Arab",
    "azj_Latn",
    "bel_Cyrl",
    "ben_Beng",
    "bul_Cyrl",
    "cat_Latn",
    "ces_Latn",
    "dan_Latn",
    "deu_Latn",
    "eng_Latn",
    "epo_Latn",
    "est_Latn",
    "eus_Latn",
    "fin_Latn",
    "fra_Latn",
    "gle_Latn",
    "glg_Latn",
    "guj_Gujr",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hun_Latn",
    "hye_Armn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kan_Knda",
    "kat_Geor",
    "kaz_Cyrl",
    "khm_Khmr",
    "kor_Hang",
    "lao_Laoo",
    "lit_Latn",
    "mal_Mlym",
    "mar_Deva",
    "mkd_Cyrl",
    "mya_Mymr",
    "nld_Latn",
    "nob_Latn",
    "pan_Guru",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "som_Latn",
    "spa_Latn",
    "srp_Cyrl",
    "sun_Latn",
    "swe_Latn",
    "tam_Taml",
    "tel_Telu",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "uzn_Latn",
    "vie_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "zho_Hans",
    "zho_Hant",
]

target_langs = []

for lang in sib200_langs:
    if lang not in source_langs:
        target_langs.append(lang)

print("\nTarget Languages:")
print("', '".join(source_langs))

print("\nTarget Languages:")
print("', '".join(target_langs))

# Target Languages:
source_langs = [ 
    'afr_Latn', 'amh_Ethi', 'arb_Arab', 'azj_Latn', 'bel_Cyrl', 'ben_Beng', 'bul_Cyrl', 'cat_Latn', 'ces_Latn', 'dan_Latn', 
    'deu_Latn', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'fin_Latn', 'fra_Latn', 'gle_Latn', 'glg_Latn', 'guj_Gujr', 
    'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hun_Latn', 'hye_Armn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 
    'kan_Knda', 'kat_Geor', 'kaz_Cyrl', 'khm_Khmr', 'kor_Hang', 'lao_Laoo', 'lit_Latn', 'mal_Mlym', 'mar_Deva', 'mkd_Cyrl', 
    'mya_Mymr', 'nld_Latn', 'nob_Latn', 'pan_Guru', 'pol_Latn', 'por_Latn', 'ron_Latn', 'rus_Cyrl', 'sin_Sinh', 'slk_Latn', 
    'slv_Latn', 'som_Latn', 'spa_Latn', 'srp_Cyrl', 'sun_Latn', 'swe_Latn', 'tam_Taml', 'tel_Telu', 'tha_Thai', 'tur_Latn', 
    'ukr_Cyrl', 'urd_Arab', 'uzn_Latn', 'vie_Latn', 'xho_Latn', 'ydd_Hebr', 'zho_Hans', 'zho_Hant' ]

# Target Languages:
target_langs =[ 
    'tat_Cyrl', 'twi_Latn', 'dzo_Tibt', 'vec_Latn', 'ayr_Latn', 'lus_Latn', 'lvs_Latn', 'bod_Tibt', 'dik_Latn', 'kon_Latn', 
    'oci_Latn', 'zul_Latn', 'pap_Latn', 'ceb_Latn', 'quy_Latn', 'bug_Latn', 'tpi_Latn', 'bjn_Latn', 'ban_Latn', 'kir_Cyrl', 
    'kbp_Latn', 'mos_Latn', 'luo_Latn', 'mai_Deva', 'hne_Deva', 'ilo_Latn', 'min_Arab', 'kam_Latn', 'ssw_Latn', 'tgl_Latn', 
    'tir_Ethi', 'bho_Deva', 'lug_Latn', 'ckb_Arab', 'fij_Latn', 'smo_Latn', 'cym_Latn', 'uig_Arab', 'pag_Latn', 'als_Latn', 
    'kab_Latn', 'grn_Latn', 'sag_Latn', 'hrv_Latn', 'pbt_Arab', 'swh_Latn', 'kea_Latn', 'acm_Arab', 'pes_Arab', 'aka_Latn', 
    'gla_Latn', 'bam_Latn', 'tum_Latn', 'yor_Latn', 'dyu_Latn', 'taq_Latn', 'tuk_Latn', 'bem_Latn', 'sna_Latn', 'fon_Latn', 
    'knc_Latn', 'ewe_Latn', 'snd_Arab', 'zsm_Latn', 'gaz_Latn', 'min_Latn', 'arb_Latn', 'acq_Arab', 'kas_Deva', 'nya_Latn', 
    'kac_Latn', 'mni_Beng', 'azb_Arab', 'cjk_Latn', 'ltz_Latn', 'taq_Tfng', 'kik_Latn', 'bos_Latn', 'npi_Deva', 'ars_Arab', 
    'umb_Latn', 'ltg_Latn', 'lua_Latn', 'tso_Latn', 'hat_Latn', 'mri_Latn', 'mlt_Latn', 'fao_Latn', 'scn_Latn', 'lin_Latn', 
    'ory_Orya', 'tsn_Latn', 'mag_Deva', 'yue_Hant', 'apc_Arab', 'asm_Beng', 'tzm_Tfng', 'crh_Latn', 'ace_Arab', 'lim_Latn', 
    'kas_Arab', 'run_Latn', 'sat_Olck', 'nno_Latn', 'ace_Latn', 'lij_Latn', 'ary_Arab', 'san_Deva', 'prs_Arab', 'arz_Arab', 
    'awa_Deva', 'tgk_Cyrl', 'aeb_Arab', 'ast_Latn', 'fuv_Latn', 'sot_Latn', 'lmo_Latn', 'ell_Grek', 'kmr_Latn', 'nqo_Nkoo', 
    'kin_Latn', 'bak_Cyrl', 'khk_Cyrl', 'ibo_Latn', 'war_Latn', 'ajp_Arab', 'bjn_Arab', 'knc_Arab', 'szl_Latn', 'kmb_Latn', 
    'wol_Latn', 'nso_Latn', 'nus_Latn', 'plt_Latn', 'fur_Latn', 'srd_Latn', 'shn_Mymr'
]
