# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it ms nl pl pt ru tr
# non-Wiktionary datasets
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc cssk/cshyphen is/hyphenation-is th/orchid de/wortliste

# parse Wiktionary dumps into wordlists
process_wikt: prepare_wikt
	$(foreach l,$(WIKT_LANGS),rm -f ./data/$(l)/wiktionary/*.wlh;)
	$(foreach l,$(WIKT_LANGS),python ./scripts/process_dump.py --lang $(l);)

# create translate files for all available datasets
translate_all: translate_wikt translate_other

# create translate files for Wiktionary wordlists (which are created at first)
translate_wikt: process_wikt
	$(foreach l,$(WIKT_LANGS),rm -f ./data/$(l)/wiktionary/$(l)_*.tra;)
	$(foreach l,$(WIKT_LANGS),python ./scripts/make_tr.py ./data/$(l)/wiktionary/*.wlh;)

# create translate files for non-Wiktionary wordlists
translate_other: prepare_other
	$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*.tra;)
	$(foreach d,$(OTHER_DATASETS),python ./scripts/make_tr.py ./data/$(d)/*.wlh;)

# extract data from compressed Wiktionary dump and prepare directory structure
prepare_wikt:
	unzip ./wikt_dump.zip -d .
	$(foreach l,$(WIKT_LANGS),mkdir -p ./data/$(l)/wiktionary;)

prepare_other:
	python ./scripts/expand_weights.py ./data/cssk/cshyphen/cssk_all_weighted.wlh
