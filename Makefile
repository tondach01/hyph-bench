# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it ms nl pl pt ru tr
# non-Wiktionary datasets
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc cssk/cshyphen is/hyphenation-is th/orchid de/wortliste

# get statistics of all datasets
stats_all_datasets: #process_wikt prepare_other
	@python ./scripts/statistics.py -d --header
	@$(foreach d,$(wildcard data/*/*/*.wlh),python ./scripts/statistics.py -d -t --file $(d);)

# parse Wiktionary dumps into wordlists
process_wikt: prepare_wikt
	$(foreach d,$(wildcard data/*/wiktionary/*.wlh),rm -f $(d);)
	$(foreach l,$(WIKT_LANGS),python ./scripts/process_dump.py --lang $(l);)

# create translate files for all available datasets
translate_all: translate_wikt translate_other

# create translate files for Wiktionary wordlists (which are created at first)
translate_wikt: process_wikt
	$(foreach l,$(wildcard data/*/wiktionary/*.wlh.tra),rm -f $(l);)
	$(foreach l,$(wildcard data/*/wiktionary/*.wlh),python ./scripts/make_tr.py $(l);)

# create translate files for non-Wiktionary wordlists
translate_other: prepare_other
	$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*.tra;)
	$(foreach d,$(OTHER_DATASETS),python ./scripts/make_tr.py ./data/$(d)/*.wlh;)

# extract data from compressed Wiktionary dump and prepare directory structure
prepare_wikt:
	unzip ./wikt_dump.zip -d .
	$(foreach l,$(WIKT_LANGS),mkdir -p ./data/$(l)/wiktionary;)

prepare_other:
	python ./scripts/expand_weights.py ./data/cssk/cshyphen/cssk-all-weighted.wlhw
