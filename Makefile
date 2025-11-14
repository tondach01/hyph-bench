# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it ms nl pl pt ru tr

# non-Wiktionary datasets
# cssk/cshyphen is special as it is weighted
CSSK = cssk/cshyphen
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc is/hyphenation-is th/orchid de/wortliste

# cross-validate all datasets
cross_validate_all: translate_all
	@$(foreach d,$(wildcard data/*/*),echo $(d); python ./scripts/train_test.py -v -n 2 $(d);)
	@$(foreach d,$(wildcard data/*/*/*.train),rm -f $(d);)
	@$(foreach d,$(wildcard data/*/*/*.test),rm -f $(d);)

# get statistics of all datasets
stats_all_datasets: process_wikt prepare_other
	@$(foreach d,$(wildcard data/*/*/*_dis.wlh),python ./scripts/statistics.py -d -t $(d);)

# parse Wiktionary dumps into wordlists
process_wikt: prepare_wikt
	@$(foreach d,$(wildcard data/*/wiktionary/*.wlh),rm -f $(d);)
	@$(foreach l,$(WIKT_LANGS),python ./scripts/process_dump.py --lang $(l);)

# create translate files for all available datasets
translate_all: translate_wikt translate_other

# create translate files for Wiktionary wordlists (which are created at first)
translate_wikt: disambiguate_wikt
	@$(foreach l,$(wildcard data/*/wiktionary/*.tra),rm -f $(l);)
	@$(foreach l,$(wildcard data/*/wiktionary/*_dis.wlh),python ./scripts/make_tr.py $(l);)

# create translate files for non-Wiktionary wordlists
translate_other: disambiguate_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*.tra;)
	@$(foreach d,$(OTHER_DATASETS),python ./scripts/make_tr.py ./data/$(d)/*_dis.wlh;)
	@rm -f ./data/$(CSSK)/*.tra
	@python ./scripts/make_tr.py ./data/$(CSSK)/*_dis.wlh

disambiguate_wikt: process_wikt
	@$(foreach d,$(wildcard data/*/wiktionary/*_dis.wlh),rm -f $(d);)
	@$(foreach d,$(WIKT_LANGS),python ./scripts/disambiguate.py ./data/$(d)/wiktionary/*.wlh;)

disambiguate_other: prepare_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*_dis.wlh;)
	@$(foreach d,$(OTHER_DATASETS),python ./scripts/disambiguate.py ./data/$(d)/*.wlh;)

disambiguate_all: disambiguate_wikt disambiguate_other

# extract data from compressed Wiktionary dump and prepare directory structure
prepare_wikt:
	@unzip ./wikt_dump.zip -d .
	@$(foreach l,$(WIKT_LANGS),mkdir -p ./data/$(l)/wiktionary;)

prepare_other:
	@python ./scripts/expand_weights.py ./data/cssk/cshyphen/cssk-all-weighted.wlhw
