# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it ms nl pl pt ru tr

# non-Wiktionary datasets
# cssk/cshyphen is special as it is weighted
CSSK = cssk/cshyphen
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc is/hyphenation-is th/orchid de/wortliste

# cross-validate all datasets
cross_validate_all: translate_all
	@$(foreach d,$(wildcard data/*/*),echo $(d); python ./scripts/train_test.py -t -v -n 10 -p ./profiles/base.in $(d);)
	@$(foreach d,$(wildcard data/*/*),echo $(d); python ./scripts/train_test.py -t -v -n 10 -p ./profiles/cshyphen.in $(d);)
	@$(foreach d,$(wildcard data/*/*),echo $(d); python ./scripts/train_test.py -t -v -n 10 -p ./profiles/wortliste.in $(d);)

# get statistics of all datasets
stats_all_datasets: disambiguate_all
	@$(foreach d,$(wildcard data/*/*/*_dis.wlh),python ./scripts/statistics.py -d -t $(d);)

# parse Wiktionary dumps into wordlists
process_wikt: prepare_wikt
	@$(foreach d,$(wildcard data/*/wiktionary/*.wlh),rm -f $(d);)
	@$(foreach l,$(WIKT_LANGS),python ./scripts/process_dump.py --lang $(l);)

# create translate files
translate_all: translate_wikt translate_other

translate_wikt: disambiguate_wikt
	@$(foreach l,$(wildcard data/*/wiktionary/*.tra),rm -f $(l);)
	@$(foreach l,$(wildcard data/*/wiktionary/*_dis.wlh),python ./scripts/make_tr.py $(l);)

translate_other: disambiguate_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*.tra;)
	@$(foreach d,$(OTHER_DATASETS),python ./scripts/make_tr.py ./data/$(d)/*_dis.wlh;)
	@rm -f ./data/$(CSSK)/*.tra
	@python ./scripts/make_tr.py ./data/$(CSSK)/*_dis.wlh


# resolve data ambiguities and long words
disambiguate_all: disambiguate_wikt disambiguate_other

disambiguate_wikt: process_wikt
	@$(foreach d,$(wildcard data/*/wiktionary/*_dis.wlh),rm -f $(d);)
	@$(foreach d,$(WIKT_LANGS),python ./scripts/disambiguate.py ./data/$(d)/wiktionary/*.wlh;)

disambiguate_other: prepare_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*_dis.wlh;)
	@$(foreach d,$(OTHER_DATASETS),python ./scripts/disambiguate.py ./data/$(d)/*.wlh;)

# extract data from compressed Wiktionary dump and prepare directory structure
prepare_wikt:
	@unzip -o -d . ./wikt_dump.zip
	@$(foreach l,$(WIKT_LANGS),mkdir -p ./data/$(l)/wiktionary;)

# expand weighted cssk/cshyphen dataset
prepare_other:
	@python ./scripts/expand_weights.py ./data/$(CSSK)/*.wlhw
