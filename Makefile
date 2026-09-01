# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it nl pl pt ru tr

# non-Wiktionary datasets
# cssk/cshyphen is special as it is weighted
CSSK = cssk/cshyphen
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc is/hyphenation-is th/orchid de/wortliste uk/wiktionary

# cross-validate all datasets
cross_validate_all: translate_all
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/base.in $(d);)
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/cshyphen.in $(d);)
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/wortliste.in $(d);)
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/wortliste8.in $(d);)

# get statistics of all datasets (wiktionary *_dis.wlh files are tracked in git)
stats_all_datasets: disambiguate_other
	@$(foreach d,$(wildcard data/*/*/*_dis.wlh),uv run python -m scripts.statistics -d -t $(d);)

# member name of a language's dump inside wikt_dump.zip
wikt_jsonl = $(if $(filter $(1),pl pt),$(1)_enwiktionary.jsonl,$(1)_wiktionary.jsonl)

# parse Wiktionary dumps into wordlists; extracts each language's dump member
# individually and deletes it after processing, so peak extra disk stays around
# the largest member (~3 GB) instead of the full ~10 GB archive contents.
# Set KEEP_JSONL=1 to keep the extracted dumps for reruns.
process_wikt:
	@mkdir -p ./wikt_dump
	@$(foreach l,$(WIKT_LANGS),\
		{ test -f ./wikt_dump/$(call wikt_jsonl,$(l)) || unzip -o ./wikt_dump.zip $(call wikt_jsonl,$(l)) -d ./wikt_dump; } && \
		uv run python -m scripts.process_dump --lang $(l) && \
		$(if $(KEEP_JSONL),:,rm -f ./wikt_dump/$(call wikt_jsonl,$(l)));)

# create translate files
translate_all: translate_wikt translate_other

# regenerates .tra from the *_dis.wlh files tracked in git; does not need the dump
translate_wikt:
	@$(foreach l,$(wildcard data/*/wiktionary/*_dis.wlh.tra),rm -f $(l);)
	@$(foreach l,$(wildcard data/*/wiktionary/*_dis.wlh),uv run python -m scripts.make_tr $(l);)

translate_other: disambiguate_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*_dis.wlh.tra;)
	@$(foreach d,$(OTHER_DATASETS),uv run python -m scripts.make_tr ./data/$(d)/*_dis.wlh;)
	@rm -f ./data/$(CSSK)/*_expanded.wlh.tra
	@uv run python -m scripts.make_tr ./data/$(CSSK)/*_expanded.wlh


# resolve data ambiguities and long words
disambiguate_all: disambiguate_wikt disambiguate_other

disambiguate_wikt: process_wikt
	@$(foreach d,$(wildcard data/*/wiktionary/*_dis.wlh),rm -f $(d);)
	@$(foreach d,$(WIKT_LANGS),uv run python -m scripts.disambiguate ./data/$(d)/wiktionary/*.wlh;)

disambiguate_other: prepare_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*_dis.wlh;)
	@$(foreach d,$(OTHER_DATASETS),uv run python -m scripts.disambiguate ./data/$(d)/*.wlh;)


# expand weighted cssk/cshyphen dataset
prepare_other:
	@uv run python -m scripts.expand_weights ./data/$(CSSK)/*.wlhw