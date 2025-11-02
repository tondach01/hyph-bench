# languages provided in Wiktionary dump
langs = cs de el es it ms nl pl pt ru tr

# parse Wiktionary dumps into wordlists
process_all: prepare
	$(foreach l,$(langs),rm -f ./data/$(l)/wiktionary/$(l)_*.wlh;)
	$(foreach l,$(langs),python ./scripts/process_dump.py --lang $(l);)

# create translate files for Wiktionary wordlists (which are created at first)
translate_all: process_all
	$(foreach l,$(langs),rm -f ./data/$(l)/wiktionary/$(l)_*.tra;)
	$(foreach l,$(langs),python ./scripts/make_tr.py ./data/$(l)/wiktionary/$(l)_*.wlh;)

# extract data from compressed Wiktionary dump and prepare directory structure
prepare:
	unzip ./data/wikt_dump.zip -d ./data
	$(foreach l,$(langs),mkdir -p ./data/$(l)/wiktionary;)
